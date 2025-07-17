import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from lqp_py.solve_box_qp_admm_torch import SolveBoxQP
from lqp_py.control import box_qp_control
import line_profiler
from icecream import ic
import torch.multiprocessing as mp
import kaiwu as kw
import numpy as np
from tqdm import tqdm
import pandas as pd


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class SASolver:
    def __init__(self, args_solver: dict):
        assert "user_id" in args_solver, "args_solver must contain 'user_id'"
        assert "sdk_code" in args_solver, "args_solver must contain 'sdk_code'"

        solver_args = {
            "num_process": 1,
            "is_check": True,
            "sa_num_process": 1,
            "initial_temperature": 100,
            "alpha": 0.99,
            "cutoff_temperature": 1e-3,
            "iterations_per_t": 10,
            "size_limit": 10,
            "rand_seed": None,
            "timeout": 10,
        }
        solver_args.update(args_solver)
        self.args_solver = solver_args

        kw.license.init(self.args_solver["user_id"], self.args_solver["sdk_code"])

    def forward(self, Q, p, A, b, lb, ub):
        p = p.squeeze(-1)
        if self.args_solver["is_check"]:
            assert Q.shape[-1] == Q.shape[-2], "Q must be a square matrix"
            assert p.shape[-1] == Q.shape[-1], "p must match the last dimension of Q"
            assert A is None, "no constrain support"
            assert b is None, "no constrain support"

        device = Q.device
        dtype = Q.dtype
        Q = Q.cpu().detach().numpy()
        p = p.cpu().detach().numpy()
        batch_size = Q.shape[0]
        print(batch_size)
        result = []
        for b in tqdm(range(batch_size), desc="Solving...", leave=False):
            result.append(self._solve_step(Q[b, :, :], p[b, :], b))
        exit(1)
        results = np.stack(result, axis=0)
        results = torch.from_numpy(results).to(device, dtype=dtype)
        return results

    def _solve_step(self, Q, p, idx):
        n = Q.shape[0]

        qubo_model = kw.qubo.QuboModel()
        x = kw.qubo.ndarray((n, 1), "x", kw.qubo.Binary)

        objective = (0.5 * x.T @ Q @ x + x.T @ p).item()
        qubo_model.set_objective(objective)

        qubo_mat = qubo_model.get_qubo_matrix(bit_width=8)
        pd.DataFrame(qubo_mat).to_csv(f"./mat/{idx}.csv", index=False, header=False)

        # solve
        # worker = kw.solver.SimpleSolver(
        #     kw.classical.SimulatedAnnealingOptimizer(
        #         initial_temperature=self.args_solver["initial_temperature"],
        #         alpha=self.args_solver["alpha"],
        #         cutoff_temperature=self.args_solver["cutoff_temperature"],
        #         iterations_per_t=self.args_solver["iterations_per_t"],
        #         size_limit=self.args_solver["size_limit"],
        #         rand_seed=self.args_solver["rand_seed"],
        #         process_num=self.args_solver["sa_num_process"],
        #     )
        # )
        # sol_dict, _ = worker.solve_qubo(qubo_model)
        # x = kw.qubo.get_array_val(x, sol_dict)
        # return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        lambda_=0.1,
        solver="admm",
        args_solver: dict = {},
    ):
        super().__init__()

        self.lambda_ = lambda_
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        if solver == "admm":
            solver_args = {
                "eps_rel": 1e-6,
                "eps_abs": 1e-6,
                "verbose": False,
                "reduce": "max",
            }  # , 'max_iters': 1000}
            solver_args.update(args_solver)
            control = box_qp_control(
                eps_rel=solver_args["eps_rel"],
                eps_abs=solver_args["eps_abs"],
                verbose=solver_args["verbose"],
                reduce=solver_args["reduce"],
            )
            self.solver = SolveBoxQP(control=control)
        elif solver == "kaiwu_sa":
            self.solver = SASolver(args_solver)
        else:
            raise ValueError(f"Unsupported solver: {solver}")

    def forward(self, Q, V):
        b, n, _ = Q.shape
        _, m, _ = V.shape
        V = V / m

        Q1 = 2 * torch.matmul(V, V.transpose(1, 2))
        P = -2 * torch.einsum("bmd,bnd->bnm", V, Q).unsqueeze(-1)
        lambda_term = self.lambda_ * torch.ones(b, 1, m, 1, device=P.device) / m
        P = P + lambda_term

        Q1_stacked = Q1.unsqueeze(1).expand(-1, n, -1, -1).reshape(b * n, m, m)
        P_stacked = P.reshape(b * n, m, 1)
        lb = torch.zeros(b, m, 1, device=P.device)
        ub = torch.ones(b, m, 1, device=P.device)
        lb_stacked = lb.unsqueeze(1).expand(-1, n, -1, -1).reshape(b * n, m, 1)
        ub_stacked = ub.unsqueeze(1).expand(-1, n, -1, -1).reshape(b * n, m, 1)

        x_stacked = self.solver.forward(
            Q1_stacked, P_stacked, None, None, lb_stacked, ub_stacked
        )

        x = x_stacked.reshape(b, n, m, 1)

        x = torch.where(x > 0.5, torch.ones_like(x), torch.zeros_like(x))
        sparse_coeffs = x.squeeze(-1)

        sparse_coeffs = sparse_coeffs / (
            torch.abs(sparse_coeffs).sum(dim=-1, keepdim=True) + 1e-10
        )

        out = torch.matmul(sparse_coeffs, V)

        return out


class MLP(nn.Module):
    def __init__(self, dim, num_classes):

        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


class OptimAttn(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim,
        channels=1,
        solver="admm",
        args_solver: dict = {},
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=patch_height,
                pw=patch_width,
            ),
            nn.LayerNorm(patch_dim),
        )

        self.Wq = nn.Sequential(
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.Wv = nn.Sequential(
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, patch_dim))

        self.attention = Attention(
            dim,
            lambda_=1,
            solver=solver,
            args_solver=args_solver,
        )
        self.mlp = MLP(dim, num_classes)

    # @line_profiler.profile
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]

        Q = self.Wq(x)
        V = self.Wv(x)
        x = self.attention(Q, V)

        x = x.mean(dim=1)
        x = self.mlp(x)

        return x


if __name__ == "__main__":
    torch.set_printoptions(edgeitems=1000)
    model = OptimAttn(
        image_size=(28, 28),
        patch_size=(4, 4),
        num_classes=10,
        dim=64,
        channels=1,
        solver="kaiwu_sa",
        args_solver={
            "user_id": "69878024601862146",
            "sdk_code": "0i4T6LY1XygfwN3MWa8Fjq27OaT0sq",
        },
    ).to("cuda")
    x = torch.randn(128, 1, 28, 28).to("cuda")
    out = model(x)
    print(out.shape)
