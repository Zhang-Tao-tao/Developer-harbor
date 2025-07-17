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


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        lambda_=0.1,
        args_solver: dict = {},
    ):
        super().__init__()

        self.lambda_ = lambda_
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

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

        # x = torch.where(x > 0.5, torch.ones_like(x), torch.zeros_like(x))
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


