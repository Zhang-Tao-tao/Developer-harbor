import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from icecream import ic


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        # Compute attention scores
        attn_scores = torch.einsum("bnd,bmd->bnm", Q, K)
        attn_scores = attn_scores / (Q.shape[-1] ** 0.5)  # Scale scores
        attn_weights = F.softmax(
            attn_scores, dim=-1
        )  # Apply softmax to get attention weights
        # Compute the output as a weighted sum of values
        out = torch.einsum("bnm,bmd->bnd", attn_weights, V)
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


class Net(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim,
        channels=1,
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

        self.Wk = nn.Sequential(
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, patch_dim))

        self.attention = Attention()
        self.mlp = MLP(dim, num_classes)

    # @line_profiler.profile
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]

        Q = self.Wq(x)
        V = self.Wv(x)
        K = self.Wk(x)
        x = self.attention(Q, K, V)

        x = x.mean(dim=1)
        x = self.mlp(x)

        return x


if __name__ == "__main__":
    torch.set_printoptions(edgeitems=1000)
    model = Net(
        image_size=(28, 28),
        patch_size=(4, 4),
        num_classes=10,
        dim=64,
        channels=1,
    ).to("cuda")
    x = torch.randn(128, 1, 28, 28).to("cuda")
    out = model(x)
    print(out.shape)
