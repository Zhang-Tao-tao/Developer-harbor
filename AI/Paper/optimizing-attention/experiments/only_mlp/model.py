import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from icecream import ic


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


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
        
        self.mlp = nn.Sequential(
            nn.Linear(patch_dim, dim),
            nn.ReLU(),
        )
        self.softmax = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.Softmax(dim=1),
        )

    # @line_profiler.profile
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x = self.mlp(x)
        x = torch.mean(x, dim=1) 
        x = self.softmax(x)

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
