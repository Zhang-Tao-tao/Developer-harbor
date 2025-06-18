from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
import numpy as np
import json

all_x = np.zeros((7, 7, 7, 7), dtype=np.float32)
for idx0 in range(7):
    for idx1 in range(7):
        idx = idx0 * 7 + idx1
        data = json.load(
            open(f"./cim/QBosonCIMResult/OptimizingAttention{idx}.log", "r")
        )
        data = data[0]
        assert data["result"] == 0
        x = data["solutionVector"]
        x = np.array(x).reshape(7, 7)
        all_x[idx0, idx1, :, :] = x

fig = plt.figure(figsize=(14, 14))
gs = GridSpec(7, 7, figure=fig, wspace=0.1, hspace=0.2)


for idx0 in range(7):
    for idx1 in range(7):
        ax = fig.add_subplot(gs[idx0, idx1])
        ax.imshow(
            np.flipud(all_x[idx0, idx1]),
            cmap=ListedColormap(["#241942", "#e49f1b"]),
            interpolation="nearest",
        )

        ax.text(
            0.5,
            -0.1,
            f"Query-{idx0 * 7 + idx1}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )

        for i in range(7):
            for j in range(7):
                ax.text(
                    j,
                    6 - i,
                    f"V{i * 7 + j}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=6,
                )

        for i in range(7):
            ax.axhline(y=i - 0.5, color="green", linewidth=0.5)
            ax.axvline(x=i - 0.5, color="green", linewidth=0.5)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.5)

        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, 6.5)
        ax.set_xticks([])
        ax.set_yticks([])

        # axs[idx0, idx1].set_title(f'OptimizingAttention{idx0 * 7 + idx1}')
        # axs[idx0, idx1].axis('off')
# plt.tight_layout()
# plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

plt.savefig("OptimizingAttentionHeatmaps.png", dpi=600, bbox_inches="tight")
# plt.show()
