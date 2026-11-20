"""Figures for the README: validation precision-recall curve and test confusion matrix."""
import os

# Only PNG files are written. Inside Colab/Jupyter, MPLBACKEND points to the notebook
# backend, which a script cannot load, so always use the file-only "Agg" backend.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def pr_curve(recall, precision, chosen_recall, chosen_precision, threshold, file, title):
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=130, facecolor=SURFACE)
    _style(ax)
    ax.plot(recall, precision, color=SERIES_1, linewidth=2)
    ax.scatter([chosen_recall], [chosen_precision], s=64, color=SERIES_1,
               edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"chosen threshold {threshold:.2f}\nP {chosen_precision:.2f}, R {chosen_recall:.2f}",
                (chosen_recall, chosen_precision), xytext=(-12, -34), textcoords="offset points",
                ha="right", fontsize=9, color=TEXT_2)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (profanity class)", color=TEXT, fontsize=10)
    ax.set_ylabel("Precision (profanity class)", color=TEXT, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(file, facecolor=SURFACE)
    plt.close(fig)


def confusion_matrix(cm, file, title):
    """cm = [[TN, FP], [FN, TP]]: rows are the true label, columns the prediction."""
    cm = np.asarray(cm)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=130, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    row_share = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    ax.imshow(row_share, cmap=cmap, vmin=0, vmax=1)
    names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            dark = row_share[i, j] > 0.55
            ax.text(j, i, f"{cm[i, j]}\n{names[i][j]}  {row_share[i, j]:.0%} of row", ha="center",
                    va="center", fontsize=10, color="#ffffff" if dark else TEXT)
    ax.set_xticks([0, 1], ["0  no profanity", "1  profanity"], color=TEXT_2, fontsize=9)
    ax.set_yticks([0, 1], ["0  no profanity", "1  profanity"], color=TEXT_2, fontsize=9)
    ax.set_xlabel("Predicted label", color=TEXT, fontsize=10)
    ax.set_ylabel("True label", color=TEXT, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.axhline(0.5, color=SURFACE, linewidth=2)       # thin gap between the cells
    ax.axvline(0.5, color=SURFACE, linewidth=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(file, facecolor=SURFACE)
    plt.close(fig)
