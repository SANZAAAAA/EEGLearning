"""绘图工具 (英文标签, 避免中文字体缺失问题)。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")           # 无窗口环境
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_training_curves(histories: list[dict], path: Path, title: str = "") -> None:
    """每个折的 loss / accuracy 曲线。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

    for i, hist in enumerate(histories):
        label = f"fold {i + 1}" if len(histories) > 1 else "run"
        axes[0].plot(hist["train_loss"], lw=1, alpha=0.7, label=f"{label} train")
        axes[1].plot(hist["train_acc"], lw=1, alpha=0.7, label=f"{label} train")
        if hist.get("val_loss"):
            axes[0].plot(hist["val_loss"], lw=1.2, alpha=0.9, label=f"{label} val")
            axes[1].plot(hist["val_acc"], lw=1.2, alpha=0.9, label=f"{label} val")

    axes[0].set_xlabel("epoch"), axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Loss")
    axes[1].set_xlabel("epoch"), axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(0.0, 1.02)
    if len(histories) <= 3:
        axes[0].legend(fontsize=6), axes[1].legend(fontsize=6)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, path: Path, title: str = "Confusion matrix") -> None:
    cm = np.asarray(cm, dtype=float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = cm / np.maximum(row_sum, 1)

    fig, ax = plt.subplots(figsize=(3.6 + 0.4 * len(class_names), 3.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=20, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted"), ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_subject_summary(df: "pd.DataFrame", path: Path) -> None:
    """每个被试的准确率 + 跨被试均值。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(df) + 2), 3.4))
    x = np.arange(len(df))
    err = df["fold_accuracy_std"].to_numpy() if "fold_accuracy_std" in df else None
    ax.bar(x, df["accuracy"], yerr=err, capsize=3, color="#4C72B0", alpha=0.85,
           error_kw={"lw": 0.8, "alpha": 0.6})
    mean_acc = float(df["accuracy"].mean())
    ax.axhline(mean_acc, color="#C44E52", ls="--", lw=1.2, label=f"mean = {mean_acc:.3f}")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance = 0.5")
    ax.set_xticks(x, [f"{int(s)}" for s in df["subject"]])
    ax.set_xlabel("subject"), ax.set_ylabel("accuracy (cross-validated)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("EEGNet decoding: imagined left vs right hand", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
