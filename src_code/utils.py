import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_training_curves(history, save_path, title_prefix=""):
    ensure_dir(os.path.dirname(save_path) or ".")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.get("train_loss", []), label="train_loss")
    axes[0].plot(history.get("val_loss", []), label="val_loss")
    axes[0].set_title(f"{title_prefix} Loss".strip())
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.get("train_acc", []), label="train_acc")
    axes[1].plot(history.get("val_acc", []), label="val_acc")
    axes[1].set_title(f"{title_prefix} Accuracy".strip())
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, labels, save_path, title="Confusion Matrix"):
    ensure_dir(os.path.dirname(save_path) or ".")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
