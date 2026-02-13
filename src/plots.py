from __future__ import annotations
from typing import Dict, List
import os
import matplotlib.pyplot as plt
from .utils import ensure_dir

def plot_loss_curves(history: Dict[str, List[float]], out_path: str, title: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    fig = plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_len_bucket(metric_by_bucket: Dict[str, Dict[str, float]], out_path: str, metric_name: str, title: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    keys = list(metric_by_bucket.keys())
    vals = [metric_by_bucket[k].get(metric_name, 0.0) for k in keys]
    fig = plt.figure()
    plt.plot(range(len(keys)), vals, marker="o")
    plt.xticks(range(len(keys)), keys, rotation=45, ha="right")
    plt.xlabel("docstring length bucket (tokens)")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
