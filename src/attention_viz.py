from __future__ import annotations
from typing import List
import os

import torch
import matplotlib.pyplot as plt

from .utils import ensure_dir, strip_special

@torch.no_grad()
def generate_attention_heatmaps(model, loader, vocab, device: torch.device, max_len: int, out_dir: str, num_examples: int = 3) -> List[str]:
    ensure_dir(out_dir)
    model.eval()
    model.to(device)

    saved = []
    for batch in loader:
        src = batch["src"].to(device)
        src_lens = batch["src_lens"].to(device)

        tokens, attn = model.greedy_decode(src, src_lens, max_len=max_len, return_attn=True)
        B = tokens.size(0)

        for i in range(B):
            if len(saved) >= num_examples:
                return saved

            src_ids = strip_special(src[i].tolist(), vocab.bos_id, vocab.eos_id, vocab.pad_id)
            src_toks = vocab.decode(src_ids)

            pred_ids = strip_special(tokens[i].tolist(), vocab.bos_id, vocab.eos_id, vocab.pad_id)
            pred_toks = vocab.decode(pred_ids)

            attn_i = attn[i]  # (T, S)
            S = len(src_toks)
            T = len(pred_toks)
            attn_i = attn_i[:T, :S].cpu().numpy()

            fig = plt.figure(figsize=(max(6, S * 0.35), max(4, T * 0.35)))
            ax = plt.gca()
            im = ax.imshow(attn_i, aspect="auto")
            ax.set_xticks(range(S))
            ax.set_yticks(range(T))
            ax.set_xticklabels(src_toks, rotation=60, ha="right", fontsize=8)
            ax.set_yticklabels(pred_toks, fontsize=8)
            ax.set_xlabel("Docstring tokens")
            ax.set_ylabel("Generated code tokens")
            ax.set_title("Bahdanau Attention Heatmap")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            path = os.path.join(out_dir, f"attn_{len(saved)+1}.png")
            plt.tight_layout()
            plt.savefig(path, dpi=200)
            plt.close(fig)
            saved.append(path)
    return saved
