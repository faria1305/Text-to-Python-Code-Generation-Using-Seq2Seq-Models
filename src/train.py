from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import ensure_dir, save_checkpoint

def compute_loss(logits: torch.Tensor, tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    B, Tm1, V = logits.size()
    target = tgt[:, 1:1+Tm1].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    return loss_fn(logits.view(B*Tm1, V), target.view(B*Tm1))

@torch.no_grad()
def eval_loss(model: torch.nn.Module, loader, device: torch.device, pad_id: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_lens = batch["src_lens"].to(device)
        logits = model(src, src_lens, tgt, teacher_forcing=0.0)
        loss = compute_loss(logits, tgt, pad_id)
        total += loss.item()
        n += 1
    return total / max(1, n)

def train_model(model, train_loader, val_loader, device, pad_id, epochs, lr, teacher_forcing, ckpt_dir, ckpt_name, meta):
    ensure_dir(ckpt_dir)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", leave=False)
        for batch in pbar:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_lens = batch["src_lens"].to(device)
            opt.zero_grad()
            logits = model(src, src_lens, tgt, teacher_forcing=teacher_forcing)
            loss = compute_loss(logits, tgt, pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
            n += 1
            pbar.set_postfix(loss=running/max(1,n))
        train_loss = running / max(1, n)
        val_loss = eval_loss(model, val_loader, device, pad_id)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(f"{ckpt_dir}/{ckpt_name}.pt", model, opt, {"best_val": best_val, **meta, "history": history})

    return history
