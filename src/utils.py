from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@dataclass
class RunConfig:
    dataset_name: str = "Nan-Do/code-search-net-python"
    train_size: int = 8000
    max_src_len: int = 50
    max_tgt_len: int = 80
    vocab_size: int = 30000
    embed_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 12
    lr: float = 3e-4
    teacher_forcing: float = 0.5
    seed: int = 42
    num_attention_examples: int = 3

def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, meta: Dict[str, Any]) -> None:
    torch.save({"model_state": model.state_dict(), "optim_state": optimizer.state_dict(), "meta": meta}, path)

def strip_special(tokens: List[int], bos_id: int, eos_id: int, pad_id: int) -> List[int]:
    out: List[int] = []
    for t in tokens:
        if t in (pad_id, bos_id):
            continue
        if t == eos_id:
            break
        out.append(t)
    return out

def safe_detok(tokens: List[str]) -> str:
    return " ".join(tokens)
