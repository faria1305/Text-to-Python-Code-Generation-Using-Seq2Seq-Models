from __future__ import annotations
import os
import re
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from .vocab import Vocab, build_vocab
from .utils import ensure_dir, save_json, load_json, RunConfig

# Simple code-friendly tokenizer (fallback if token columns are not present):
_PUNCT = r"([\[\]\(\)\{\},\.:;])"
_OPS = r"(==|!=|<=|>=|\+=|-=|\*=|/=|//=|\*\*|->|<<|>>|\+|\-|\*|/|%|=|<|>)"

def tokenize(text: str) -> List[str]:
    text = (text or "").strip()
    text = re.sub(_OPS, r" \1 ", text)
    text = re.sub(_PUNCT, r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return text.split(" ")

def clip(tokens: List[str], max_len: int) -> List[str]:
    return tokens[:max_len]

class PairDataset(Dataset):
    def __init__(self, src_ids: List[List[int]], tgt_ids: List[List[int]]):
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids
    def __len__(self) -> int:
        return len(self.src_ids)
    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return {"src": self.src_ids[idx], "tgt": self.tgt_ids[idx]}

def pad_batch(batch: List[Dict[str, List[int]]], pad_id: int) -> Dict[str, torch.Tensor]:
    src_lens = torch.tensor([len(x["src"]) for x in batch], dtype=torch.long)
    tgt_lens = torch.tensor([len(x["tgt"]) for x in batch], dtype=torch.long)
    max_src = int(src_lens.max().item())
    max_tgt = int(tgt_lens.max().item())
    src = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    tgt = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        src[i, :len(x["src"])] = torch.tensor(x["src"], dtype=torch.long)
        tgt[i, :len(x["tgt"])] = torch.tensor(x["tgt"], dtype=torch.long)
    return {"src": src, "tgt": tgt, "src_lens": src_lens, "tgt_lens": tgt_lens}

def _partition_splits(ds_train):
    """Nan-Do/code-search-net-python provides a 'partition' column containing train/valid/test labels."""
    if "partition" not in ds_train.column_names:
        return None

    def norm(x): return str(x).strip().lower()

    sample = ds_train.select(range(min(2000, len(ds_train))))
    parts = set(norm(x) for x in sample["partition"])

    def filt(p):
        return ds_train.filter(lambda ex: norm(ex.get("partition", "")) == p)

    train_split = filt("train") if "train" in parts else None
    val_split = filt("valid") if "valid" in parts else (filt("validation") if "validation" in parts else None)
    test_split = filt("test") if "test" in parts else None

    if train_split is None or val_split is None or test_split is None:
        return None
    return train_split, val_split, test_split

def prepare_data(cfg: RunConfig, cache_dir: str) -> Dict[str, str]:
    """Download dataset, make train/val/test, subset train to 5k–10k, cap lengths (50/80), build vocab, cache."""
    ensure_dir(cache_dir)
    meta_path = os.path.join(cache_dir, "meta.json")
    if os.path.exists(meta_path):
        return load_json(meta_path)

    ds = load_dataset(cfg.dataset_name)

    # split handling
    if "train" in ds and "validation" in ds and "test" in ds:
        train_split = ds["train"]
        val_split = ds["validation"]
        test_split = ds["test"]
    else:
        if "train" not in ds:
            raise ValueError(f"Dataset has no 'train' split. Available: {list(ds.keys())}")
        maybe = _partition_splits(ds["train"])
        if maybe is not None:
            train_split, val_split, test_split = maybe
        else:
            full = ds["train"]
            split = full.train_test_split(test_size=0.2, seed=cfg.seed)
            train_split = split["train"]
            tmp = split["test"].train_test_split(test_size=0.5, seed=cfg.seed + 1)
            val_split = tmp["train"]
            test_split = tmp["test"]

    # subset train for feasibility
    train_split = train_split.select(range(min(cfg.train_size, len(train_split))))

    def get_tokens(ex):
        # Prefer provided tokenizer columns when available
        if "docstring_tokens" in ex and isinstance(ex["docstring_tokens"], (list, tuple)):
            src_toks = [str(t) for t in ex["docstring_tokens"]]
        else:
            src_toks = tokenize(ex.get("docstring", ""))
        if "code_tokens" in ex and isinstance(ex["code_tokens"], (list, tuple)):
            tgt_toks = [str(t) for t in ex["code_tokens"]]
        else:
            tgt_toks = tokenize(ex.get("code", ""))
        return clip(src_toks, cfg.max_src_len), clip(tgt_toks, cfg.max_tgt_len)

    def tok_split(split):
        src_tok, tgt_tok = [], []
        for ex in split:
            s, t = get_tokens(ex)
            src_tok.append(s)
            tgt_tok.append(t)
        return src_tok, tgt_tok

    train_src, train_tgt = tok_split(train_split)
    val_src, val_tgt = tok_split(val_split)
    test_src, test_tgt = tok_split(test_split)

    vocab = build_vocab(list(train_src) + list(train_tgt), max_size=cfg.vocab_size)

    def numerize(src_tok, tgt_tok):
        src_ids, tgt_ids = [], []
        for s_toks, t_toks in zip(src_tok, tgt_tok):
            s_ids = [vocab.bos_id] + vocab.encode(s_toks) + [vocab.eos_id]
            t_ids = [vocab.bos_id] + vocab.encode(t_toks) + [vocab.eos_id]
            src_ids.append(s_ids)
            tgt_ids.append(t_ids)
        return src_ids, tgt_ids

    train_src_ids, train_tgt_ids = numerize(train_src, train_tgt)
    val_src_ids, val_tgt_ids = numerize(val_src, val_tgt)
    test_src_ids, test_tgt_ids = numerize(test_src, test_tgt)

    torch.save({"src": train_src_ids, "tgt": train_tgt_ids}, os.path.join(cache_dir, "train.pt"))
    torch.save({"src": val_src_ids, "tgt": val_tgt_ids}, os.path.join(cache_dir, "val.pt"))
    torch.save({"src": test_src_ids, "tgt": test_tgt_ids}, os.path.join(cache_dir, "test.pt"))
    save_json(os.path.join(cache_dir, "vocab.json"), {"itos": vocab.itos})

    meta = {
        "cache_dir": cache_dir,
        "train_pt": os.path.join(cache_dir, "train.pt"),
        "val_pt": os.path.join(cache_dir, "val.pt"),
        "test_pt": os.path.join(cache_dir, "test.pt"),
        "vocab_json": os.path.join(cache_dir, "vocab.json"),
        "dataset_name": cfg.dataset_name,
        "train_size": cfg.train_size,
        "max_src_len": cfg.max_src_len,
        "max_tgt_len": cfg.max_tgt_len,
    }
    save_json(meta_path, meta)
    return meta

def load_vocab(vocab_json: str) -> Vocab:
    obj = load_json(vocab_json)
    itos = obj["itos"]
    stoi = {t:i for i,t in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)

def make_loaders(meta: Dict[str, str], vocab: Vocab, cfg: RunConfig, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train = torch.load(meta["train_pt"])
    val = torch.load(meta["val_pt"])
    test = torch.load(meta["test_pt"])

    train_ds = PairDataset(train["src"], train["tgt"])
    val_ds = PairDataset(val["src"], val["tgt"])
    test_ds = PairDataset(test["src"], test["tgt"])

    collate = lambda batch: pad_batch(batch, vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)
    return train_loader, val_loader, test_loader
