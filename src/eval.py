from __future__ import annotations
from typing import Any, Dict, List, Tuple
import ast

import torch
from sacrebleu import corpus_bleu
from tqdm import tqdm

from .utils import strip_special, safe_detok

def token_accuracy(pred: List[List[int]], gold: List[List[int]], pad_id: int) -> float:
    correct = 0
    total = 0
    for p, g in zip(pred, gold):
        for pt, gt in zip(p, g):
            if gt == pad_id:
                continue
            total += 1
            correct += int(pt == gt)
    return correct / max(1, total)

def exact_match(pred: List[List[int]], gold: List[List[int]]) -> float:
    return sum(int(p == g) for p, g in zip(pred, gold)) / max(1, len(pred))

def bleu_score(pred_text: List[str], ref_text: List[str]) -> float:
    return float(corpus_bleu(pred_text, [ref_text]).score)

def try_ast_parse(code_str: str) -> bool:
    try:
        ast.parse(code_str)
        return True
    except Exception:
        return False

@torch.no_grad()
def evaluate(model, loader, vocab, device: torch.device, max_len: int):
    model.eval()
    model.to(device)

    pred_ids_all, gold_ids_all = [], []
    pred_text, ref_text = [], []
    ast_ok, ast_total = 0, 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_lens = batch["src_lens"].to(device)

        out = model.greedy_decode(src, src_lens, max_len=max_len)
        for i in range(out.size(0)):
            pred = strip_special(out[i].tolist(), vocab.bos_id, vocab.eos_id, vocab.pad_id)
            gold = strip_special(tgt[i].tolist(), vocab.bos_id, vocab.eos_id, vocab.pad_id)

            pred_ids_all.append(pred)
            gold_ids_all.append(gold)

            ptxt = safe_detok(vocab.decode(pred))
            rtxt = safe_detok(vocab.decode(gold))
            pred_text.append(ptxt)
            ref_text.append(rtxt)

            ast_total += 1
            ast_ok += int(try_ast_parse(ptxt))

    maxL = max([len(x) for x in gold_ids_all] + [1])
    pred_pad = [x + [vocab.pad_id] * (maxL - len(x)) for x in pred_ids_all]
    gold_pad = [x + [vocab.pad_id] * (maxL - len(x)) for x in gold_ids_all]

    metrics = {
        "token_accuracy": token_accuracy(pred_pad, gold_pad, vocab.pad_id),
        "exact_match": exact_match(pred_ids_all, gold_ids_all),
        "bleu": bleu_score(pred_text, ref_text),
        "ast_parse_rate": ast_ok / max(1, ast_total),
    }
    samples = {"pred_text": pred_text, "ref_text": ref_text}
    return metrics, samples

@torch.no_grad()
def evaluate_by_doc_len(model, loader, vocab, device: torch.device, max_len: int, buckets: List[Tuple[int,int]]):
    model.eval()
    model.to(device)
    acc = {f"{lo}-{hi}": {"pred": [], "gold": [], "pred_txt": [], "ref_txt": []} for lo,hi in buckets}

    for batch in tqdm(loader, desc="Eval buckets", leave=False):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_lens = batch["src_lens"].to(device)
        out = model.greedy_decode(src, src_lens, max_len=max_len)

        for i in range(out.size(0)):
            src_len = int(src_lens[i].item()) - 2
            key = None
            for lo, hi in buckets:
                if lo <= src_len <= hi:
                    key = f"{lo}-{hi}"
                    break
            if key is None:
                continue

            pred = strip_special(out[i].tolist(), vocab.bos_id, vocab.eos_id, vocab.pad_id)
            gold = strip_special(tgt[i].tolist(), vocab.bos_id, vocab.eos_id, vocab.pad_id)

            acc[key]["pred"].append(pred)
            acc[key]["gold"].append(gold)
            acc[key]["pred_txt"].append(safe_detok(vocab.decode(pred)))
            acc[key]["ref_txt"].append(safe_detok(vocab.decode(gold)))

    out_metrics = {}
    for key, d in acc.items():
        if not d["pred"]:
            out_metrics[key] = {"count": 0, "token_accuracy": 0.0, "exact_match": 0.0, "bleu": 0.0}
            continue
        maxL = max([len(x) for x in d["gold"]] + [1])
        pred_pad = [x + [vocab.pad_id] * (maxL - len(x)) for x in d["pred"]]
        gold_pad = [x + [vocab.pad_id] * (maxL - len(x)) for x in d["gold"]]
        out_metrics[key] = {
            "count": len(d["pred"]),
            "token_accuracy": token_accuracy(pred_pad, gold_pad, vocab.pad_id),
            "exact_match": exact_match(d["pred"], d["gold"]),
            "bleu": bleu_score(d["pred_txt"], d["ref_txt"]),
        }
    return out_metrics
