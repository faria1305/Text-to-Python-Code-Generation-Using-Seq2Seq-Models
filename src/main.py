from __future__ import annotations
import argparse
import os
from dataclasses import asdict
from typing import Dict, Any

import torch

from .utils import RunConfig, set_seed, get_device, ensure_dir, save_json, load_json
from .data import prepare_data, load_vocab, make_loaders
from .train import train_model
from .eval import evaluate, evaluate_by_doc_len
from .attention_viz import generate_attention_heatmaps
from .plots import plot_loss_curves, plot_len_bucket

from .models.rnn_seq2seq import RNNEncoder, RNNDecoder, Seq2SeqRNN
from .models.lstm_seq2seq import LSTMEncoder, LSTMDecoder, Seq2SeqLSTM
from .models.attn_seq2seq import BiLSTMEncoder, BahdanauAttention, AttnDecoder, Seq2SeqAttn

def build_model(model_name: str, vocab_size: int, cfg: RunConfig, pad_id: int, bos_id: int, eos_id: int):
    if model_name == "rnn":
        enc = RNNEncoder(vocab_size, cfg.embed_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, pad_id)
        dec = RNNDecoder(vocab_size, cfg.embed_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, pad_id)
        return Seq2SeqRNN(enc, dec, pad_id, bos_id, eos_id)
    if model_name == "lstm":
        enc = LSTMEncoder(vocab_size, cfg.embed_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, pad_id)
        dec = LSTMDecoder(vocab_size, cfg.embed_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, pad_id)
        return Seq2SeqLSTM(enc, dec, pad_id, bos_id, eos_id)
    if model_name == "attn":
        enc = BiLSTMEncoder(vocab_size, cfg.embed_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, pad_id)
        attn = BahdanauAttention(enc_dim=2*cfg.hidden_dim, dec_dim=cfg.hidden_dim, attn_dim=cfg.hidden_dim)
        dec = AttnDecoder(vocab_size, cfg.embed_dim, enc_dim=2*cfg.hidden_dim, dec_dim=cfg.hidden_dim,
                          num_layers=cfg.num_layers, dropout=cfg.dropout, pad_id=pad_id)
        return Seq2SeqAttn(enc, attn, dec, pad_id, bos_id, eos_id, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers)
    raise ValueError(f"Unknown model: {model_name}")

def get_paths():
    out = "outputs"
    return {
        "out": out,
        "cache": os.path.join(out, "cache"),
        "ckpt": os.path.join(out, "checkpoints"),
        "logs": os.path.join(out, "logs"),
        "plots": os.path.join(out, "plots"),
        "attn": os.path.join(out, "attention"),
    }

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prep","train","eval","attn_viz","all"], default="all")
    p.add_argument("--model", choices=["rnn","lstm","attn"], default="attn")
    p.add_argument("--dataset_name", default="Nan-Do/code-search-net-python")
    p.add_argument("--train_size", type=int, default=8000)
    p.add_argument("--max_src_len", type=int, default=50)
    p.add_argument("--max_tgt_len", type=int, default=80)
    p.add_argument("--vocab_size", type=int, default=30000)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--teacher_forcing", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_attention_examples", type=int, default=3)
    return p.parse_args()

def load_trained(model_name: str, cfg: RunConfig, pth: Dict[str,str], vocab):
    ckpt_path = os.path.join(pth["ckpt"], f"{model_name}_best.pt")
    device = get_device()
    model = build_model(model_name, len(vocab.itos), cfg, vocab.pad_id, vocab.bos_id, vocab.eos_id)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model

def main():
    args = parse_args()
    cfg = RunConfig(
        dataset_name=args.dataset_name,
        train_size=args.train_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        teacher_forcing=args.teacher_forcing,
        seed=args.seed,
        num_attention_examples=args.num_attention_examples,
    )
    set_seed(cfg.seed)

    pth = get_paths()
    for k in pth.values():
        ensure_dir(k)

    # prep or load cached meta
    if args.mode in ("prep","all"):
        meta = prepare_data(cfg, cache_dir=pth["cache"])
    else:
        meta = load_json(os.path.join(pth["cache"], "meta.json"))

    vocab = load_vocab(meta["vocab_json"])
    train_loader, val_loader, test_loader = make_loaders(meta, vocab, cfg)

    device = get_device()
    ensure_dir(pth["logs"])

    def train_one(m):
        model = build_model(m, len(vocab.itos), cfg, vocab.pad_id, vocab.bos_id, vocab.eos_id)
        hist = train_model(model, train_loader, val_loader, device, vocab.pad_id, cfg.epochs, cfg.lr,
                           cfg.teacher_forcing, pth["ckpt"], f"{m}_best",
                           {"model": m, "cfg": asdict(cfg), "vocab_size": len(vocab.itos)})
        plot_loss_curves(hist, os.path.join(pth["plots"], f"loss_{m}.png"), f"Loss Curves ({m})")
        save_json(os.path.join(pth["logs"], f"train_{m}.json"), {"history": hist})
        return hist

    def eval_one(m):
        model = load_trained(m, cfg, pth, vocab)
        metrics, samples = evaluate(model, test_loader, vocab, device, max_len=cfg.max_tgt_len)
        save_json(os.path.join(pth["logs"], f"metrics_{m}.json"), metrics)
        save_json(os.path.join(pth["logs"], f"samples_{m}.json"), {"pred_text": samples["pred_text"][:50], "ref_text": samples["ref_text"][:50]})
        buckets = [(0,10),(11,20),(21,30),(31,40),(41,cfg.max_src_len)]
        by_len = evaluate_by_doc_len(model, test_loader, vocab, device, max_len=cfg.max_tgt_len, buckets=buckets)
        save_json(os.path.join(pth["logs"], f"by_len_{m}.json"), by_len)
        plot_len_bucket(by_len, os.path.join(pth["plots"], f"bleu_by_len_{m}.png"), "bleu", f"BLEU vs Docstring Length ({m})")
        plot_len_bucket(by_len, os.path.join(pth["plots"], f"tokenacc_by_len_{m}.png"), "token_accuracy", f"Token Acc vs Docstring Length ({m})")
        return {"metrics": metrics, "by_len": by_len}

    if args.mode == "train":
        train_one(args.model); return
    if args.mode == "eval":
        eval_one(args.model); return
    if args.mode == "attn_viz":
        model = load_trained("attn", cfg, pth, vocab)
        attn_paths = generate_attention_heatmaps(model, test_loader, vocab, device, cfg.max_tgt_len, pth["attn"], cfg.num_attention_examples)
        save_json(os.path.join(pth["logs"], "attention_paths.json"), {"paths": attn_paths})
        return

    # all
    summary: Dict[str, Any] = {"cfg": asdict(cfg), "models": {}}
    for m in ["rnn","lstm","attn"]:
        train_one(m)
        summary["models"][m] = eval_one(m)
    model = load_trained("attn", cfg, pth, vocab)
    attn_paths = generate_attention_heatmaps(model, test_loader, vocab, device, cfg.max_tgt_len, pth["attn"], cfg.num_attention_examples)
    summary["attention_paths"] = attn_paths
    save_json(os.path.join(pth["logs"], "run_summary.json"), summary)

if __name__ == "__main__":
    main()
