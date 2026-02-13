from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, c) = self.lstm(packed)
        return h, c

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward_step(self, inp: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]):
        emb = self.embedding(inp).unsqueeze(1)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out.squeeze(1))
        return logits, hidden

class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder: LSTMEncoder, decoder: LSTMDecoder, pad_id: int, bos_id: int, eos_id: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt: torch.Tensor, teacher_forcing: float = 0.5) -> torch.Tensor:
        B, T = tgt.size()
        device = tgt.device
        hidden = self.encoder(src, src_lens)
        outputs = []
        inp = tgt[:, 0]
        for t in range(1, T):
            logits, hidden = self.decoder.forward_step(inp, hidden)
            outputs.append(logits.unsqueeze(1))
            use_tf = (torch.rand(1, device=device).item() < teacher_forcing)
            inp = tgt[:, t] if use_tf else logits.argmax(dim=-1)
        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, src_lens: torch.Tensor, max_len: int) -> torch.Tensor:
        B = src.size(0)
        device = src.device
        hidden = self.encoder(src, src_lens)
        inp = torch.full((B,), self.bos_id, dtype=torch.long, device=device)
        out_tokens = [inp]
        for _ in range(max_len):
            logits, hidden = self.decoder.forward_step(inp, hidden)
            inp = logits.argmax(dim=-1)
            out_tokens.append(inp)
        return torch.stack(out_tokens, dim=1)
