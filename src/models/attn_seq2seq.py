from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor):
        emb = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return enc_out, (h, c)

class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        self.W_h = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_out: torch.Tensor, dec_h: torch.Tensor, mask: torch.Tensor):
        score = self.v(torch.tanh(self.W_h(enc_out) + self.W_s(dec_h).unsqueeze(1))).squeeze(-1)
        score = score.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(score, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)
        return ctx, attn

class AttnDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, enc_dim: int, dec_dim: int, num_layers: int = 1, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim + enc_dim, dec_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(dec_dim + enc_dim, vocab_size)

    def forward_step(self, inp: torch.Tensor, ctx: torch.Tensor, hidden):
        emb = self.embedding(inp).unsqueeze(1)
        x = torch.cat([emb, ctx.unsqueeze(1)], dim=-1)
        out, hidden = self.lstm(x, hidden)
        dec = out.squeeze(1)
        logits = self.fc(torch.cat([dec, ctx], dim=-1))
        return logits, hidden

class Seq2SeqAttn(nn.Module):
    def __init__(self, encoder: BiLSTMEncoder, attention: BahdanauAttention, decoder: AttnDecoder,
                 pad_id: int, bos_id: int, eos_id: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.encoder = encoder
        self.attention = attention
        self.decoder = decoder
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.bridge_h = nn.Linear(2 * hidden_dim, hidden_dim)
        self.bridge_c = nn.Linear(2 * hidden_dim, hidden_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def _bridge(self, enc_h: torch.Tensor, enc_c: torch.Tensor):
        L2, B, H = enc_h.shape
        L = L2 // 2
        enc_h = enc_h.view(L, 2, B, H).transpose(1, 2)
        enc_c = enc_c.view(L, 2, B, H).transpose(1, 2)
        h_cat = torch.cat([enc_h[:, :, 0, :], enc_h[:, :, 1, :]], dim=-1)
        c_cat = torch.cat([enc_c[:, :, 0, :], enc_c[:, :, 1, :]], dim=-1)
        h0 = torch.tanh(self.bridge_h(h_cat))
        c0 = torch.tanh(self.bridge_c(c_cat))
        return h0, c0

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt: torch.Tensor, teacher_forcing: float = 0.5):
        B, T = tgt.size()
        device = tgt.device
        enc_out, (enc_h, enc_c) = self.encoder(src, src_lens)
        dec_hidden = self._bridge(enc_h, enc_c)

        S = enc_out.size(1)
        mask = (torch.arange(S, device=device).unsqueeze(0) < src_lens.unsqueeze(1)).long()

        outputs = []
        inp = tgt[:, 0]
        for t in range(1, T):
            dec_h_top = dec_hidden[0][-1]
            ctx, _ = self.attention(enc_out, dec_h_top, mask)
            logits, dec_hidden = self.decoder.forward_step(inp, ctx, dec_hidden)
            outputs.append(logits.unsqueeze(1))
            use_tf = (torch.rand(1, device=device).item() < teacher_forcing)
            inp = tgt[:, t] if use_tf else logits.argmax(dim=-1)
        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, src_lens: torch.Tensor, max_len: int, return_attn: bool = False):
        B = src.size(0)
        device = src.device
        enc_out, (enc_h, enc_c) = self.encoder(src, src_lens)
        dec_hidden = self._bridge(enc_h, enc_c)

        S = enc_out.size(1)
        mask = (torch.arange(S, device=device).unsqueeze(0) < src_lens.unsqueeze(1)).long()

        inp = torch.full((B,), self.bos_id, dtype=torch.long, device=device)
        out_tokens = [inp]
        attn_steps = []
        for _ in range(max_len):
            dec_h_top = dec_hidden[0][-1]
            ctx, attn = self.attention(enc_out, dec_h_top, mask)
            logits, dec_hidden = self.decoder.forward_step(inp, ctx, dec_hidden)
            inp = logits.argmax(dim=-1)
            out_tokens.append(inp)
            if return_attn:
                attn_steps.append(attn.unsqueeze(1))
        tokens = torch.stack(out_tokens, dim=1)
        if return_attn:
            attn_tensor = torch.cat(attn_steps, dim=1) if attn_steps else torch.empty((B,0,S), device=device)
            return tokens, attn_tensor
        return tokens
