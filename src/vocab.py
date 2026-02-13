from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def pad_id(self) -> int: return self.stoi["<pad>"]
    @property
    def bos_id(self) -> int: return self.stoi["<bos>"]
    @property
    def eos_id(self) -> int: return self.stoi["<eos>"]
    @property
    def unk_id(self) -> int: return self.stoi["<unk>"]

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in ids]

def build_vocab(token_stream: Iterable[List[str]], max_size: int) -> Vocab:
    cnt = Counter()
    for toks in token_stream:
        cnt.update(toks)
    itos = list(SPECIAL_TOKENS)
    for tok, _ in cnt.most_common(max(0, max_size - len(itos))):
        if tok in SPECIAL_TOKENS:
            continue
        itos.append(tok)
    stoi = {t:i for i,t in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)
