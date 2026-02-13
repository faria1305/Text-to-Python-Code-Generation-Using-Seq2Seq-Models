"""
Data preprocessing utilities for text-to-code generation.
"""

import re
import torch
from collections import Counter


class Vocabulary:
    """Vocabulary class for managing token-to-index mappings."""
    
    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'
        
        # Initialize special tokens
        self.add_token(self.PAD_TOKEN)
        self.add_token(self.SOS_TOKEN)
        self.add_token(self.EOS_TOKEN)
        self.add_token(self.UNK_TOKEN)
        
    def add_token(self, token):
        """Add a token to the vocabulary."""
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
            
    def __len__(self):
        return len(self.token2idx)
    
    def get_idx(self, token):
        """Get index for a token, return UNK if not found."""
        return self.token2idx.get(token, self.token2idx[self.UNK_TOKEN])
    
    def get_token(self, idx):
        """Get token for an index."""
        return self.idx2token.get(idx, self.UNK_TOKEN)


def tokenize_text(text):
    """Simple tokenization for natural language text."""
    # Convert to lowercase and split by whitespace/punctuation
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def tokenize_code(code):
    """Tokenization for Python code."""
    # Split by whitespace and common Python operators/symbols
    tokens = re.findall(r'\b\w+\b|[^\w\s]', code)
    return tokens


def build_vocabulary(texts, min_freq=1):
    """Build vocabulary from a list of tokenized texts."""
    vocab = Vocabulary()
    counter = Counter()
    
    for text in texts:
        for token in text:
            counter[token] += 1
    
    # Add tokens that appear at least min_freq times
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab.add_token(token)
    
    return vocab


def encode_sequence(tokens, vocab, max_len=None):
    """Encode a sequence of tokens to indices."""
    indices = [vocab.get_idx(token) for token in tokens]
    
    if max_len is not None:
        if len(indices) < max_len:
            # Pad with PAD tokens
            indices = indices + [vocab.token2idx[vocab.PAD_TOKEN]] * (max_len - len(indices))
        else:
            # Truncate
            indices = indices[:max_len]
    
    return indices


def decode_sequence(indices, vocab):
    """Decode a sequence of indices to tokens."""
    tokens = []
    for idx in indices:
        token = vocab.get_token(idx)
        if token == vocab.EOS_TOKEN:
            break
        if token != vocab.PAD_TOKEN and token != vocab.SOS_TOKEN:
            tokens.append(token)
    return tokens


class CodeDataset(torch.utils.data.Dataset):
    """Dataset for text-to-code pairs."""
    
    def __init__(self, text_sequences, code_sequences, text_vocab, code_vocab, max_text_len=50, max_code_len=100):
        """
        Args:
            text_sequences: List of tokenized text sequences
            code_sequences: List of tokenized code sequences
            text_vocab: Vocabulary for text
            code_vocab: Vocabulary for code
            max_text_len: Maximum length for text sequences
            max_code_len: Maximum length for code sequences
        """
        self.text_sequences = text_sequences
        self.code_sequences = code_sequences
        self.text_vocab = text_vocab
        self.code_vocab = code_vocab
        self.max_text_len = max_text_len
        self.max_code_len = max_code_len
        
    def __len__(self):
        return len(self.text_sequences)
    
    def __getitem__(self, idx):
        text_tokens = self.text_sequences[idx]
        code_tokens = self.code_sequences[idx]
        
        # Encode sequences
        text_indices = encode_sequence(text_tokens, self.text_vocab, self.max_text_len)
        
        # Add SOS and EOS to code
        code_with_markers = [self.code_vocab.SOS_TOKEN] + code_tokens + [self.code_vocab.EOS_TOKEN]
        code_indices = encode_sequence(code_with_markers, self.code_vocab, self.max_code_len + 2)
        
        return {
            'text': torch.tensor(text_indices, dtype=torch.long),
            'code': torch.tensor(code_indices, dtype=torch.long),
            'text_len': min(len(text_tokens), self.max_text_len),
            'code_len': min(len(code_with_markers), self.max_code_len + 2)
        }


def create_sample_dataset():
    """Create a sample dataset for testing/demonstration."""
    # Sample docstring-code pairs
    samples = [
        ("calculate the sum of two numbers", "def add(a, b): return a + b"),
        ("multiply two numbers and return result", "def multiply(x, y): return x * y"),
        ("subtract second number from first", "def subtract(a, b): return a - b"),
        ("divide first number by second", "def divide(a, b): return a / b if b != 0 else 0"),
        ("check if number is even", "def is_even(n): return n % 2 == 0"),
        ("check if number is odd", "def is_odd(n): return n % 2 != 0"),
        ("find maximum of two numbers", "def max_num(a, b): return a if a > b else b"),
        ("find minimum of two numbers", "def min_num(a, b): return a if a < b else b"),
        ("calculate square of a number", "def square(x): return x * x"),
        ("calculate cube of a number", "def cube(x): return x * x * x"),
        ("check if string is empty", "def is_empty(s): return len(s) == 0"),
        ("get length of string", "def get_length(s): return len(s)"),
        ("convert string to uppercase", "def to_upper(s): return s.upper()"),
        ("convert string to lowercase", "def to_lower(s): return s.lower()"),
        ("reverse a string", "def reverse(s): return s[::-1]"),
        ("check if list is empty", "def is_empty_list(lst): return len(lst) == 0"),
        ("get first element of list", "def first(lst): return lst[0]"),
        ("get last element of list", "def last(lst): return lst[-1]"),
        ("append element to list", "def append_elem(lst, elem): lst.append(elem)"),
        ("calculate absolute value", "def abs_val(x): return abs(x)"),
    ]
    
    texts = [tokenize_text(text) for text, _ in samples]
    codes = [tokenize_code(code) for _, code in samples]
    
    return texts, codes
