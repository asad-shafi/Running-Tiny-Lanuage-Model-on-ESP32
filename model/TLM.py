import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import sentencepiece as spm
from typing import List, Dict
# ---- attention + block (replace original) ----
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask sized by block_size
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)  # moves with model

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# ---- Transformer changes: construct blocks with block_size ----
class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        # create ModuleList so we can pass block_size
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
        # weight tying
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx, targets=None, pad_id=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        # compute loss here only if pad_id provided, else return logits,loss None
        if pad_id is None:
            return logits, None
        logits_flat = logits.view(B*T, -1)
        targets_flat = targets.view(B*T)
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_id)
        return logits, loss
