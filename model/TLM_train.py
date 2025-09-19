# tinystories_loader_and_train.py
import os
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from torch import nn
from tqdm import tqdm
import  tinymodel


# --------------------------
# Config / assumptions
# --------------------------
TXT_PATH = "E:\\tinyllm\\tiny_stories.txt"    # your .txt file (one story per line ideally)
SP_MODEL = "E:\\tinyllm\\sp_tinystory.model"           # your SentencePiece model file
MAX_LEN = 64                           # truncate stories to this many tokens
BATCH_SIZE = 32
NUM_WORKERS = 0                         # set 0 to avoid duplicated tqdm in notebooks
PAD_SEARCH_TOKENS = ["<pad>", "[PAD]", "</s>", "<s>"]  # fallback search
SHIFT_TARGETS_BY_ONE = True            # causal LM default: targets = input_ids shifted left

# --------------------------
# SentencePiece wrapper
# --------------------------
sp = spm.SentencePieceProcessor()
sp.Load(SP_MODEL)

# Try to find a PAD id; fall back to using '</s>' or 0 if not present
def find_pad_id(sp_processor):
    for token in PAD_SEARCH_TOKENS:
        try:
            pid = sp_processor.piece_to_id(token)
            # piece_to_id returns >= 0 for found tokens; if not found usually returns 0 for unk
            # We'll require the piece string is actually present (id != 0 or token == piece(0))
            # Simpler: check that id maps back to same piece
            if sp_processor.id_to_piece(pid) == token:
                return pid
        except Exception:
            pass
    # Fallback: use last index as pad (requires expanding model embeddings before training)
    fallback = sp_processor.get_piece_size()  # index equal to vocab_size (new index)
    return fallback

pad_id = find_pad_id(sp)

# Note: if pad_id == vocab_size you will need to ensure your model's token embedding
# is resized to accommodate the extra token. See note below.

# --------------------------
# Dataset: read .txt & tokenize
# --------------------------
class TinyStoriesTxtDataset(Dataset):
    def __init__(self, txt_path: str, sp_processor: spm.SentencePieceProcessor, max_len: int = 512):
        self.sp = sp_processor
        self.max_len = max_len
        # Load lines; filter empty lines
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f]
        # Keep non-empty lines
        self.examples = [ln for ln in lines if ln]
        if len(self.examples) == 0:
            raise ValueError("No examples found in the txt file.")
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        text = self.examples[idx]
        # Use encode (returns list of ints)
        ids = self.sp.encode(text, out_type=int)
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        return {"input_ids": ids}

# --------------------------
# Collate function
# --------------------------
def collate_fn(batch: List[dict], pad_id: int = pad_id, shift_targets: bool = SHIFT_TARGETS_BY_ONE):
    # batch: list of {"input_ids": [...]}
    seqs = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
    B = len(seqs)
    L = max([s.size(0) for s in seqs])
    padded = torch.full((B, L), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, L), dtype=torch.long)
    for i, s in enumerate(seqs):
        l = s.size(0)
        padded[i, :l] = s
        attention_mask[i, :l] = 1

    if shift_targets:
        # For causal LM: targets[t] = input_ids[t+1], so shift left
        targets = torch.full((B, L), pad_id, dtype=torch.long)
        if L > 1:
            targets[:, :-1] = padded[:, 1:].clone()
            targets[:, -1] = pad_id
        else:
            targets[:] = pad_id
    else:
        # No shift: model's logits at position t should predict token at position t
        targets = padded.clone()

    return padded, targets, attention_mask

# --------------------------
# Build dataloader
# --------------------------
dataset = TinyStoriesTxtDataset(TXT_PATH, sp, max_len=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, pad_id=pad_id, shift_targets=SHIFT_TARGETS_BY_ONE),
                        num_workers=NUM_WORKERS, pin_memory=True)

# --------------------------
# Hardened train function (drop-in replace of your original)
# --------------------------
def train(model, dataloader, optimizer, epochs=10, device=None,
          pad_id=pad_id, smoothing_alpha=0.01, verbose=False,
          use_model_labels_arg=False,  # if True, call model(..., labels=targets) and use returned loss
          clip_grad_norm: float = None):
    """
    - use_model_labels_arg: set True if your model supports `labels=` and returns `.loss`, e.g. HF models.
      If False, we compute loss manually with CrossEntropyLoss(ignore_index=pad_id).
    """
    device = device or next(model.parameters()).device
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        smoothed_loss = None
        alpha = float(smoothing_alpha)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="it", total=len(dataloader),
                    leave=True, dynamic_ncols=True)

        for batch_idx, (input_ids, target_ids, attention_mask) in enumerate(pbar):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()

            # call the model. handle different return types:
            try:
                outputs = model(input_ids=input_ids)#, attention_mask=attention_mask)
            except TypeError:
                # fallback to positional call if model expects only input_ids
                outputs = model(input_ids)

            # Extract logits robustly:
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                # maybe model returned a tensor directly
                logits = outputs

            # Option 1: let model compute loss by passing labels (only if you prefer and model supports it).
            if use_model_labels_arg:
                try:
                    out_with_labels = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
                    # HF models return a Loss in out.loss
                    loss = out_with_labels.loss
                except Exception:
                    # fallback to manual loss if that fails
                    B, T, V = logits.shape
                    loss = criterion(logits.view(B * T, V), target_ids.view(B * T))
            else:
                B, T, V = logits.shape
                loss = criterion(logits.view(B * T, V), target_ids.view(B * T))

            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

            loss_item = loss.item()
            total_loss += loss_item

            # EMA smoothing for display
            if smoothed_loss is None:
                smoothed_loss = loss_item
            else:
                smoothed_loss = smoothed_loss * (1.0 - alpha) + loss_item * alpha

            pbar.set_postfix_str(f"loss={smoothed_loss:.3f}")

            if verbose and (batch_idx + 1) % 1000 == 0:
                pbar.write(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1} | Loss: {loss_item:.4f}")

        # Save model.state_dict (recommended)
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model, os.path.join(save_dir, f"model_epoch{epoch+1}.pth"))

        avg_loss = total_loss / max(1, len(dataloader))
        pbar.close()
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")

# --------------------------
# Example usage
# --------------------------

# Hyperparameters
vocab_size = 6000  # Will be updated based on tokenizer
n_embd = 64
n_head = 4
n_layer = 1
block_size = 64
batch_size = 32
learning_rate = 3e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = tinymodel.Transformer(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
# If pad_id == sp.get_piece_size you must ensure model embeddings are resized by 1.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train(model, dataloader, optimizer, epochs=3, device=device)
