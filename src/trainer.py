import math
import time
from dataclasses import asdict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset

from transformer import Config, DecoderBlock, EncoderBlock
from transformer_utils import device
from utils import DataLoader as FOLDataLoader
from utils import FOLTokenizerPipeline

class FOLDataset(Dataset):
    def __init__(self, df, nl_ids_col="NL_ids", fol_ids_col="FOL_ids"):
        self.src = df[nl_ids_col].tolist()
        self.tgt = df[fol_ids_col].tolist()

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.src[idx], dtype=torch.long),
            torch.tensor(self.tgt[idx], dtype=torch.long),
        )


def make_collate_fn(src_pad_id, tgt_pad_id, enc_block_size, dec_block_size, ignore_index=-100):
    def collate(batch):
        src_seqs, tgt_seqs = zip(*batch)

        src_padded = torch.full((len(batch), enc_block_size), src_pad_id, dtype=torch.long)
        tgt_padded = torch.full((len(batch), dec_block_size), tgt_pad_id, dtype=torch.long)
        src_mask = torch.zeros((len(batch), enc_block_size), dtype=torch.long)

        for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
            src_padded[i, :s.size(0)] = s
            tgt_padded[i, :t.size(0)] = t
            src_mask[i, :s.size(0)] = 1

        decoder_input_ids = tgt_padded[:, :-1]
        labels = tgt_padded[:, 1:].clone()
        labels[labels == tgt_pad_id] = ignore_index

        return {
            "encoder_input_ids": src_padded,
            "encoder_attention_mask": src_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
    return collate

def get_lr(step, warmup_steps, max_steps, base_lr, min_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)
def build_param_groups(*modules, weight_decay):
    decay, no_decay = [], []
    for module in modules:
        for _, p in module.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def main():
    # ---- data ----
    loader = FOLDataLoader(
        file_path_folio="<file_path>",
        file_path_main="<file_path>"",
        file_path_malls="<file_path>",
    )
    df = loader.get_data()

    pipeline = FOLTokenizerPipeline(df)
    df = pipeline.run_all()
    nl_tokenizer = pipeline.nl_tokenizer
    fol_tokenizer = pipeline.fol_tokenizer

    ENC_BLOCK_SIZE, DEC_BLOCK_SIZE = 128, 384
    before = len(df)
    df = df[
        (df['NL_ids'].apply(len) <= ENC_BLOCK_SIZE) &  # pyright: ignore[reportAttributeAccessIssue]
        (df['FOL_ids'].apply(len) <= DEC_BLOCK_SIZE) # pyright: ignore[reportAttributeAccessIssue]
    ].reset_index(drop=True) # pyright: ignore[reportAttributeAccessIssue]
    print(f"dropped {before - len(df)} row(s) exceeding block size ({before} -> {len(df)})")

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_val = int(len(df) * 0.05)
    val_df = df.iloc[:n_val].reset_index(drop=True)
    train_df = df.iloc[n_val:].reset_index(drop=True)
    print(f"train rows: {len(train_df)}  val rows: {len(val_df)}")

    ENC_BLOCK_SIZE, DEC_BLOCK_SIZE = 128, 384

    collate_fn = make_collate_fn(
        src_pad_id=nl_tokenizer.pad_token_id,
        tgt_pad_id=fol_tokenizer.pad_token_id,
        enc_block_size=ENC_BLOCK_SIZE,
        dec_block_size=DEC_BLOCK_SIZE,
    )
    batch_size = 8   # was 32 -- lower first if hitting MPS OOM
    train_loader = TorchDataLoader(FOLDataset(train_df), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = TorchDataLoader(FOLDataset(val_df), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    config = Config(
        enc_vocab=len(nl_tokenizer),
        dec_vocab=len(fol_tokenizer),
        enc_block_size=128,   # covers NL max of 79 with margin
        dec_block_size=384,   # covers FOL max of 313 with margin
        enc_n_layers=4,
        dec_n_layers=4,
        n_heads=4,
        n_embd=128,
        dropout=0.1,
        bias=False,
    )
    print("creating encoder...", flush=True)
    encoder = EncoderBlock(config)
    print("moving encoder to device...", flush=True)
    encoder = encoder.to(device)
    print("creating decoder...", flush=True)
    decoder = DecoderBlock(config)
    print("moving decoder to device...", flush=True)
    decoder = decoder.to(device)
    print("counting params...", flush=True)

    n_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
    print(f"total params: {n_params:,}", flush=True)

    weight_decay = 0.1
    base_lr = 3e-4
    min_lr = base_lr * 0.1
    optimizer = AdamW(build_param_groups(encoder, decoder, weight_decay=weight_decay), lr=base_lr, betas=(0.9, 0.95))

    epochs = 30
    grad_clip = 1.0
    max_steps = len(train_loader) * epochs
    warmup_steps = max(100, int(0.03 * max_steps))

    best_val_loss = float("inf")
    ckpt_path = "best_model.pt"
    step = 0

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        t0 = time.time()
        running_loss = 0.0

        for batch in train_loader:
            lr = get_lr(step, warmup_steps, max_steps, base_lr, min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            enc_ids = batch["encoder_input_ids"].to(device)
            enc_mask = batch["encoder_attention_mask"].to(device)
            dec_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            memory = encoder(enc_ids, key_padding_mask=enc_mask)
            _, loss = decoder(dec_ids, memory, memory_key_padding_mask=enc_mask, targets=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), grad_clip)
            optimizer.step()

            running_loss += loss.item()
            step += 1

        train_loss = running_loss / len(train_loader)

        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                enc_ids = batch["encoder_input_ids"].to(device)
                enc_mask = batch["encoder_attention_mask"].to(device)
                dec_ids = batch["decoder_input_ids"].to(device)
                labels = batch["labels"].to(device)

                memory = encoder(enc_ids, key_padding_mask=enc_mask)
                _, loss = decoder(dec_ids, memory, memory_key_padding_mask=enc_mask, targets=labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if device.type == "mps":
            torch.mps.empty_cache()

        dt = time.time() - t0
        print(f"epoch {epoch+1}/{epochs}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  lr {lr:.2e}  time {dt:.1f}s")  # pyright: ignore[reportPossiblyUnboundVariable]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "config": asdict(config),
                "epoch": epoch,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"  saved checkpoint (val_loss {val_loss:.4f})")


if __name__ == "__main__":
    main()
