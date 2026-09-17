import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F



# build encoder decoder blocks
@dataclass
class Config:
    enc_block_size: int = 128     # covers NL max of 79 with margin
    dec_block_size: int = 384     # covers FOL max of 313 with margin
    enc_n_layers: int = 6
    dec_n_layers: int = 6
    n_heads: int = 6
    n_embd: int = 512 
    dropout: float = 0.1
    bias: bool = False
    enc_vocab: int = 30522        # set to len(nl_tokenizer)
    dec_vocab: int = 30522        # set to len(fol_tokenizer) 

class EncoderBlock(nn.Module):
    config: Config
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        assert config.enc_vocab is not None
        assert config.enc_block_size is not None

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.enc_vocab, config.n_embd),
            'wpe': nn.Embedding(config.enc_block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Encoder(config) for _ in range(config.enc_n_layers)]),
            'ln_f': LayerNorm(config.n_embd, bias=config.bias),
        })

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.enc_n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, key_padding_mask=None):
        device = idx.device
        _, t = idx.size()
        assert t <= self.config.enc_block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.enc_block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer['wte'](idx)
        pos_emb = self.transformer['wpe'](pos)
        x = self.transformer['drop'](tok_emb + pos_emb)
        blocks: nn.ModuleList = self.transformer['h']  # pyright: ignore[reportAssignmentType]
        for block in blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.transformer['ln_f'](x)

        return x

class DecoderBlock(nn.Module):
    config: Config
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.dec_vocab, config.n_embd),
            'wpe': nn.Embedding(config.dec_block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Decoder(config) for _ in range(config.dec_n_layers)]),
            'ln_f': LayerNorm(config.n_embd, bias=config.bias),
        })
        self.lm_head = nn.Linear(config.n_embd, config.dec_vocab, bias=False)
        self.transformer['wte'].weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(3 * config.dec_n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, memory, memory_key_padding_mask=None, targets=None):
        device = idx.device
        _, t = idx.size()
        assert t <= self.config.dec_block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.dec_block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer['wte'](idx)
        pos_emb = self.transformer['wpe'](pos)
        x = self.transformer['drop'](tok_emb + pos_emb)
        blocks: nn.ModuleList = self.transformer['h']  # pyright: ignore[reportAssignmentType]
        for block in blocks:
            x = block(x, memory, memory_key_padding_mask=memory_key_padding_mask)
        x = self.transformer['ln_f'](x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
