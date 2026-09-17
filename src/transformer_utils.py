import math

import torch
from mps_flash_attn import replace_sdpa
from torch import nn
from torch.nn import functional as F

replace_sdpa()
device = torch.device("mps" if torch.mps.is_available() else "cpu")

class LayerNorm(nn.Module):
    def __init__(self, dim, bias = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps=1e-6)

class SelfAttention(nn.Module):
    def __init__(self, config, is_causal):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.is_causal = is_causal

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("Warning: scaled_dot_product_attention is not available, using slow attention")

        # this buffer is only actually used in the non-flash causal fallback path;
        # decoder self-attn needs dec_block_size, encoder self-attn doesn't use it at all
        block_size = config.dec_block_size if is_causal else config.enc_block_size
        self.bias: torch.Tensor
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                            .view(1, 1, block_size, block_size))

        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)


    def forward(self, x, key_padding_mask=None):
        # key_padding_mask: (B, T) with 1 = real token, 0 = pad. Only meaningful
        # for non-causal (encoder) self-attention -- causal decoder self-attention
        # already can't see trailing pad under right-padding, so this is a no-op
        # there even if passed.
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        attn_bias = None
        if key_padding_mask is not None and not self.is_causal:
            # (B, T) -> (B, 1, 1, T) additive bias, broadcast over heads and queries
            attn_bias = (1.0 - key_padding_mask[:, None, None, :].to(q.dtype)) * torch.finfo(q.dtype).min

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=self.attention_dropout.p if self.training else 0.0, is_causal=self.is_causal)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.is_causal:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            if attn_bias is not None:
                att = att + attn_bias
            att = F.softmax(att, dim=-1)
            att = self.attention_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

                    # output projection
        y = self.residual_dropout(self.c_proj(y))
        return y

class CrossAttention(nn.Module): # Inherit directly from nn.Module
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd

        # Dedicated projections for Cross Attention
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.c_kv = nn.Linear(self.n_embd, 2 * self.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("Warning: scaled_dot_product_attention is not available, using slow attention")

        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x, memory, memory_key_padding_mask=None):
        # memory_key_padding_mask: (B, T_mem) with 1 = real encoder token, 0 = pad.
        B, T, C = x.size()
        _, T_mem, C_mem = memory.size()

        # Q from decoder sequence (x), K and V from encoder sequence (memory)
        q = self.c_q(x)
        kv = self.c_kv(memory)
        k_mem, v_mem = kv.split(self.n_embd, dim=2)

        k_mem = k_mem.view(B, T_mem, self.n_heads, C_mem // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v_mem = v_mem.view(B, T_mem, self.n_heads, C_mem // self.n_heads).transpose(1, 2)

        attn_bias = None
        if memory_key_padding_mask is not None:
            attn_bias = (1.0 - memory_key_padding_mask[:, None, None, :].to(q.dtype)) * torch.finfo(q.dtype).min

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k_mem, v_mem, attn_mask=attn_bias, dropout_p=self.attention_dropout.p if self.training else 0.0, is_causal=False
            )
        else:
            att = (q @ k_mem.transpose(-2, -1)) * (1.0 / math.sqrt(k_mem.size(-1)))
            if attn_bias is not None:
                att = att + attn_bias
            att = F.softmax(att, dim=-1)
            att = self.attention_dropout(att)
            y = att @ v_mem

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.residual_dropout(self.c_proj(y))

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.atten = SelfAttention(config, is_causal=False)

    def forward(self, x, key_padding_mask=None):
        x = x + self.atten(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln3 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.atten = SelfAttention(config, is_causal=True)
        self.cross_atten = CrossAttention(config)

    def forward(self, x, encoder_output, memory_key_padding_mask=None):
        x = x + self.atten(self.ln1(x))  # causal self-attn: no key_padding_mask needed under right-padding
        x = x + self.cross_atten(self.ln2(x), encoder_output, memory_key_padding_mask=memory_key_padding_mask)
        x = x + self.mlp(self.ln3(x))
        return x
