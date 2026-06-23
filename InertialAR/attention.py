import math 
import logging

import torch
from torch import Tensor, nn
if __package__:
    from .layers import Linear, RMSNorm
else:
    from layers import Linear, RMSNorm
from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_func
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=16)
def get_causal_mask(seq_q, seq_k, device):
    offset = seq_k - seq_q
    i = torch.arange(seq_q, device=device).unsqueeze(1)
    j = torch.arange(seq_k, device=device).unsqueeze(0)
    causal_mask = (j > (offset + i)).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    return causal_mask

class Attention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        bias=False,
        rope=None,
        deterministic=False,
        causal=False,
        use_qk_layernorm=False,
        apply_selective_rope="full",
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        
        self.apply_selective_rope = apply_selective_rope
        if self.apply_selective_rope == "selective" or self.apply_selective_rope == "separate":
            self.rope_dim = embed_dim // 2
        elif self.apply_selective_rope == "full":
            self.rope_dim = embed_dim
            
        self.q_proj = Linear(self.rope_dim, self.rope_dim, bias=bias, init="bert")
        self.k_proj = Linear(self.rope_dim, self.rope_dim, bias=bias, init="bert")
        self.v_proj = Linear(self.rope_dim, self.rope_dim, bias=bias, init="bert")
        
        if self.apply_selective_rope == "separate":
            nystrom_dim = self.embed_dim - self.rope_dim
            self.q_proj_separate = Linear(nystrom_dim, nystrom_dim, bias=bias, init="bert")
            self.k_proj_separate = Linear(nystrom_dim, nystrom_dim, bias=bias, init="bert")
            self.v_proj_separate = Linear(nystrom_dim, nystrom_dim, bias=bias, init="bert")
        
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, init="final")
        self.rope = rope
        self.scale = self.head_dim**-0.5
        self.deterministic = deterministic
        self.causal = causal
        self.use_qk_layernorm = use_qk_layernorm

        if self.use_qk_layernorm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        query,
        cu_lens,
        max_len,
        index_pos,
        mask,
    ) -> Tensor:

        if self.apply_selective_rope == "selective" or self.apply_selective_rope == "separate":
            query_rope = query[..., : self.embed_dim // 2]
            query_nystrom = query[..., self.embed_dim // 2 :]

            q_rope_proj = self.q_proj(query_rope)
            k_rope_proj = self.k_proj(query_rope)
            v_rope_proj = self.v_proj(query_rope)

            query_size = query.size()
            rope_head_dim = self.embed_dim  // self.num_heads

            q_rope = q_rope_proj.view(*query_size[:-1], self.num_heads // 2, rope_head_dim)
            k_rope = k_rope_proj.view(*query_size[:-1], self.num_heads // 2, rope_head_dim)
            v_rope = v_rope_proj.view(*query_size[:-1], self.num_heads // 2, rope_head_dim)

            if self.rope is not None:
                q_rope, k_rope = self.rope.apply_qk(q_rope, k_rope)

            # Keep autocast dtype so FlashAttention receives fp16/bf16 tensors.
            query_nystrom = query_nystrom.to(dtype=q_rope_proj.dtype)

            nystrom_head_dim = self.embed_dim // self.num_heads
            if self.apply_selective_rope == "selective":
                query_ny = query_nystrom.view(*query_size[:-1], self.num_heads // 2, nystrom_head_dim)

                q = torch.cat([q_rope, query_ny], dim=-2)
                k = torch.cat([k_rope, query_ny], dim=-2)
                v = torch.cat([v_rope, query_ny], dim=-2)
            elif self.apply_selective_rope == "separate":
                q_nystrom_proj = self.q_proj_separate(query_nystrom)
                k_nystrom_proj = self.k_proj_separate(query_nystrom)
                v_nystrom_proj = self.v_proj_separate(query_nystrom)
                
                q_nystrom = q_nystrom_proj.view(*query_size[:-1], self.num_heads // 2, nystrom_head_dim)
                k_nystrom = k_nystrom_proj.view(*query_size[:-1], self.num_heads // 2, nystrom_head_dim)
                v_nystrom = v_nystrom_proj.view(*query_size[:-1], self.num_heads // 2, nystrom_head_dim)
                
                q = torch.cat([q_rope, q_nystrom], dim=-2)
                k = torch.cat([k_rope, k_nystrom], dim=-2)
                v = torch.cat([v_rope, v_nystrom], dim=-2)

        elif self.apply_selective_rope == "full":
            # --- Standard RoPE Branch (applied to all features) ---
            query_size = query.size()
            assert query_size[-1] == self.embed_dim
            q, k, v = (
                self.q_proj(query).view(*query_size[:-1], self.num_heads, -1),
                self.k_proj(query).view(*query_size[:-1], self.num_heads, -1),
                self.v_proj(query).view(*query_size[:-1], self.num_heads, -1),
            )
            # In the standard case, RoPE is applied to the full vectors.
            if self.rope is not None:
                q, k = self.rope.apply_qk(q, k)

        if self.use_qk_layernorm:
            q = self.q_norm(q).to(v.dtype)
            k = self.k_norm(k).to(v.dtype)
        
        if cu_lens is not None and max_len is not None:
            out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_lens,
                cu_lens,
                max_len,
                max_len,
                (self.dropout if self.training else 0.0),
                deterministic=self.deterministic,
                causal=self.causal,
            )
        else:
            assert not self.training
            assert len(q.shape) == 4, (q.shape, query.shape)

            seq_q = q.shape[1]
            if seq_q >= 128 and q.dtype in [torch.float16, torch.bfloat16]:
                out = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    deterministic=self.deterministic,
                    causal=self.causal,
                )
            else:
                if seq_q >= 128:
                    print(f"Warning: Using standard attention fallback for dtype {q.dtype} (FlashAttention requires fp16/bf16)")

                q = q * self.scale
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)

                attn = q @ k.transpose(-1, -2)

                seq_k = attn.shape[-1]
                if self.causal and seq_q > 1:
                    mask = get_causal_mask(seq_q, seq_k, attn.device)
                    attn.masked_fill_(mask, float("-inf"))

                attn = torch.softmax(attn, dim=-1)
                out = attn @ v
                out = out.permute(0, 2, 1, 3).contiguous()

        out = out.view(*query_size[:-1], -1)
        return self.out_proj(out)
