from functools import partial
import torch
from torch import nn
import torch.utils
if __package__:
    from .layers import RMSNorm, DropPath, FeedForward
    from .attention import Attention
else:
    from layers import RMSNorm, DropPath, FeedForward
    from attention import Attention


class TransformerLayer(nn.Module):

    def __init__(
        self,
        dim,
        heads,
        mlp_dim,
        residual_dropout=0.1,
        attn_dropout=0.1,
        rope=None,
        deterministic=False,
        causal=False,
        use_qk_layernorm=False,
        apply_selective_rope="selective",
    ):
        super().__init__()

        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=heads,
            dropout=attn_dropout,
            bias=False,
            rope=rope,
            deterministic=deterministic,
            causal=causal,
            use_qk_layernorm=use_qk_layernorm,
            apply_selective_rope=apply_selective_rope,
        )
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(
            dim,
            mlp_dim,
            256,
        )

        self.dropout = DropPath(residual_dropout)

    def forward(
        self,
        x,
        cu_lens=None,
        max_len=None,
        index_pos=None,
        mask=None,
    ):
        x = x + self.dropout(
            self.attn(self.attn_norm(x), cu_lens, max_len, index_pos, mask)
        )
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        residual_dropout=0.1,
        attn_dropout=0.1,
        rope=None,
        checkpoint_activation_threshold=100000,
        deterministic=False,
        causal=False,
        use_qk_layernorm=False,
        apply_selective_rope="selective",
    ):
        super().__init__()
        self.checkpoint_activation_threshold = checkpoint_activation_threshold
        self.layers = nn.ModuleList([])
        droppath_probs = [x.item() for x in torch.linspace(0, residual_dropout, depth)]
        for i in range(depth):
            self.layers.append(
                TransformerLayer(
                    dim,
                    heads,
                    mlp_dim,
                    droppath_probs[i],
                    attn_dropout,
                    rope=rope,
                    deterministic=deterministic,
                    causal=causal,
                    use_qk_layernorm=use_qk_layernorm,
                    apply_selective_rope=apply_selective_rope,
                )
            )

    def forward(
        self,
        x,
        cu_lens=None,
        max_len=None,
        index_pos=None,
        mask=None,
    ):

        layers = [
            partial(
                b,
                cu_lens=cu_lens,
                max_len=max_len,
            )
            for b in self.layers
        ]
        if (
            cu_lens is not None
            and cu_lens[-1] > self.checkpoint_activation_threshold
            and self.training
        ):
            for i, layer in enumerate(layers):
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        else:
            for i, layer in enumerate(self.layers):
                x = layer(
                    x,
                    cu_lens,
                    max_len,
                    index_pos,
                    mask,
                )

        return x
