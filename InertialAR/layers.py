import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init: str = "default",
    ):
        super(Linear, self).__init__(d_in, d_out, bias=bias)

        self.use_bias = bias

        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            self._glorot_uniform_init()
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            self._normal_init()
        elif init == "bert":
            self._bert_init()
        elif init == "final":
            self._zero_init(False)
        else:
            raise ValueError("Invalid init method.")

    def _trunc_normal_init(self, scale=1.0):
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _glorot_uniform_init(self):
        nn.init.xavier_uniform_(self.weight, gain=1)

    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")

    def _bert_init(self, std=0.02):
        nn.init.normal_(self.weight, mean=0.0, std=std)


class Embedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
    ):
        super(Embedding, self).__init__(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self._normal_init()

        if padding_idx is not None:
            self.weight.data[self.padding_idx].zero_()

    def _normal_init(self, std=0.02):
        nn.init.normal_(self.weight, mean=0.0, std=std)


class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = prob

    def forward(self, x):
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return f"prob={self.drop_prob}"


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(dim, hidden_dim, bias=False, init="bert")
        self.w2 = Linear(hidden_dim, dim, bias=False, init="final")
        self.w3 = Linear(dim, hidden_dim, bias=False, init="bert")

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = Embedding(1, K)
        self.stds = Embedding(1, K)

        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        
        self.mul = Embedding(edge_types, 1)
        self.bias = Embedding(edge_types, 1)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)
        
    def forward(self, x, edge_types):
        """
        x: [B, L, L] pairwise atom-distance matrix.
        """
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)        
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)  # [B, L, L, K]

        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5    
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings

class TaskHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.dense = Linear(embed_dim, embed_dim, bias=False, init="bert")
        self.out = Linear(embed_dim, output_dim, bias=False, init="final")

    def forward(self, x, **kwargs):
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.dense(x)
        x = F.gelu(x)
        x = self.out(x)
        return x


class FourierFeatureMapping:
    def __init__(
        self,
        input_dims,
        num_freqs,
        include_input=True,
        log_sampling=True,
        periodic_fns=[torch.sin, torch.cos],
    ):
        self.include_input = include_input
        self.periodic_fns = periodic_fns
        max_freq_log2 = num_freqs - 1
        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, steps=num_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq_log2, steps=num_freqs)
        self.freq_bands = freq_bands.unsqueeze(0).repeat(input_dims, 1)

    def forward(self, x):
        freq_bands = self.freq_bands.to(x.device)
        x_proj = x @ freq_bands
        x_proj = torch.cat([fn(x_proj) for fn in self.periodic_fns], dim=-1)
        if self.include_input:
            x_proj = torch.cat([x, x_proj], dim=-1)
        return x_proj


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class AdaNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), Linear(dim, 2 * dim))

    def forward(self, x, c):
        if c is None:
            return self.norm_final(x)
        scale, shift = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return x

class RMSNorm(nn.Module):
    """
    An RMSNorm implementation that falls back to manual implementation if F.rms_norm is not available.
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
            
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        else:
            self.register_parameter('weight', None)

    def _manual_rms_norm(self, x):
        """Manual implementation of RMS normalization"""
        # Compute the mean of squares along the last dimensions
        mean_squares = x.pow(2).mean(dim=tuple(range(-len(self.normalized_shape), 0)), keepdim=True)
        # Compute RMS normalization
        rms = torch.rsqrt(mean_squares + self.eps)
        normalized = x * rms
        
        # Apply elementwise affine transformation if enabled
        if self.elementwise_affine:
            normalized = normalized * self.weight
            
        return normalized

    def forward(self, x):
        # Try to use the optimized functional API from PyTorch 2.1+ if available
        if hasattr(F, 'rms_norm') and self.elementwise_affine:
            try:
                return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
            except Exception:
                # Fall back to manual implementation
                return self._manual_rms_norm(x)
        else:
            # Use manual implementation
            return self._manual_rms_norm(x)

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"
