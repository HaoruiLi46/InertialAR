"""InertialAR molecular autoregressive model."""
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

if __package__:
    from .config import InertialARConfig
    from .rope import RoPEnD
    from .transformer_encoder import Transformer
    from .layers import (
        Embedding,
        Linear,
        TaskHead,
        LabelEmbedder,
        RMSNorm,
        GaussianLayer,
    )
    from .diffusion_loss import EDMPrecond, EDMLoss_per_atom, EDMLoss_per_molecule
else:
    from config import InertialARConfig
    from rope import RoPEnD
    from transformer_encoder import Transformer
    from layers import (
        Embedding,
        Linear,
        TaskHead,
        LabelEmbedder,
        RMSNorm,
        GaussianLayer,
    )
    from diffusion_loss import EDMPrecond, EDMLoss_per_atom, EDMLoss_per_molecule
from InertialTransformer.datasets.tokenization_dictionary import VOCAB


logger = logging.getLogger(__name__)


def pairwise_squared_distances(x, y):
    x_norm = x.pow(2).sum(dim=1, keepdim=True)  # [N, 1]
    y_norm = y.pow(2).sum(dim=1, keepdim=True).T  # [1, M]
    dist = x_norm + y_norm - 2 * x @ y.T
    return dist

class InertialAR(nn.Module):
    """Autoregressive molecular generator with atom-token and coordinate heads."""

    def __init__(self, config: InertialARConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.padding_token_id = config.vocab_size - 1
        self.loss_weight_pos = config.loss_weight_pos
        self.head_dim = config.final_repr_dim // config.n_head
        self.block_size = config.block_size
        self.recycle = config.recycle if hasattr(config, 'recycle') else 1
        self.EPS = config.EPS
        self.nystrom_MLP = Linear(config.num_anchor * config.num_sigma, config.num_anchor, init="bert")
        self.register_buffer("anchor_points", config.anchor_points)
        self.register_buffer("sigma_list", config.sigma_list)
        self.register_buffer("L_inv_list", config.L_inv_list)

        self.tok_emb = Embedding(config.vocab_size, config.idx_repr_dim)
        self.tok_emb_proj = Linear(config.idx_repr_dim, config.final_repr_dim, init="bert")
        self.index_pos_emb = Embedding(config.block_size, config.final_repr_dim)

        self.cls_token_num = config.cls_token_num
        self.cls_embedding = LabelEmbedder(config.num_classes, config.final_repr_dim, config.class_dropout_prob)

        self.scale = config.scale
        self.rope_theta = config.rope_theta
        self.rope = RoPEnD(
            self.head_dim,
            self.rope_theta,
            n=3,
        )
        self.transformer = Transformer(
            dim=config.final_repr_dim,
            depth=config.n_layer,
            heads=config.n_head,
            mlp_dim=config.final_repr_dim * 4,
            rope=self.rope,
            residual_dropout=config.resid_pdrop,
            attn_dropout=config.attn_pdrop,
            checkpoint_activation_threshold=100000,
            deterministic=False,
            causal=True,
            use_qk_layernorm=config.use_qk_layernorm,
            apply_selective_rope=config.apply_selective_rope,
        )
        self.transformer_diff = Transformer(
            dim=config.final_repr_dim,
            depth=config.n_layer_diffusion,
            heads=config.n_head,
            mlp_dim=config.final_repr_dim * 4,
            rope=self.rope if config.apply_second_rope else None,
            residual_dropout=config.resid_pdrop,
            attn_dropout=config.attn_pdrop,
            checkpoint_activation_threshold=100000,
            deterministic=False,
            causal=True,
            use_qk_layernorm=config.use_qk_layernorm,
            apply_selective_rope="full",
        )
        self.net = EDMPrecond(
            z_dim=config.final_repr_dim,
            hidden_dim=config.dim_diffmlp,
            num_hidden_layers=config.layers_diffmlp,
            dropout_rate=config.diffusion_dropout,
            sigma_data=config.sigma_data,
        )
        if config.loss_type == 'per_atom':
            self.loss_fn = EDMLoss_per_atom(P_mean=config.P_mean, P_std=config.P_std, sigma_data=config.sigma_data)
        elif config.loss_type == 'per_molecule':
            self.loss_fn = EDMLoss_per_molecule(P_mean=config.P_mean, P_std=config.P_std, sigma_data=config.sigma_data)
        else:
            raise ValueError(f"Invalid loss type: {config.loss_type}")
        if config.use_layernorm_atom_type:
            self.layer_norm_first = RMSNorm(config.final_repr_dim)
        if config.use_layernorm_position:
            self.layer_norm_second = RMSNorm(config.final_repr_dim)
        self.head = TaskHead(config.final_repr_dim, config.vocab_size, dropout=0.0)
        self.atom_types = config.vocab_size  
        self.edge_types = config.vocab_size * config.vocab_size
        if config.use_bias_term:
            self.gbf = GaussianLayer(K=config.num_gaussian_kernels, edge_types=self.edge_types) 
            self.bias_proj = Linear(config.num_gaussian_kernels, config.n_head, init="bert")

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        return

    def configure_optimizers(self, train_config):
        """
        Separates parameters into groups for weight decay, following Uni-Core's principles.
        This approach is more streamlined and robust than manually managing whitelists/blacklists.
        """
        decay_params = []
        no_decay_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # Preserve the special case for position embedding (no weight decay)
            if name.startswith('index_pos_emb'):
                no_decay_params.append(p)
                continue
            # Apply Uni-Core's general rule: no decay for biases and 1D parameters.
            # This covers LayerNorm, RMSNorm, and all bias terms automatically.
            if p.ndim == 1 or name.endswith(".bias"):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        # Note: The original special handling for `bias_proj.weight` is removed.
        # Its weight (ndim=2) is correctly assigned to the decay group by the rule above.
        optim_groups = [
            {"params": decay_params, "weight_decay": train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        # Verify that all parameters have been assigned
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_grouped_params = sum(p.numel() for g in optim_groups for p in g['params'])
        assert num_params == num_grouped_params, "Some parameters were not assigned to an optimizer group"

        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def get_block_size(self):
        return self.block_size

    def forward(self, input_ids, positions, targets, targets_positions, index_pos, cu_seqlens, max_len, condition_id, **kwargs):

        cond_embeddings = self.cls_embedding(condition_id, train=self.training)[:,:self.cls_token_num]
        condition_mask = (index_pos == 0)  # [TotalTokens]
        condition_indices = torch.where(condition_mask)[0]

        batch_size = len(cu_seqlens) - 1
        assert len(condition_indices) == batch_size, f"Condition token count mismatch"

        idx = input_ids
        # Training with packed sequences (from collator). Tensors are shaped [TotalTokens, ...], lacking a batch dimension.
        self.rope.calc_and_cache(positions * self.scale)
        position_embeddings = self.index_pos_emb(index_pos)

        norm = torch.norm(positions, dim=-1, keepdim=True)  # [T, 1] or [B, max_len, 1] 
        weighted_positions = torch.where(norm > self.EPS, positions, positions) # [T, 3] or [B, max_len, 3]
        weighted_normalized_positions = torch.where(norm > self.EPS, positions / norm, positions)  # [T, 3] or [B, max_len, 3]
        token_embeddings = self.tok_emb(idx) # [T, dim] or [B, max_len, dim]

        positions_flat = positions.reshape(-1, 3)  # [TotalTokens, 3]
        D_C = pairwise_squared_distances(positions_flat, self.anchor_points)  # [TotalTokens, num_anchor]
        C_all = torch.exp(-D_C.unsqueeze(0) / (2 * self.sigma_list**2))  # [num_sigma, TotalTokens, num_anchor]
        nystrom_flat_all_sigmas = C_all @ self.L_inv_list.transpose(-2, -1)
        nystrom_features_flat = nystrom_flat_all_sigmas.permute(1, 0, 2).reshape(positions_flat.shape[0], -1)
        nystrom_features = nystrom_features_flat.view(*positions.shape[:-1], -1)
        masked_nystrom_features = torch.where(norm > self.EPS, nystrom_features, 0)
        nystrom_positions = self.nystrom_MLP(masked_nystrom_features)  # [..., num_anchor]
        # In amp, inputs from DataLoader (positions) can be float32, while model layers (tok_emb) are bf16. 
        weighted_positions = weighted_positions.to(token_embeddings.dtype)
        weighted_normalized_positions = weighted_normalized_positions.to(token_embeddings.dtype)
        nystrom_positions = nystrom_positions.to(token_embeddings.dtype)
        node_repr = torch.cat([token_embeddings, weighted_positions, weighted_normalized_positions, nystrom_positions], dim=-1)
        final_atom_repr = node_repr.clone()
        cond_embeddings = cond_embeddings.squeeze(1)
        if final_atom_repr.dtype != cond_embeddings.dtype:
            cond_embeddings = cond_embeddings.to(final_atom_repr.dtype)
        final_atom_repr[condition_indices] = cond_embeddings
        h = F.dropout(final_atom_repr + position_embeddings, self.config.embd_pdrop, self.training)
        h_init = h.detach()
        
        for i in range(self.recycle):
            h = self.transformer(
                h,
                cu_lens=cu_seqlens,
                max_len=max_len,
            )

        if self.config.use_layernorm_atom_type:
            h_norm = self.layer_norm_first(h)
        else:
            h_norm = h
        logits = self.head(h_norm)
        if hasattr(self.config, 'allowed_token_ids') and self.config.allowed_token_ids is not None and len(self.config.allowed_token_ids) > 0:
            allowed_list = list(self.config.allowed_token_ids) + [VOCAB["<eos>"]]
            allowed = torch.tensor(sorted(set(allowed_list)), device=logits.device, dtype=torch.long)
            invalid_mask_cols = torch.ones(logits.size(-1), dtype=torch.bool, device=logits.device)
            invalid_mask_cols[allowed] = False
            logits[:, invalid_mask_cols] = -float('inf')

        atom_type = targets
        z_atom = self.tok_emb_proj(self.tok_emb(atom_type))
        z = h + z_atom + h_init

        for i in range(self.recycle):
            z = self.transformer_diff(
                z,
                cu_lens=cu_seqlens,
                max_len=max_len,
            )

        if self.config.use_layernorm_position:
            z = self.layer_norm_second(z)
        else:
            z = z

        num_tokens = targets_positions.shape[0]
        pred_pos = None

        # Token prediction loss (cross entropy)
        loss_token = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
        loss_token = (loss_token).sum() / num_tokens

        # Position prediction loss (diffusion) - a cu_seqlens is passed for per-molecule sigma sampling
        if self.config.loss_type == 'per_atom':
            loss_position_raw = self.loss_fn(self.net, targets_positions, z, num_t_samples=self.config.num_t_samples) # Shape: [TotalTokens*N_t, 3]
        elif self.config.loss_type == 'per_molecule':
            loss_position_raw = self.loss_fn(self.net, targets_positions, z, cu_seqlens, num_t_samples=self.config.num_t_samples) # Shape: [TotalTokens*N_t, 3]
        else:
            raise ValueError(f"Invalid loss type: {self.config.loss_type}")
        
        pos_loss_per_token = loss_position_raw.sum(dim=-1) # Shape: [TotalTokens*N_t]
        num_expanded_tokens = loss_position_raw.shape[0]

        loss_position = pos_loss_per_token.sum() / num_expanded_tokens

        if torch.isnan(loss_position) or torch.isinf(loss_position):
            print(f" Position loss is NaN/Inf, zeroing it out but keeping graph. Value: {loss_position}")
            print(f"  Z stats: min={z.min():.3f}, max={z.max():.3f}, mean={z.mean():.3f}")
            print(f"  Targets_positions stats: min={targets_positions.min():.3f}, max={targets_positions.max():.3f}")
            # Keep the graph connected so DDP can still reduce gradients.
            loss_position = torch.sum(z) * 0.0

        total_loss = (1 - self.loss_weight_pos) * loss_token + self.loss_weight_pos * loss_position
        
        return logits, pred_pos, total_loss, loss_token, loss_position

    def inference(self, input_ids=None, positions=None, condition_id=None, **kwargs):
        
        device = positions.device
        if input_ids is not None:
            idx = input_ids
            b, t = idx.shape
            assert t <= self.block_size, f"Cannot forward, model block size ({self.block_size}) is smaller than input sequence length ({t})."
        elif condition_id is not None:
            b, t = condition_id.shape
            assert t == 1, "t should be 1 for sampling"

        self.rope.calc_and_cache(positions * self.scale)
        if condition_id is not None and input_ids is None:
            node_repr = self.cls_embedding(condition_id, train=self.training)[:,:self.cls_token_num]
            node_repr = node_repr.squeeze(1)
        elif condition_id is not None and input_ids is not None:
            input_ids = input_ids[:, 1:]
            positions = positions[:, 1:, :]

            norm = torch.norm(positions, dim=-1, keepdim=True)  # [T, 1] or [B, max_len, 1] 
            weighted_positions = torch.where(norm > self.EPS, positions, positions) # [T, 3] or [B, max_len, 3]
            weighted_normalized_positions = torch.where(norm > self.EPS, positions / norm, positions)  # [T, 3] or [B, max_len, 3]
            token_embeddings = self.tok_emb(input_ids) # [T, dim] or [B, max_len, dim]

            positions_flat = positions.reshape(-1, 3)  # [TotalTokens, 3]
            D_C = pairwise_squared_distances(positions_flat, self.anchor_points)  # [TotalTokens, num_anchor]
            C_all = torch.exp(-D_C.unsqueeze(0) / (2 * self.sigma_list**2))  # [num_sigma, TotalTokens, num_anchor]
            nystrom_flat_all_sigmas = C_all @ self.L_inv_list.transpose(-2, -1)
            nystrom_features_flat = nystrom_flat_all_sigmas.permute(1, 0, 2).reshape(positions_flat.shape[0], -1)
            nystrom_features = nystrom_features_flat.view(*positions.shape[:-1], -1)
            masked_nystrom_features = torch.where(norm > self.EPS, nystrom_features, 0)
            nystrom_positions = self.nystrom_MLP(masked_nystrom_features)  # [..., num_anchor]
            # In amp, inputs from DataLoader (positions) can be float32, while model layers (tok_emb) are bf16. 
            weighted_positions = weighted_positions.to(token_embeddings.dtype)
            weighted_normalized_positions = weighted_normalized_positions.to(token_embeddings.dtype)
            nystrom_positions = nystrom_positions.to(token_embeddings.dtype)
            node_repr = torch.cat([token_embeddings, weighted_positions, weighted_normalized_positions, nystrom_positions], dim=-1)
            cond_repr = self.cls_embedding(condition_id, train=self.training)[:,:self.cls_token_num]
            cond_repr = cond_repr.squeeze(1)
            node_repr = torch.cat([cond_repr, node_repr], dim=1)
        # Full-sequence generation uses positional embeddings over the current prefix.
        pos_ids = torch.arange(t, device=device)
        position_embeddings = self.index_pos_emb(pos_ids).unsqueeze(0).expand(b, -1, -1)

        h = F.dropout(node_repr + position_embeddings, self.config.embd_pdrop, self.training) ## [B, T, D]
        h_init = h.detach()
        for i in range(self.recycle):
            h = self.transformer(
                h,
                cu_lens=None,              # No packed sequences in batch inference
                max_len=None,              # No packed sequences in batch inference
            )
            
        if self.config.use_layernorm_atom_type:
            h_norm = self.layer_norm_first(h)
        else:
            h_norm = h
        logits = self.head(h_norm)
        logits_for_pred_type = logits.clone()
        allowed_token_ids = kwargs.get('allowed_token_ids', None)
        if allowed_token_ids is not None:
            allowed_list = allowed_token_ids.tolist() if hasattr(allowed_token_ids, 'tolist') else list(allowed_token_ids)
            allowed_list = list(set(allowed_list + [VOCAB["<eos>"]]))
            allowed = torch.tensor(sorted(allowed_list), device=logits.device, dtype=torch.long)
            invalid_mask_cols = torch.ones(logits.size(-1), dtype=torch.bool, device=logits.device)
            invalid_mask_cols[allowed] = False
            logits_for_pred_type[:, :, invalid_mask_cols] = -float('inf')

        atom_type = logits_for_pred_type.argmax(dim=-1).long()
        z_atom = self.tok_emb_proj(self.tok_emb(atom_type))
        z = h + z_atom + h_init

        for i in range(self.recycle):
            z = self.transformer_diff(
                z,
                cu_lens=None,              # No packed sequences in batch inference
                max_len=None,              # No packed sequences in batch inference
            )
        assert z.ndim == 3, "z's shape should be [b, t, d]"

        if self.config.use_layernorm_position:
            z = self.layer_norm_second(z)
        else:
            z = z


        return logits, z
