import logging
import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)



class FourierEncoder(nn.Module):
    """
    Fourier feature encoder with an RFF mapping followed by a learnable MLP.
    """
    def __init__(self, num_channels: int, bandwidth: float, input_dim: int, hidden_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()

        self.num_channels = num_channels
        self.bandwidth = bandwidth
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.register_buffer('frequencies', torch.randn(self.input_dim, self.num_channels, dtype=dtype))
        self.register_buffer('phases', torch.rand(self.input_dim, self.num_channels, dtype=dtype))
        
        mlp_input_dim = input_dim * num_channels
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

    def _fourier_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor with per-dimension Fourier features.
        """
        original_shape = x.shape[:-1]
        
        assert x.shape[-1] == self.input_dim, f"Expected input_dim {self.input_dim}, but got {x.shape[-1]}"
        
        # x: (..., input_dim) -> (..., input_dim, 1)
        x_expanded = x.unsqueeze(-1) 

        projections = 2 * math.pi * (
            x_expanded * (self.bandwidth * self.frequencies) + self.phases
        ) # Shape: (..., input_dim, num_channels)

        fourier_features = math.sqrt(2) * torch.cos(projections) # Shape: (..., input_dim, num_channels)

        encoded_features = fourier_features.reshape(*original_shape, self.input_dim * self.num_channels)

        return encoded_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded_features = self._fourier_encoding(x)
        output = self.mlp(encoded_features)
        return output

    def get_output_dim(self) -> int:
        """
        Return the final output dimension.
        """
        return self.hidden_dim

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x.type_as(self.linear.weight))
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing


        self.pos_fourier_emb = FourierEncoder(num_channels=512, bandwidth=20, input_dim=in_channels, hidden_dim=model_channels)
        self.time_fourier_emb = FourierEncoder(num_channels=model_channels, bandwidth=1, input_dim=1, hidden_dim=model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(
                ResBlock(
                    model_channels,
                )
            )

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize position embedding MLP
        nn.init.normal_(self.pos_fourier_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.pos_fourier_emb.mlp[2].weight, std=0.02)
        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_fourier_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_fourier_emb.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        # --- Type Casting for Mixed-Precision Compatibility ---
        # To robustly handle mixed-precision training (e.g., model.half()), we must ensure
        # that input tensors for each module match the dtype of the module's weights.
        
        # 1. Process position input x (shape: [N, 3])
        # Cast x once to the target dtype, determined by a representative weight.
        x_casted = x.type_as(self.input_proj.weight)
        x_linear = self.input_proj(x_casted)
        x_fourier = self.pos_fourier_emb(x_casted)

        # 2. Process time input t (shape: [N])
        # Reshape t from [N] to [N, 1] for the FourierEncoder and cast its dtype.
        t_casted = t.unsqueeze(-1).type_as(self.time_fourier_emb.mlp[0].weight)
        t_fourier = self.time_fourier_emb(t_casted)
        
        # 3. Process condition input c (shape: [N, z_channels])
        # Cast c to the appropriate dtype for the embedding layer.
        c_linear = self.cond_embed(c.type_as(self.cond_embed.weight))

        # --- Combine Embeddings ---
        x = x_linear + x_fourier
        y = t_fourier + c_linear

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

class EDMPrecond(torch.nn.Module):
    def __init__(self,
        z_dim,
        hidden_dim,
        num_hidden_layers,
        dropout_rate,
        sigma_data      = 1.4, # Adjust based on your coordinate data std
        sigma_min       = 1e-4,
        sigma_max       = 80,
        use_fp16        = False,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        
        self.model = SimpleMLPAdaLN(
            in_channels=3,
            model_channels=self.hidden_dim,
            out_channels=3, # Predicting 3D coordinates
            z_channels=self.z_dim,
            num_res_blocks=self.num_hidden_layers, 
            grad_checkpointing=False
        )

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def forward(self, x, sigma, cond, force_fp32=True):
        """
        Process packed sequences directly without batch dimension.
        
        Args:
            x_tokens: (total_tokens, 3) - packed coordinate sequences
            sigma_tokens: (total_tokens, 1) - noise level for each token
            z_tokens: (total_tokens, z_dim) - conditioning for each token
        
        Returns:
            D_x_tokens: (total_tokens, 3) - denoised coordinates
        """
        x_tokens = x
        sigma_tokens = sigma
        z_tokens = cond
        # Preserve the input dtype for AMP unless fp32 is explicitly requested.
        if force_fp32:
            x_tokens = x_tokens.to(torch.float32)
        total_tokens, coord_dim = x_tokens.shape
        
        # Handle sigma's shape
        sigma_input = sigma_tokens.to(x_tokens.dtype if not force_fp32 else torch.float32)
        
        if sigma_input.ndim == 2 and sigma_input.shape == (total_tokens, 1):
            sigma_tokens = sigma_input
        else:
            raise ValueError(f"Unsupported sigma shape: {sigma_input.shape}. Expected ({total_tokens}, 1).")

        if force_fp32:
            dtype = torch.float32
        else:
            dtype = x_tokens.dtype
        
        # --- Preconditioning coefficients (token-wise) ---
        c_skip = self.sigma_data ** 2 / (sigma_tokens ** 2 + self.sigma_data ** 2) # (total_tokens, 1)
        c_out = sigma_tokens * self.sigma_data / (sigma_tokens ** 2 + self.sigma_data ** 2).sqrt() # (total_tokens, 1)
        c_in = 1 / (self.sigma_data ** 2 + sigma_tokens ** 2).sqrt() # (total_tokens, 1)
        
        c_noise_tokens = sigma_tokens.log() / 4 # (total_tokens, 1)

        # --- Prepare inputs for DiffMLP (no reshaping needed for packed format) ---
        x_in = (c_in * x_tokens).to(dtype)  # (total_tokens, 3)
        c_noise_in = c_noise_tokens.squeeze(-1).to(dtype)  # (total_tokens, )
        z_in = z_tokens.to(self.model.cond_embed.weight.dtype)

        # --- Model call ---
        F_x_tokens = self.model(x_in, c_noise_in, z_in)  # (total_tokens, 3)
        # In AMP environment, output dtype might differ from input dtype
        # Convert to expected dtype instead of strict assertion
        F_x_tokens = F_x_tokens.to(dtype)

        D_x_tokens = c_skip * x_tokens + c_out * F_x_tokens.to(x_tokens.dtype)  # (total_tokens, 3)
        
        return D_x_tokens


    def forward_with_cfg(self, x, sigma, cond, cfg_scale, force_fp32=True):
        D_x_tokens = self.forward(x, sigma, cond, force_fp32=force_fp32)
        cond_D_x, uncond_D_x = torch.split(D_x_tokens, len(D_x_tokens) // 2, dim=0)
        D_x = uncond_D_x + cfg_scale * (cond_D_x - uncond_D_x)
        D_x = torch.cat([D_x, D_x], dim=0)
        return D_x

    def forward_with_cfg_2_conditions(self, x, sigma, cond, first_cfg_scale, second_cfg_scale, force_fp32=True):
        D_x_tokens = self.forward(x, sigma, cond, force_fp32=force_fp32)
        first_cond_D_x, second_cond_D_x, uncond_D_x = torch.split(D_x_tokens, len(D_x_tokens) // 3, dim=0)
        D_x = uncond_D_x + first_cfg_scale * (first_cond_D_x - uncond_D_x) + second_cfg_scale * (second_cond_D_x - uncond_D_x)
        D_x = torch.cat([D_x, D_x, D_x], dim=0)
        return D_x

class EDMLoss_per_atom:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1.4):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, targets_positions, z, num_t_samples=1):
        """
        Apply diffusion loss to packed sequences.
        
        Args:
            targets_positions: (total_tokens, 3) - packed position sequences
            z: (total_tokens, z_dim) - packed conditioning sequences
            num_t_samples: number of independent t samples per token
            
        Returns:
            loss_per_coord: (total_tokens * num_t_samples, 3) - loss per coordinate
        """
        total_tokens, coord_dim = targets_positions.shape
        
        # --- Expand data for multiple t samples (token-wise expansion) ---
        if num_t_samples > 1:
            # Use .repeat() to correctly duplicate the entire packed sequence structure.
            # repeat_interleave would break the molecular atom sequences.
            expanded_targets_positions = targets_positions.repeat(num_t_samples, 1)
            expanded_z = z.repeat(num_t_samples, 1)
            effective_total_tokens = total_tokens * num_t_samples
        else:
            expanded_targets_positions = targets_positions
            expanded_z = z
            effective_total_tokens = total_tokens

        # --- Sigma sampling for each token ---
        rnd_normal = torch.randn([effective_total_tokens, 1], device=targets_positions.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp() # Shape: (effective_total_tokens, 1)

        # --- Weighting ---
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2 # Shape: (effective_total_tokens, 1)

        # --- Noise addition ---
        noise = torch.randn_like(expanded_targets_positions) * sigma
        y_noisy = expanded_targets_positions + noise # (effective_total_tokens, 3)

        # --- Network call with per-token sigmas ---
        D_yn = net(y_noisy, sigma, expanded_z) # (effective_total_tokens, 3)

        # --- Loss calculation ---
        loss_per_coord = weight * ((D_yn - expanded_targets_positions) ** 2) # (effective_total_tokens, 3)
        
        return loss_per_coord

class EDMLoss_per_molecule:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1.4):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, targets_positions, z, cu_seqlens, num_t_samples=1):
        """
        Apply diffusion loss with per-molecule sigma sampling.
        """
        assert cu_seqlens is not None, "cu_seqlens must be provided for per-molecule sigma sampling."
        
        total_tokens = targets_positions.shape[0]
        num_molecules = len(cu_seqlens) - 1
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]

        # --- Robustness fix for potential data collation bug ---
        # Ensure the sum of lengths from cu_seqlens matches the actual tensor length.
        # This handles cases where cu_seqlens[-1] might not equal total_tokens.
        calculated_len = torch.sum(lengths).item()
        if calculated_len != total_tokens:
            # Correct the length of the last molecule in the batch.
            lengths[-1] = lengths[-1] - (calculated_len - total_tokens)
            # Sanity check to ensure the correction was successful
            assert torch.sum(lengths).item() == total_tokens, "Length correction failed"

        if num_t_samples > 1:
            expanded_targets_positions = targets_positions.repeat(num_t_samples, 1)
            expanded_z = z.repeat(num_t_samples, 1)
            expanded_lengths = lengths.repeat(num_t_samples)
            expanded_num_molecules = num_molecules * num_t_samples
        else:
            expanded_targets_positions = targets_positions
            expanded_z = z
            expanded_lengths = lengths
            expanded_num_molecules = num_molecules

        rnd_normal_per_mol = torch.randn([expanded_num_molecules, 1], device=targets_positions.device)
        sigma_per_mol = (rnd_normal_per_mol * self.P_std + self.P_mean).exp()
        sigma = sigma_per_mol.repeat_interleave(expanded_lengths, dim=0)
        
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(expanded_targets_positions) * sigma
        y_noisy = expanded_targets_positions + noise
        
        D_yn = net(y_noisy, sigma, expanded_z)
        
        loss_per_coord = weight * ((D_yn - expanded_targets_positions) ** 2)
        
        return loss_per_coord

def edm_sampler(
    net, cond, cfg_scale=1.0, randn_like=torch.randn_like,
    num_steps=30, 
    S_churn=0, S_min=0.05, S_max=50, S_noise=1,
    sigma_min=1e-4, sigma_max=80, rho=7,
    force_fp32=True,
):
    """
    EDM sampler that works with packed sequences.
    
    Args:
        net: EDMPrecond network
        cond: (total_tokens, z_dim) - conditioning for each token
    """
    if cfg_scale != 1.0:
        noise = torch.randn(cond.shape[0] // 2, 3).to(cond.device)
        noise = torch.cat([noise, noise], dim=0)
        model_kwargs = dict(cond=cond, cfg_scale=cfg_scale)
        sample_fn = net.forward_with_cfg
    else:
        noise = torch.randn(cond.shape[0], 3).to(cond.device)
        model_kwargs = dict(cond=cond)
        sample_fn = net.forward
        
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Use float32 time-step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=cond.device)
    
    sigma_min = torch.tensor(sigma_min, dtype=torch.float32, device=cond.device)
    sigma_max = torch.tensor(sigma_max, dtype=torch.float32, device=cond.device)
    rho = torch.tensor(rho, dtype=torch.float32, device=cond.device)
    
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    t_steps = t_steps.to(torch.float32)

    # Main sampling loop.
    x_next = noise.to(torch.float32) * t_steps[0]
    total_tokens = noise.shape[0]
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur).to(torch.float32)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur).to(torch.float32)

        # Use the same sigma value for each token in the current step.
        t_hat_tokens = t_hat.expand(total_tokens, 1) # (total_tokens, 1)
        t_next_tokens = t_next.expand(total_tokens, 1) # (total_tokens, 1)

        # Euler step.
        denoised = sample_fn(x=x_hat, sigma=t_hat_tokens, **model_kwargs, force_fp32=force_fp32).to(torch.float32)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = sample_fn(x=x_next, sigma=t_next_tokens, **model_kwargs, force_fp32=force_fp32).to(torch.float32)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next



def edm_sampler_2_conditions(
    net, cond, first_cfg_scale=1.0, second_cfg_scale=1.0, randn_like=torch.randn_like,
    num_steps=30, 
    S_churn=0, S_min=0.05, S_max=50, S_noise=1,
    sigma_min=1e-4, sigma_max=80, rho=7,
    force_fp32=True,
):
    """
    EDM sampler that works with packed sequences.
    
    Args:
        net: EDMPrecond network
        cond: (total_tokens, z_dim) - conditioning for each token
    """
    if first_cfg_scale != 0.0 or second_cfg_scale != 0.0:
        noise = torch.randn(cond.shape[0] // 3, 3).to(cond.device)
        noise = torch.cat([noise, noise, noise], dim=0)
        model_kwargs = dict(cond=cond, first_cfg_scale=first_cfg_scale, second_cfg_scale=second_cfg_scale)
        sample_fn = net.forward_with_cfg_2_conditions
    else:
        noise = torch.randn(cond.shape[0], 3).to(cond.device)
        model_kwargs = dict(cond=cond)
        sample_fn = net.forward
        
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Use float32 time-step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=cond.device)
    
    sigma_min = torch.tensor(sigma_min, dtype=torch.float32, device=cond.device)
    sigma_max = torch.tensor(sigma_max, dtype=torch.float32, device=cond.device)
    rho = torch.tensor(rho, dtype=torch.float32, device=cond.device)
    
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    t_steps = t_steps.to(torch.float32)

    # Main sampling loop.
    x_next = noise.to(torch.float32) * t_steps[0]
    total_tokens = noise.shape[0]
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur).to(torch.float32)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur).to(torch.float32)

        # Use the same sigma value for each token in the current step.
        t_hat_tokens = t_hat.expand(total_tokens, 1) # (total_tokens, 1)
        t_next_tokens = t_next.expand(total_tokens, 1) # (total_tokens, 1)

        # Euler step.
        denoised = sample_fn(x=x_hat, sigma=t_hat_tokens, **model_kwargs, force_fp32=force_fp32).to(torch.float32)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = sample_fn(x=x_next, sigma=t_next_tokens, **model_kwargs, force_fp32=force_fp32).to(torch.float32)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next
