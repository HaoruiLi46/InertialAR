from ast import arg
import pandas as pd
import argparse
import os
import sys
import random
import copy
import time
import torch
from torch.nn import functional as F
import numpy as np
import math
import re
import json
import random
from tqdm import tqdm
import time
import csv

from model import InertialAR

from config import InertialARConfig, SUPPORTED_MODEL_NAMES, normalize_model_name
from diffusion_loss import edm_sampler
from InertialTransformer.datasets.tokenization_dictionary import ATOM_TYPE_MAPPINGS, VOCAB
from InertialTransformer.datasets.functional_group_mapping import load_condition_mapping, analyze_condition_distribution
# Try to import EMA, but don't fail if it's not available
try:
    from ema import ExponentialMovingAverageModel
    EMA_AVAILABLE = True
except ImportError:
    EMA_AVAILABLE = False
    print("Warning: EMA module not found, EMA model weights cannot be loaded.")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model(model_path, config, device, use_ema=True, ema_decay=0.999):
    """
    Loads a model from a checkpoint, precisely mirroring the trainer's logic for
    robustness, including EMA weight swapping and DDP prefix handling.
    """
    # Use map_location for efficient loading
    map_location = device if "cuda" in str(device) and torch.cuda.is_available() else 'cpu'
    
    print(f"Loading checkpoint from {model_path} to device '{map_location}'...")
    checkpoint = torch.load(model_path, map_location=map_location)

    # 1. Create the single, main model instance.
    model = InertialAR(config)
    
    state_dict_to_load = None

    # 2. Decide which state_dict to prepare: EMA or standard.
    if use_ema and EMA_AVAILABLE and 'ema_state_dict' in checkpoint:
        print("Found EMA state in checkpoint. Preparing to apply EMA weights...")
        try:
            # Create a temporary EMA wrapper on the base model structure
            ema_wrapper = ExponentialMovingAverageModel(args=None, model=model, decay=ema_decay, is_flattened=False)
            # Load the EMA state into the wrapper
            ema_wrapper.load_state_dict(checkpoint['ema_state_dict'])
            
            # Get the state_dict from the averaged model *within* the wrapper
            if hasattr(ema_wrapper, 'model_ema'):
                state_dict_to_load = ema_wrapper.model_ema.state_dict()
                print("Successfully prepared EMA weights for loading.")
            else: # Compatibility for other EMA implementations
                state_dict_to_load = ema_wrapper.ema_model.state_dict()
                print("Successfully prepared EMA weights for loading (compat mode).")

        except Exception as e:
            print(f"Error preparing EMA weights: {e}. Falling back to standard weights.")
            state_dict_to_load = None # Ensure fallback
            
    # Fallback if EMA is not used, not available, or failed.
    if state_dict_to_load is None:
        if use_ema:
             print("Warning: EMA not applied. Loading standard model weights instead.")
        else:
             print("Loading standard model weights.")
        
        state_dict_key = 'model_state_dict'
        if state_dict_key in checkpoint:
            state_dict_to_load = checkpoint[state_dict_key]
        else:
            # Handle checkpoints that are just a raw state_dict without keys
            print(f"Warning: '{state_dict_key}' not in checkpoint. Assuming checkpoint is a raw state_dict.")
            state_dict_to_load = checkpoint

    # 3. Handle 'module.' prefix from DDP-trained models (CRITICAL STEP)
    if state_dict_to_load:
        # Check if the state_dict is from a DDP model
        is_ddp_model = all(k.startswith('module.') for k in state_dict_to_load.keys())
        if is_ddp_model:
            print("Stripping 'module.' prefix from state_dict keys.")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict_to_load.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict_to_load = new_state_dict
    
    # 4. Load the prepared and cleaned state_dict into our main model instance.
    if state_dict_to_load:
        model.load_state_dict(state_dict_to_load, strict=True)
        print("Model weights loaded into the main model instance successfully.")
    else:
        print("Error: Could not find any valid model weights to load. Exiting.")
        exit(1)
    
    # 5. Prepare the model for generation and return it.
    model.to(device)
    model.eval()
    return model


### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True, allowed_token_ids=None):        
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    # Optional: restrict logits to allowed atom and stop tokens.
    if allowed_token_ids is not None:
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, allowed_token_ids] = False
        logits[mask] = -float("Inf")
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs

# --- New Batch Generation Function ---
def generate_batch(model, idx, positions, args, sample_logits=False):

    model.eval()
    torch.set_grad_enabled(False)
    torch.cuda.empty_cache()
    device = idx.device
    B, T_init = idx.size()
    max_new_tokens = args.max_len - 1
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    eos_token_id = VOCAB["<eos>"]
    pad_token_id = VOCAB["<pad>"]
    # Generation evaluates the full prefix sequence at each step.

    generated_idx = idx.clone()
    generated_positions = positions.clone()
    cond_idx = idx.clone()

    
    if args.cfg_scale > 1.0:
        cond_null = torch.ones_like(cond_idx) * model.config.num_classes
        generated_idx = torch.cat([generated_idx, cond_null])
        generated_positions = torch.cat([generated_positions, generated_positions])
        cond_idx = torch.cat([cond_idx, cond_null])
        has_stopped = torch.zeros(2*B, dtype=torch.bool, device=device)
    else:
        generated_idx = generated_idx
        generated_positions = generated_positions
        cond_idx = cond_idx
        has_stopped = torch.zeros(B, dtype=torch.bool, device=device)

    with torch.no_grad():
        pbar = tqdm(range(max_new_tokens), desc="Generating")
        # Whitelist dataset-specific atom tokens and EOS for stopping.
        allowed_token_ids = None
        try:
            atom_map = ATOM_TYPE_MAPPINGS.get(args.dataset.lower(), None)
            if atom_map is not None and len(atom_map) > 0:
                allowed_set = set(atom_map.keys()) | {int(eos_token_id)}
                allowed_token_ids = torch.tensor(sorted(allowed_set), device=device)
                print(allowed_token_ids)
        except Exception:
            allowed_token_ids = None
        for i in pbar:
            if has_stopped.all():
                pbar.set_description(f"All sequences stopped at step {i}")
                break

            if i == 0:
                logits, z = model.inference(input_ids=None, positions=generated_positions, condition_id=generated_idx, allowed_token_ids=allowed_token_ids)
            else:
                logits, z = model.inference(input_ids=generated_idx, positions=generated_positions, condition_id=cond_idx, allowed_token_ids=allowed_token_ids)

            cond = z[:, -1, :]  # (B, 3)

            if args.cfg_scale > 1.0:
                cond_logits, uncond_logits = torch.split(logits, len(logits) // 2, dim=0) 
                logits = uncond_logits + (cond_logits - uncond_logits) * args.cfg_scale
                next_token_ids, _ = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature, sample_logits=False, allowed_token_ids=allowed_token_ids)
                next_pos_pred = edm_sampler(
                    net=model.net,
                    cond=cond,
                    cfg_scale=args.cfg_scale,
                    randn_like=torch.randn_like,
                    num_steps=model.config.num_steps,
                    S_churn=model.config.S_churn, S_min=model.config.S_min, 
                    S_max=model.config.S_max, S_noise=model.config.S_noise,
                    sigma_min=model.net.sigma_min,
                    sigma_max=model.net.sigma_max,
                )
                next_token_ids = torch.cat([next_token_ids, next_token_ids], dim=0)
            else:
                next_token_ids, _ = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature, sample_logits=False, allowed_token_ids=allowed_token_ids)
                next_pos_pred = edm_sampler(
                    net=model.net,
                    cond=cond,
                    cfg_scale=args.cfg_scale,
                    randn_like=torch.randn_like,
                    num_steps=model.config.num_steps,
                    S_churn=model.config.S_churn, S_min=model.config.S_min, 
                    S_max=model.config.S_max, S_noise=model.config.S_noise,
                    sigma_min=model.net.sigma_min,
                    sigma_max=model.net.sigma_max,
                )

            # Append to sequences
            generated_idx = torch.cat((generated_idx, next_token_ids), dim=1)          # (B, T+1)
            generated_positions = torch.cat((generated_positions, next_pos_pred.unsqueeze(1)), dim=1)  # (B, T+1, 3)

            # --- Stopping condition ---
            just_generated_stop_token = (next_token_ids.squeeze(-1) == eos_token_id) | \
                                        (next_token_ids.squeeze(-1) == pad_token_id)
            has_stopped.logical_or_(just_generated_stop_token)

    # --- Process Results ---
    # Truncate sequences that generated eos_token_id
    if args.cfg_scale > 1.0:
        generated_idx, _ = torch.split(generated_idx, B, dim=0)
        generated_positions, _ = torch.split(generated_positions, B, dim=0)
    
    final_idx_list = []
    final_pos_list = []
    for i in range(B):
        # Find the first occurrence of eos_token_id *after* the initial sequence
        eos_indices = torch.where(generated_idx[i, T_init:] == eos_token_id)[0] # Search for EOS token
        pad_indices = torch.where(generated_idx[i, T_init:] == pad_token_id)[0] # Search for PAD token
        
        if len(pad_indices) > 0:
            pad_pos = pad_indices[0].item() + T_init
        else:
            pad_pos = generated_idx.size(1)
        
        
        if len(eos_indices) > 0:
            eos_pos = eos_indices[0].item() + T_init
        else:
            # No EOS token found, sequence reached max_new_tokens
            eos_pos = generated_idx.size(1)
            
        end_pos = min(pad_pos, eos_pos)

        final_idx_list.append(generated_idx[i, T_init:end_pos]) # Truncate up to (but not including) EOS
        final_pos_list.append(generated_positions[i, T_init:end_pos, :]) # Truncate positions accordingly

    # Return lists of tensors (variable length)
    return final_idx_list, final_pos_list

def sample_conditions_from_distribution(condition_distribution, num_samples, device='cpu'):
    if not condition_distribution:
        raise ValueError("condition_distribution must not be empty.")

    # Split condition ids and their sampling weights.
    ids = [item[0] for item in condition_distribution]
    weights = [item[1] for item in condition_distribution]

    # Convert weights to a tensor on the target device.
    weights_tensor = torch.tensor(weights, dtype=torch.float, device=device)

    # Sample condition indices with replacement.
    sampled_indices = torch.multinomial(weights_tensor, num_samples, replacement=True)

    # Map sampled indices back to condition ids.
    condition_ids_tensor = torch.tensor(ids, device=device)
    sampled_ids = condition_ids_tensor[sampled_indices]

    # Return shape: (num_samples, 1).
    return sampled_ids.unsqueeze(1)

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # --- Generation Params ---
    parser.add_argument('--cond_id', type=int, default=-1)
    parser.add_argument('--cond_pos', type=float, default=0.0)
    parser.add_argument('--num_generate', type=int, default=1000, help="Total number of molecules to generate.", required=True)
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size for generation.", required=False)
    parser.add_argument('--seed', type=int, default=42, help="Random seed.", required=False)
    parser.add_argument("--loss_weight_pos", type=float, default=0.8, help='Weight for position prediction loss')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--recycle', type=int, default=1,
                        help="recycle", required=False)
    parser.add_argument('--device', type=str, default='auto', help="Device to use for generation.", required=False)
    ## cfg config
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    ## ckpt config
    parser.add_argument('--ckpt_path', default='./inertial_ar/weights', required=True, help="Directory to load model checkpoints from.") # Added for clarity
    parser.add_argument('--epoch', type=int, default=2000, help="Epoch number of the checkpoint to load (0 for .pt, >0 for _epX.pt).", required=False) # Allow 0 for base name
    ## diffloss config
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--S_churn", type=float, default=40)
    parser.add_argument("--S_min", type=float, default=0.05)
    parser.add_argument("--S_max", type=float, default=50)
    parser.add_argument("--S_noise", type=float, default=1)
    ## dataset config
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="Drug")
    parser.add_argument("--position_pad_token", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--position_eos_token", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--node_class", type=int, default=119)
    ## model config
    parser.add_argument('--model_3d', type=str, choices=SUPPORTED_MODEL_NAMES, default="InertialAR")
    parser.add_argument('--max_len', type=int, default=32,
                        help="max_len", required=False)
    parser.add_argument('--n_layer', type=int, default=6,
                        help="number of layers", required=False)
    parser.add_argument("--n_layer_diffusion", type=int, default=6,
                        help="number of layers for diffusion", required=False)
    parser.add_argument('--n_head', type=int, default=12,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=768,
                        help="embedding dimension", required=False)
    ## Conditional Token for CFG Config
    parser.add_argument('--cls_token_num', type=int, default=1)
    parser.add_argument('--class_dropout_prob', type=float, default=0.1)
    ## EMA Config
    parser.add_argument('--no_ema', action='store_true', default=False,
                        help="disable exponential moving average of parameters", required=False)
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help="EMA decay rate", required=False)
    ## Bias Term Config
    parser.add_argument("--use_bias_term", type=int, default=0)
    parser.add_argument("--num_gaussian_kernels", type=int, default=128)
    ## LayerNorm Config    
    parser.add_argument('--use_qk_layernorm', action='store_true', default=False,
                        help="use qk-layernorm for attention stability")
    parser.add_argument('--use_layernorm_atom_type', action='store_true', default=False,
                        help="use layernorm for atom type")
    parser.add_argument('--use_layernorm_position', action='store_true', default=False,
                        help="use layernorm for position")
    ## RoPE Config
    parser.add_argument('--apply_selective_rope', type=str, choices=['full', 'selective', 'separate'], default='selective',
                        help="RoPE application mode: 'full' (apply to all features), 'selective' (apply only to first half), 'separate' (separate projections for each half)")    
    parser.add_argument('--apply_second_rope', action='store_true', default=False)
    parser.add_argument("--rope_theta", type=float, default=100.0)
    parser.add_argument("--scale", type=float, default=1)
    ## Nystrom Config
    parser.add_argument("--max_distance", type=float, default=6.5)
    parser.add_argument("--EPS", type=float, default=1e-8)
    parser.add_argument("--RBF_num_sigma", type=int, default=1)
    parser.add_argument("--nystrom_anchor_count", type=int, default=None)
    parser.add_argument("--nystrom_anchor_budget", type=int, default=None)
    parser.add_argument("--nystrom_sigma_scale", type=float, default=1.0)
    parser.add_argument("--nystrom_anchor_seed", type=int, default=0)
    ## DiffLoss Config
    parser.add_argument("--dim_diffmlp", type=int, default=1024)
    parser.add_argument("--layers_diffmlp", type=int, default=6)
    parser.add_argument("--num_t_samples", type=int, default=4)
    parser.add_argument("--P_mean", type=float, default=-1.2)
    parser.add_argument("--P_std", type=float, default=1.2)
    parser.add_argument("--sigma_data", type=float, default=1.4)
    parser.add_argument('--loss_type', type=str, choices=['per_atom', 'per_molecule'], default='per_atom',
                        help="loss type: 'per_atom' or 'per_molecule'")

    args = parser.parse_args()
    set_seed(args.seed)

    # --- Device Setup ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif "cuda" in args.device and not torch.cuda.is_available():
        print(f"Warning: Requested device '{args.device}' but CUDA is not available. Using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Define Padding and EOS Token IDs ---
    pad_token_id = VOCAB["<pad>"]
    eos_token_id = VOCAB["<eos>"]
    print(f"Using PAD & EOS Token ID: {pad_token_id}, {eos_token_id}")



    # --- Prepare ALL Initial Inputs Before Batching ---
    data_root = os.path.join(args.data_root, args.dataset, "processed")
    num_samples = args.num_generate
    print(f"Sampling {args.num_generate} starting atoms (before batching)...")
    print(f"Loading dataset from: {data_root}")

    condition_ids, pattern_to_id_dict, id_to_pattern_dict = load_condition_mapping(data_root)
    with open(os.path.join(data_root, "functional_group_header.csv"), 'r', newline='') as f:
        reader = csv.reader(f)
        # Read the header row.
        headers = next(reader)

    num_classes = len(pattern_to_id_dict)
    index_split_path = os.path.join(data_root, "index_split.pt")
    if os.path.exists(index_split_path):
        split = torch.load(index_split_path)
        train_idx_list = split.get("train_idx", [])
        if len(train_idx_list) == 0:
            condition_distribution = analyze_condition_distribution(condition_ids, headers, id_to_pattern_dict, min_freq_filter=None)
        else:
            train_condition_ids = [condition_ids[i] for i in train_idx_list]
            condition_distribution = analyze_condition_distribution(train_condition_ids, headers, id_to_pattern_dict, min_freq_filter=None)
    else:
        condition_distribution = analyze_condition_distribution(condition_ids, headers, id_to_pattern_dict, min_freq_filter=None)

    if args.cond_id == -1:
        cond_idx = sample_conditions_from_distribution(condition_distribution, num_samples, device=device)
    else:
        cond_idx = torch.full((num_samples, 1), fill_value=args.cond_id, dtype=torch.long, device=device)

    cond_pos = torch.full((num_samples, 1, 3), fill_value=args.cond_pos, dtype=torch.float, device=device)

    # --- Model Configuration ---

    mconf = InertialARConfig(
        ## Transformer
        model_3d=args.model_3d,
        vocab_size=args.node_class, block_size=args.max_len, dropout=args.dropout,
        loss_weight_pos=args.loss_weight_pos,
        latent_dim=args.n_embd, n_head=args.n_head, 
        n_layer=args.n_layer, n_layer_diffusion=args.n_layer_diffusion,  
        recycle=args.recycle, 
        ## RoPE
        apply_selective_rope=args.apply_selective_rope, apply_second_rope=args.apply_second_rope, 
        scale=args.scale, rope_theta=args.rope_theta,
        ## Nystrom
        max_distance=args.max_distance, EPS=args.EPS, num_sigma=args.RBF_num_sigma,
        nystrom_anchor_count=args.nystrom_anchor_count,
        nystrom_anchor_budget=args.nystrom_anchor_budget,
        nystrom_sigma_scale=args.nystrom_sigma_scale,
        nystrom_anchor_seed=args.nystrom_anchor_seed,
        ## layernorm
        use_qk_layernorm=args.use_qk_layernorm, use_layernorm_atom_type=args.use_layernorm_atom_type, use_layernorm_position=args.use_layernorm_position,
        ## diffloss
        dim_diffmlp=args.dim_diffmlp, layers_diffmlp=args.layers_diffmlp,
        loss_type=args.loss_type, num_t_samples=args.num_t_samples,
        P_mean=args.P_mean, P_std=args.P_std, sigma_data=args.sigma_data,
        ## Conditional Token for CFG
        num_classes=num_classes, class_dropout_prob=args.class_dropout_prob, cls_token_num=args.cls_token_num,
        ## Bias Term Config
        use_bias_term=args.use_bias_term, num_gaussian_kernels=args.num_gaussian_kernels,
        ## sampling config
        num_steps=args.num_steps, 
        S_churn=args.S_churn, S_min=args.S_min, S_max=args.S_max, S_noise=args.S_noise,
        )
    args.model_3d = normalize_model_name(args.model_3d)
    if args.epoch == 0:
        model_filename = 'checkpoint_best.pt'
        gen_suffix = f'best_k{args.top_k}_t{args.temperature:.1f}'
    else:
        model_filename = f'epoch_{args.epoch}.pt'
        gen_suffix = f'epoch_{args.epoch}_k{args.top_k}_t{args.temperature:.1f}'

    model_path = os.path.join(args.ckpt_path, model_filename)
    if not os.path.exists(model_path):
         print(f"Error: Model checkpoint not found at {model_path}")
         exit(1)

    print(f"Loading model from: {model_path}")
    use_ema_flag = not args.no_ema
    model = load_model(model_path, mconf, device, use_ema=use_ema_flag, ema_decay=args.ema_decay)
    print("Model loaded successfully.")
    print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')
    
    # --- Batch Generation Loop ---
    all_generated_idx_list = []
    all_generated_pos_list = []
    num_batches = math.ceil(args.num_generate / args.batch_size)
    total_generated_count = 0
    generation_start_time = time.time()

    print(f"Starting generation of {args.num_generate} molecules in {num_batches} batches of size {args.batch_size}...")

    for i in tqdm(range(num_batches), desc="Generating Batches"):
        batch_start_idx = i * args.batch_size
        batch_end_idx = min((i + 1) * args.batch_size, args.num_generate)
        current_batch_size = batch_end_idx - batch_start_idx

        # Slice the pre-sampled starting points for the current batch
        start_idx = cond_idx[batch_start_idx:batch_end_idx] # (B, 1)
        start_position = cond_pos[batch_start_idx:batch_end_idx] # (B, 1, 3)

        # Generate for the current batch
        try:
            generated_idx_list, generated_pos_list = generate_batch(
                model=model,
                idx=start_idx,
                positions=start_position,
                args=args
            )
            # Append results from this batch to the overall lists
            all_generated_idx_list.extend(generated_idx_list)
            all_generated_pos_list.extend(generated_pos_list)
            total_generated_count += len(generated_idx_list)

        except Exception as e:
            print(f"\nError during generation in batch {i+1}: {e}")
            print(f"Skipping batch {i+1}. Check model, inputs, or memory.")
            # Decide if you want to continue or exit on batch failure
            # For now, we continue, but the final count might be lower.
            continue # Skip to the next batch

    generation_end_time = time.time()
    print(f"\nGeneration finished. Successfully processed {total_generated_count}/{args.num_generate} molecules across all batches in {generation_end_time - generation_start_time:.2f} seconds.")

    # --- Process and Save Combined Results ---
    if total_generated_count == 0:
        print("No molecules were generated successfully across all batches.")
    else:
        # --- Restructure data for saving ---
        molecules_to_save = []
        print("Processing generated data for saving...")
        for i in tqdm(range(total_generated_count), desc="Processing Results"):
            # Convert tensors back to CPU and NumPy arrays
            # Handle potential errors if a batch failed and lists are shorter than expected
            try:
                atom_types_tensor = all_generated_idx_list[i]
                coordinates_tensor = all_generated_pos_list[i]
                atom_types_np = atom_types_tensor.cpu().numpy()
                coordinates_np = coordinates_tensor.cpu().numpy()

                molecule_data = {
                    'atom_types': atom_types_np,   # (num_atoms,)
                    'coordinates': coordinates_np  # (num_atoms, 3)
                }
                molecules_to_save.append(molecule_data)
            except IndexError:
                print(f"Warning: Could not process result at index {i}, likely due to a previous batch failure.")
                continue

        # --- Save results ---
        output_dir = args.ckpt_path # Save in the same directory as the checkpoint
        # Construct a more informative output filename using the final successful count
        sampling_suffix = (
            f"steps{args.num_steps}_schurn{args.S_churn:g}_smin{args.S_min:g}_"
            f"smax{args.S_max:g}_snoise{args.S_noise:g}_topp{args.top_p:g}"
        )
        output_filename_final = f"generated_{gen_suffix}_{sampling_suffix}_{len(molecules_to_save)}_{args.cfg_scale}cfg_cond_ID_{args.cond_id}.npz" # Use actual saved count
        output_filename_txt = f"generated_{gen_suffix}_{sampling_suffix}_{len(molecules_to_save)}_{args.cfg_scale}cfg_cond_ID_{args.cond_id}.txt" # Use actual saved count
        output_file_path = os.path.join(output_dir, output_filename_final)
        output_file_path_txt = os.path.join(output_dir, output_filename_txt)
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        print(f"Saving {len(molecules_to_save)} generated molecules to {output_file_path}")
        print(f"Example command: DATASET_NAME=drug bash scripts/evaluation/evaluate_generated.sh {output_file_path} {output_file_path_txt}")
        try:
            np.savez_compressed(
                output_file_path,
                molecules=np.array(molecules_to_save, dtype=object)
            )
            print("Saving complete.")
        except Exception as e:
            print(f"Error saving results to {output_file_path}: {e}")
