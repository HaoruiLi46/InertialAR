import pandas as pd
import argparse
import numpy as np
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from config import InertialARConfig, SUPPORTED_MODEL_NAMES, normalize_model_name
from trainer_ddp import Trainer, TrainerConfig
from model import InertialAR
import math
import re
import torch.distributed as dist
import random
import logging
import warnings
import pickle
import torch._dynamo
from splitters import split_b3lyp_indices, split_dataset_by_indices, split_qm9_indices
from InertialTransformer.datasets import DatasetInertialSeqQM9, DatasetInertialSeqDrug, DatasetInertialSeqB3LYP
from InertialTransformer.datasets.tokenization_dictionary import ATOM_TYPE_MAPPINGS
from inertial_ar_dataset import InertialARDataset
from torch.utils.data import DataLoader as DataLoader_pytorch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split(dataset, data_root, args):
    if args.dataset_type == "qm9":
        train_idx, valid_idx, test_idx = split_qm9_indices(len(dataset), seed=args.seed)
        train_dataset, valid_dataset, test_dataset = split_dataset_by_indices(
            dataset, train_idx, valid_idx, test_idx
        )
        print("using QM9 split")
    elif args.dataset_type == "b3lyp":
        train_idx, valid_idx, test_idx = split_b3lyp_indices(len(dataset), seed=args.seed)
        train_dataset, valid_dataset, test_dataset = split_dataset_by_indices(
            dataset, train_idx, valid_idx, test_idx
        )
        print("using B3LYP split")
    else:
        raise ValueError(f"Invalid dataset type for splitting: {args.dataset_type}")
    print(len(train_dataset), "\t", len(valid_dataset), "\t", len(test_dataset))
    
    return train_dataset, valid_dataset, test_dataset

def run(args, global_rank=None, local_rank=None):
    os.environ["WANDB_MODE"] = "dryrun"

    print("making dataset... \n")
    data_root = os.path.join(args.data_root, args.dataset)

    if args.dataset_type == "qm9":
        print("using QM9 dataset")
        dataset = DatasetInertialSeqQM9(
            data_root,
            dataset=args.dataset,
            task=args.task,
            max_seq_len=args.max_len,
            position_pad_token=args.position_pad_token,
            position_eos_token=args.position_eos_token,
        )
    elif args.dataset_type == "drug":
        print("using Drug dataset")
        dataset = DatasetInertialSeqDrug(
            data_root,
            position_pad_token=args.position_pad_token,
            position_eos_token=args.position_eos_token,
            max_seq_len=args.max_len,
        )
    elif args.dataset_type == "b3lyp":
        print("using B3LYP dataset")
        dataset = DatasetInertialSeqB3LYP(
            data_root,
            max_seq_len=args.max_len,
            position_pad_token=args.position_pad_token,
            position_eos_token=args.position_eos_token,
        )
    else:
        raise ValueError("Invalid dataset type.")
    num_classes = dataset.num_classes
    if args.dataset_type == "qm9":
        train_dataset, valid_dataset, test_dataset = split(dataset, data_root, args)
    elif args.dataset_type == "drug":
        train_idx = dataset.train_idx
        train_dataset = [dataset[i] for i in train_idx]
        del dataset
    elif args.dataset_type == "b3lyp":
        train_dataset, valid_dataset, test_dataset = split(dataset, data_root, args)
    else:
        raise ValueError("Invalid dataset type.")
    print(f"using {args.dataset_type} conditional dataset")
    train_dataset = InertialARDataset(train_dataset)

    data_collator = train_dataset.collater

    print(f"train dataset size: {len(train_dataset)}")

    if args.load_checkpoint_path:
        load_checkpoint_path = args.load_checkpoint_path
    else:
        load_checkpoint_path = None

    print("loading model... \n")
    args.model_3d = normalize_model_name(args.model_3d)

    # Keep train and generation vocabularies aligned for atom-token prediction.
    dataset_key = args.dataset.lower()
    atom_map = ATOM_TYPE_MAPPINGS.get(dataset_key, {})
    allowed_token_ids = sorted(list(atom_map.keys())) if len(atom_map) > 0 else None
    print(f"allowed token ids: {allowed_token_ids}")

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
        ## Allowed tokens for atom type (mask invalid classes during training/inference)
        allowed_token_ids=allowed_token_ids,
        )
    if (not args.dist) or (args.dist and global_rank == 0):
        print_config_detailed(mconf, "Model Configuration (InertialARConfig)")

    model = InertialAR(mconf)
    if args.pre_model_path is not None:
        print("loading pretrained model: ", args.pre_model_path)
        model_path = args.pre_model_path
        try:
            if torch.cuda.is_available():
                if local_rank is not None:
                    map_location = f'cuda:{local_rank}'
                else:
                    map_location = f'cuda:{torch.cuda.current_device()}'
                pretrained_checkpoint = torch.load(model_path, map_location=map_location)
            else:
                pretrained_checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(pretrained_checkpoint, dict) and 'model_state_dict' in pretrained_checkpoint:
                model.load_state_dict(pretrained_checkpoint['model_state_dict'], strict=True)
            else:
                model.load_state_dict(pretrained_checkpoint, strict=True)
            print(f"Successfully loaded pretrained model from {model_path}")
        except FileNotFoundError:
            print(f"Error: Pretrained model file not found at {model_path}")
            raise
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            raise
    if load_checkpoint_path is not None:
        if not os.path.exists(load_checkpoint_path):
            print(f"Error: Checkpoint file not found at {load_checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {load_checkpoint_path}")
        print(f"Checkpoint file found: {load_checkpoint_path}")
        print("Note: Checkpoint will be loaded by trainer with full state (optimizer, scaler, etc.)")

    if args.amp_dtype == 'bf16':
        print('AMP dtype: bf16 | Using autocast(bf16) without GradScaler; keeping weights in FP32')
    elif args.amp_dtype == 'fp16':
        print('AMP dtype: fp16 | Using autocast(fp16) with GradScaler; keeping weights in FP32')

    print('total params:', sum(p.numel() for p in model.parameters()))

    if args.use_torch_compile:
        try:
            print("Compiling model with torch.compile...")
            model = torch.compile(
                model,
                mode="reduce-overhead",
                dynamic=True,
                fullgraph=False 
            )
            print("Model compilation successful!")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")
            print("Continuing without compilation...")
            torch._dynamo.reset()

    tconf = TrainerConfig(
        ## Baisic config
        max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, amp_dtype=args.amp_dtype, grad_norm_clip=args.grad_norm_clip,
        ## EMA
        use_ema=not args.no_ema, ema_decay=args.ema_decay,
        ## lr config
        lr_scheduler=args.lr_scheduler, min_lr=args.min_lr, warmup_updates=args.warmup_updates, warmup_ratio=args.warmup_ratio,
        ## ckpt saving config
        ckpt_path=args.output_model_dir, save_start_epoch=args.save_start_epoch, save_interval_epoch=args.save_interval_epoch,
        log_interval=args.log_interval,
        ## data config
        data_collator=data_collator, num_workers=args.num_workers, 
        ## load & resume training config
        load_checkpoint_path=load_checkpoint_path,
        ## model config
        model_3d=args.model_3d, dist=args.dist, rank=global_rank, local_rank=local_rank, seed=args.seed,
        ## validation config
        validate_interval=args.validate_interval,
        validate_interval_updates=args.validate_interval_updates,
        validate_with_ema=args.validate_with_ema,
        best_checkpoint_metric=args.best_checkpoint_metric,)
    
    if (not args.dist) or (args.dist and global_rank == 0):
        print_config_detailed(tconf, "Training Configuration (TrainerConfig)")
    trainer = Trainer(model, train_dataset, None, tconf)
    df = trainer.train()

def print_config_detailed(config, config_name="Config"):
    print(f"\n{'='*60}")
    print(f"{config_name:^60}")
    print(f"{'='*60}")
    attrs = []
    for attr_name in dir(config):
        if not attr_name.startswith('_'):
            try:
                attr_value = getattr(config, attr_name)
                if not callable(attr_value):
                    attrs.append((attr_name, attr_value))
            except:
                continue
    attrs.sort(key=lambda x: x[0])
    for attr_name, attr_value in attrs:
        if isinstance(attr_value, (list, tuple)) and len(attr_value) > 10:
            value_str = f"{type(attr_value).__name__}(length={len(attr_value)})"
        elif isinstance(attr_value, torch.Tensor):
            value_str = f"Tensor{tuple(attr_value.shape)} on {attr_value.device}"
        elif isinstance(attr_value, str) and len(attr_value) > 50:
            value_str = attr_value[:47] + "..."
        else:
            value_str = str(attr_value)
        print(f"{attr_name:<25}: {value_str}")
    print(f"{'='*60}\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ## Transformer Config
    parser.add_argument('--model_3d', type=str, choices=SUPPORTED_MODEL_NAMES, default="InertialAR")
    parser.add_argument('--max_len', type=int, default=32,
                        help="max_len", required=False)
    parser.add_argument('--n_layer', type=int, default=4,
                        help="number of layers", required=False)
    parser.add_argument("--n_layer_diffusion", type=int, default=4,
                        help="number of layers for diffusion", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=512,
                        help="embedding dimension", required=False)
    ## Dataset Config
    parser.add_argument('--num_workers', type=int, default=8,
                        help="number of workers for data loaders", required=False)
    parser.add_argument('--dataset_type', type=str, choices=["qm9", "drug", "b3lyp"], default="qm9")
    parser.add_argument('--node_class', type=int, default=119)
    parser.add_argument('--position_pad_token', type=int, default=0)
    parser.add_argument('--position_eos_token', type=int, default=0)
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default='QM9',help="name of the dataset to train on", required=False)
    parser.add_argument('--task', type=str, default='alpha')
    ## Conditional Token for CFG Config
    parser.add_argument('--cls_token_num', type=int, default=1)
    parser.add_argument('--class_dropout_prob', type=float, default=0.1)
    ## Bias Term Config
    parser.add_argument("--use_bias_term", type=int, default=0)
    parser.add_argument("--num_gaussian_kernels", type=int, default=128)
    ## Training Config
    parser.add_argument('--seed', type=int, default=42,
                        help="seed", required=False)
    parser.add_argument("--loss_weight_pos", type=float, default=0.7, help='Weight for position prediction loss')
    parser.add_argument('--learning_rate', type=float,
                        default=4e-4, help="learning rate (max learning rate)", required=False)
    parser.add_argument('--amp_dtype', type=str, choices=['fp16', 'bf16', 'none'], default='fp16',
                        help="mixed-precision dtype: fp16, bf16, or none (fp32)")
    parser.add_argument('--grad_norm_clip', type=float, default=1.0,
                        help="gradient norm clipping. smaller values mean stronger normalization.", required=False)
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help="weight decay for optimizer. paper uses 1e-5", required=False)
    parser.add_argument('--max_epochs', type=int, default=2000,
                        help="total epochs", required=False)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument('--recycle', type=int, default=3,
                        help="recycle", required=False)
    parser.add_argument('--batch_size', type=int, default=32,
                        help="batch size", required=False)
    ## EMA Config
    parser.add_argument('--no_ema', action='store_true', default=False,
                        help="disable exponential moving average of parameters", required=False)
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help="EMA decay rate", required=False)
    ## LR Scheduler Config
    parser.add_argument('--lr_scheduler', type=str, default='CosineLRSchedule',
                        help="learning rate scheduler: CosineLRSchedule, WarmupConstantLRSchedule", required=False)
    parser.add_argument('--min_lr', type=float, default=1e-9,
                        help="minimum learning rate", required=False)
    parser.add_argument('--warmup_updates', default=0, type=int, metavar='N',
                        help='warmup the learning rate linearly for the first N updates. Overrides warmup_ratio if set.')
    parser.add_argument('--warmup_ratio', default=0.1, type=float, metavar='N',
                        help='warmup the learning rate linearly for the first N-percent of total updates.')
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
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--scale", type=float, default=10)
    ## Nystrom Config
    parser.add_argument("--max_distance", type=float, default=4)
    parser.add_argument("--EPS", type=float, default=1e-8)
    parser.add_argument("--RBF_num_sigma", type=int, default=64)
    parser.add_argument("--nystrom_anchor_count", type=int, default=None)
    parser.add_argument("--nystrom_anchor_budget", type=int, default=None)
    parser.add_argument("--nystrom_sigma_scale", type=float, default=1.0)
    parser.add_argument("--nystrom_anchor_seed", type=int, default=0)
    ## DiffLoss Config
    parser.add_argument("--dim_diffmlp", type=int, default=768)
    parser.add_argument("--layers_diffmlp", type=int, default=6)
    parser.add_argument("--num_t_samples", type=int, default=3)
    parser.add_argument("--P_mean", type=float, default=-1.2)
    parser.add_argument("--P_std", type=float, default=1.2)
    parser.add_argument("--sigma_data", type=float, default=1.4)
    parser.add_argument('--loss_type', type=str, choices=['per_atom', 'per_molecule'], default='per_atom',
                        help="loss type: 'per_atom' or 'per_molecule'")
    ## Output Config
    parser.add_argument('--output_model_dir', type=str, default='./inertial_ar/weights/',
                        help="output model directory", required=False)
    parser.add_argument('--log_interval', type=int, default=500,
                        help="log training status every N updates")
    parser.add_argument('--save_start_epoch', type=int, default=500,
                        help="save model start epoch", required=False)
    parser.add_argument('--save_interval_epoch', type=int, default=50,
                        help="save model epoch interval", required=False)
    ## Resume & Pretrain Config
    parser.add_argument('--load_checkpoint_path', type=str, default=None,
                        help="Path to load training checkpoint (if resuming training)", required=False)
    parser.add_argument('--pre_root_path', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--pre_model_path', default=None,
                        help="Path to the pretrain model", required=False)
    ## DDP
    parser.add_argument('--dist', action='store_true',
                        default=False, help='use torch.distributed to train the model in parallel')
    ## Validation Config
    parser.add_argument('--validate_interval', type=int, default=20,
                        help="validate every N epochs. Used if validate_interval_updates is 0.")
    parser.add_argument('--validate_interval_updates', type=int, default=-1,
                        help="validate every N updates. Overrides validate_interval if > 0.")
    parser.add_argument('--best_checkpoint_metric', type=str, default='loss',
                        help="metric to monitor for saving the best checkpoint (e.g., 'loss', 'loss_position')")
    parser.add_argument('--validate_with_ema', action='store_true', default=True,
                        help="use the EMA model for validation")
    ## Torch Compile Config
    parser.add_argument('--use_torch_compile', action='store_true', default=False,
                        help="use torch.compile for optimization")

    args = parser.parse_args()


    if args.dist:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        if global_rank == 0:
            print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
            print(args)
        
        print(f"DDP initialized: rank={global_rank}, world_size={world_size}, local_rank={local_rank}")
        
        set_seed(args.seed)
        run(args, global_rank=global_rank, local_rank=local_rank)

        dist.destroy_process_group()
    else:
        # Print available GPU count
        if torch.cuda.is_available():
            print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("CUDA is not available. Running on CPU.")
        print(args)

        set_seed(args.seed)
        run(args, global_rank=None, local_rank=None)
