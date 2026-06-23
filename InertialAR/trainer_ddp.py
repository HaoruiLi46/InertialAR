import math
import logging
from tqdm import tqdm
import numpy as np
import copy
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler
import re
import sys
import pandas as pd
from rdkit import Chem
import os
import contextlib
import time
from lr_scheduler import CosineLRSchedule, WarmupConstantLRSchedule
import torch.distributed as dist
try:
    from ema import ExponentialMovingAverageModel
    EMA_AVAILABLE = True
except ImportError:
    EMA_AVAILABLE = False
    print("Warning: EMA not available, diffusion_loss module not found")

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 1e-5
    # new scheduler params
    min_lr = 1e-9
    warmup_updates = 0
    warmup_ratio = 0.06

    # logging and validation
    log_interval = 500
    validate_interval = 20
    validate_interval_updates = -1
    validate_with_ema = True
    save_interval_epoch = 50
    # checkpoint settings
    ckpt_path = None
    run_name = None
    num_workers = 16 # for DataLoader
    load_checkpoint_path = None
    best_checkpoint_metric = 'loss'
    amp_dtype = 'bf16'  # 'fp16', 'bf16', 'none'
    # DDP related
    dist = False
    rank = None
    seed = 42

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.num_updates = 0
        self.device = 'cpu'
        self.lr_scheduler = config.lr_scheduler
        self.data_collator = getattr(config, 'data_collator', None)
        if self.data_collator is not None:
            print("Trainer initialized with a data collator.")

        self.use_ema = getattr(config, 'use_ema', False) and EMA_AVAILABLE
        self.ema = None
        
        print('dist:', config.dist)
        if config.dist:
            self.device = config.rank
            self.model = self.model.to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.device],
                find_unused_parameters=False
            )
        elif torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        if self.use_ema:
            try:
                raw_model = self.model.module if hasattr(self.model, "module") else self.model
                self.ema = ExponentialMovingAverageModel(
                    args=None,
                    model=raw_model,
                    decay=getattr(config, 'ema_decay', 0.9999),
                    is_flattened=False
                )
                print(f"EMA initialized with decay={getattr(config, 'ema_decay', 0.9999)}")
            except Exception as e:
                print(f"Warning: Failed to initialize EMA: {e}")
                self.ema = None
                self.use_ema = False
        else:
            print("EMA disabled")

    def save_checkpoint(self, epoch, model, best_loss, optimizer, scaler, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        raw_model = model.module if hasattr(model, "module") else model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'num_updates': self.num_updates,
            'best_loss': best_loss,
        }
        
        if self.use_ema and self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        if self.config.dist:
            if self.device == 0:
                torch.save(checkpoint, save_path)
        else:
            torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, load_path, optimizer, scaler):
        try:
            if torch.cuda.is_available():
                if self.config.dist:
                    map_location = f'cuda:{self.device}'
                else:
                    map_location = f'cuda:{torch.cuda.current_device()}'
            else:
                map_location = 'cpu'
            
            print(f"Loading checkpoint from {load_path} to device {map_location}")
            checkpoint = torch.load(load_path, map_location=map_location)

            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            raw_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("Model state loaded successfully")

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded successfully")

            self.num_updates = checkpoint.get('num_updates', 0)
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("Scaler state loaded successfully")

            if self.use_ema and 'ema_state_dict' in checkpoint:
                if self.ema is None:
                    try:
                        raw_model = self.model.module if hasattr(self.model, "module") else self.model
                        self.ema = ExponentialMovingAverageModel(
                            args=None,
                            model=raw_model,
                            decay=getattr(self.config, 'ema_decay', 0.999),
                            is_flattened=False
                        )
                        print("EMA initialized during checkpoint loading")
                    except Exception as e:
                        print(f"Warning: Failed to initialize EMA during loading: {e}")
                        self.use_ema = False
                        
                if self.ema is not None:
                    self.ema.load_state_dict(checkpoint['ema_state_dict'])
                    print("EMA state loaded successfully")
            elif self.use_ema:
                print("Warning: EMA enabled but no EMA state found in checkpoint")
            
            print(f"Resume training from epoch {checkpoint['epoch'] + 1}, total updates: {self.num_updates}")
            return checkpoint['epoch'], checkpoint.get('best_loss', float('inf'))
            
        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {load_path}")
            raise
        except KeyError as e:
            print(f"Error: Missing key in checkpoint: {e}")
            print("Checkpoint keys:", list(checkpoint.keys()) if 'checkpoint' in locals() else "Unknown")
            raise
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise

    @contextlib.contextmanager
    def _validation_model(self):
        """
        A context manager to temporarily swap the model with its EMA counterpart for validation.
        This is a critical practice from Uni-Core for robust evaluation.
        """
        if not (self.use_ema and getattr(self.config, 'validate_with_ema', False)):
            # If not using EMA for validation, just yield and do nothing.
            yield
            return

        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        original_state_dict = copy.deepcopy(raw_model.state_dict())

        ema_model = self.get_ema_model()
        raw_model.load_state_dict(ema_model.state_dict())
        print("Swapped model with EMA for validation.")

        try:
            yield
        finally:
            raw_model.load_state_dict(original_state_dict)
            print("Restored original model state after validation.")

    def train(self):
        config = self.config
        train_data = self.train_dataset
        shuffle_train = True if not self.config.dist else False

        if self.config.dist:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
            train_loader = DataLoader(train_data, shuffle=False, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                sampler=train_sampler,
                                collate_fn=self.data_collator)
        else:
            train_loader = DataLoader(train_data, shuffle=shuffle_train, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                collate_fn=self.data_collator)

        # --- Create validation loader once ---
        test_loader = None
        if self.test_dataset is not None:
            test_data = self.test_dataset
            if config.dist:
                test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=False)
                test_loader = DataLoader(test_data, shuffle=False, pin_memory=True,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers,
                                    sampler=test_sampler,
                                    collate_fn=self.data_collator)
            else:
                test_loader = DataLoader(test_data, shuffle=False, pin_memory=True,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers,
                                    collate_fn=self.data_collator)

        if self.config.dist and self.config.rank is not None:
            print(f"Setting process-specific seed for rank {self.config.rank}: {self.config.seed + self.config.rank}")
            torch.manual_seed(self.config.seed + self.config.rank)
            np.random.seed(self.config.seed + self.config.rank)
            torch.cuda.manual_seed(self.config.seed + self.config.rank)

        model = self.model
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        
        if config.amp_dtype.lower() == 'none':
            autocast_enabled = False
            autocast_dtype = torch.float32
        elif config.amp_dtype.lower() == 'bf16':
            autocast_enabled = True
            autocast_dtype = torch.bfloat16
        else:
            autocast_enabled = True
            autocast_dtype = torch.float16

        scaler = GradScaler(enabled=(autocast_dtype == torch.float16))
        if (not self.config.dist) or (self.config.dist and self.config.rank == 0):
            print(f"AMP autocast enabled={autocast_enabled}, dtype={autocast_dtype}; GradScaler enabled={scaler.is_enabled()}")

        total_steps = config.max_epochs * len(train_loader)
        
        # The new scheduler handles warmup_ratio internally.
        if self.lr_scheduler == 'CosineLRSchedule':
            scheduler = CosineLRSchedule(optimizer,
                                    total_steps=total_steps,
                                    max_lr=config.learning_rate,
                                    min_lr=config.min_lr,
                                    warmup_updates=config.warmup_updates,
                                    warmup_ratio=config.warmup_ratio,
                                    warmup_init_lr=1e-9) 
        elif self.lr_scheduler == 'WarmupConstantLRSchedule':
            scheduler = WarmupConstantLRSchedule(optimizer,
                                    total_steps=total_steps,
                                    max_lr=config.learning_rate,
                                    min_lr=config.min_lr,
                                    warmup_updates=config.warmup_updates,
                                    warmup_ratio=config.warmup_ratio,
                                    warmup_init_lr=1e-9) 
        print(f"Scheduler configured: total_steps={total_steps}, warmup_steps={scheduler.warmup_updates}")

        if config.load_checkpoint_path is not None:
            print(f'resuming training from {config.load_checkpoint_path}...')
            start_epoch, best_loss = self.load_checkpoint(config.load_checkpoint_path, optimizer, scaler)
            scheduler.step_update(self.num_updates)
        else:
            start_epoch = -1
            best_loss = float('inf')
            self.num_updates = 0
            
        def run_epoch(split, loader):
            is_train = split == 'train'
            model.train(is_train)

            losses = []
            token_losses = []
            position_losses = []
            # Use the provided loader

            is_main_process = not self.config.dist or self.config.rank == 0
            is_interactive = sys.stderr.isatty()
            pbar_enabled = is_main_process and is_interactive
            pbar = tqdm(enumerate(loader), total=len(loader), disable=not pbar_enabled) if is_train else enumerate(loader)
            
            for it, batch in pbar:

                if is_train:
                    optimizer.zero_grad()

                assert isinstance(batch, dict), "The data loader must use the custom collator from InertialARDataset."

                max_len_in_batch = batch.pop('max_len', None)

                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                model_inputs = batch
                model_inputs['max_len'] = max_len_in_batch

                with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=autocast_dtype):
                    with torch.set_grad_enabled(is_train):
                        logits, pred_pos, loss, loss_token, loss_position = model(**model_inputs)
                        
                        loss = loss.mean()
                        loss_token = loss_token.mean()
                        loss_position = loss_position.mean()

                        losses.append(loss.item())
                        token_losses.append(loss_token.item())
                        position_losses.append(loss_position.item())

                if is_train:
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        optimizer.step()
                    
                    self.num_updates += 1
                    lr = scheduler.step_update(self.num_updates)
                    
                    if self.use_ema and self.ema is not None:
                        self.ema.update(raw_model.named_parameters())

                    # Get scalar loss values for logging
                    loss_token_val = loss_token.item()
                    loss_position_val = loss_position.item()

                    if config.log_interval > 0 and self.num_updates % config.log_interval == 0:
                        if (not self.config.dist) or (self.config.dist and self.config.rank == 0):
                            print(f"step_train_loss: {loss.item():.5f} (token: {loss_token_val:.5f}, pos: {loss_position_val:.5f}) train_step: {self.num_updates}, learning_rate: {lr:e}, device: {self.device}")
                    # Modified progress bar description
                    if not self.config.dist or self.config.rank == 0:
                        pbar.set_description(f"epoch {epoch+1} iter {it}: loss {loss.item():.5f} (Token:{loss_token_val:.5f}, Pos:{loss_position_val:.5f}). lr {lr:e}")
            if self.config.dist:
                local_sums = torch.tensor([
                    sum(losses), len(losses),
                    sum(token_losses), len(token_losses),
                    sum(position_losses), len(position_losses)
                ], dtype=torch.float, device=self.device)
                dist.all_reduce(local_sums, op=dist.ReduceOp.SUM)
                loss = (local_sums[0] / local_sums[1]).item() if local_sums[1] > 0 else 0.0
                token_loss = (local_sums[2] / local_sums[3]).item() if local_sums[3] > 0 else 0.0
                position_loss = (local_sums[4] / local_sums[5]).item() if local_sums[5] > 0 else 0.0
            else:
                loss = float(np.mean(losses)) if losses else 0.0
                token_loss = float(np.mean(token_losses)) if token_losses else 0.0
                position_loss = float(np.mean(position_losses)) if position_losses else 0.0
                
            if is_train:
                return loss, token_loss, position_loss
            else:
                return {'loss': loss, 'loss_token': token_loss, 'loss_position': position_loss,}
        
        # --- Main Training Loop ---
        for epoch in range(start_epoch+1, config.max_epochs):
            if config.dist:
                train_loader.sampler.set_epoch(epoch)
                if test_loader is not None:
                    test_loader.sampler.set_epoch(epoch)
            start_time = time.time()
            train_loss, train_token_loss, train_pos_loss = run_epoch('train', train_loader)
            end_time = time.time()
            # Only log on the main process
            if (not self.config.dist) or (self.config.dist and self.config.rank == 0):
                print(f"epoch_train_loss: {train_loss:.5f} (Token: {train_token_loss:.5f}, Pos: {train_pos_loss:.5f}), epoch: {epoch + 1}, time: {end_time - start_time:.2f} seconds, device: {self.device}")
            # --- Validation Logic ---
            validation_metrics = None
            if test_loader is not None:
                # Step-based validation is handled inside the training loop
                # Epoch-based validation
                if config.validate_interval_updates <= 0 and (epoch + 1) % config.validate_interval == 0:
                    with self._validation_model():
                        validation_metrics = run_epoch('test', test_loader)
                        if (not self.config.dist) or (self.config.dist and self.config.rank == 0):
                            print(f"epoch test loss: {validation_metrics['loss']:.5f} (Token: {validation_metrics['loss_token']:.5f}, Pos: {validation_metrics['loss_position']:.5f}), epoch: {epoch + 1}")

            # --- Best Checkpoint Saving Logic ---
            if validation_metrics is not None:
                metric_to_check = validation_metrics.get(config.best_checkpoint_metric)
                if metric_to_check is not None and metric_to_check < best_loss:
                    best_loss = metric_to_check
                    print(f"New best validation metric ({config.best_checkpoint_metric}: {best_loss:.5f}) at epoch {epoch+1}, update {self.num_updates}. Saving model...")
                    ckpt_path = os.path.join(self.config.ckpt_path, 'checkpoint_best.pt')
                    if self.config.dist and self.device == 0:
                        self.save_checkpoint(epoch, model, best_loss, optimizer, scaler, ckpt_path)
                    elif not self.config.dist:
                        self.save_checkpoint(epoch, model, best_loss, optimizer, scaler, ckpt_path)


            # --- Regular Interval Checkpoint Saving ---
            if ((epoch+1) >= self.config.save_start_epoch and (epoch+1) % self.config.save_interval_epoch == 0) or epoch == config.max_epochs - 1:
                ckpt_path = os.path.join(self.config.ckpt_path, f'epoch_{epoch+1}.pt')
                if self.config.dist:
                    if self.device == 0:
                        print(f'Saving at latest epoch {epoch + 1}: {ckpt_path}')
                        self.save_checkpoint(epoch, model, best_loss, optimizer, scaler, ckpt_path)
                else:
                    print(f'Saving at latest epoch {epoch + 1}: {ckpt_path}')
                    self.save_checkpoint(epoch, model, best_loss, optimizer, scaler, ckpt_path)

        return None

    def get_ema_model(self):
        """Return the EMA model for validation when available."""
        if self.use_ema and self.ema is not None:
            if hasattr(self.ema, 'ema_model'):
                return self.ema.ema_model
            elif hasattr(self.ema, 'model_ema'):
                return self.ema.model_ema
            else:
                raise AttributeError('EMA object lacks ema_model/model_ema attribute')
        else:
            return self.model.module if hasattr(self.model, "module") else self.model
