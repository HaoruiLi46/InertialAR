import math

class CosineLRSchedule:
    """
    A learning rate scheduler that mimics Uni-Core's Cosine scheduler.
    It provides a linear warmup phase followed by a cosine decay phase,
    with support for restarts and learning rate shrinking.
    This implementation is strictly aligned with the Uni-Core version.
    """
    def __init__(self, optimizer, total_steps, max_lr, min_lr, 
                 warmup_updates=0, warmup_ratio=0.0, warmup_init_lr=-1.0, 
                 t_mult=1.0, lr_period_updates=-1, lr_shrink=0.1):
        """
        Args:
            optimizer: The optimizer instance.
            total_steps (int): The total number of training steps.
            max_lr (float): The maximum learning rate after warmup.
            min_lr (float): The minimum learning rate to decay to.
            warmup_updates (int): Number of steps for the warmup phase. Overrides warmup_ratio if > 0.
            warmup_ratio (float): Fraction of total_steps to use for warmup.
            warmup_init_lr (float): Initial learning rate for warmup. Defaults to min_lr if < 0.
            t_mult (float): Factor to grow the length of each period.
            lr_period_updates (int): Initial number of updates per period. Defaults to total_steps - warmup_updates.
            lr_shrink (float): Shrink factor for annealing.
        """
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        # Determine warmup updates
        if warmup_ratio > 0 and warmup_updates == 0:
            self.warmup_updates = int(warmup_ratio * total_steps)
        else:
            self.warmup_updates = warmup_updates

        # Determine warmup initial LR
        self.warmup_init_lr = warmup_init_lr if warmup_init_lr >= 0 else self.min_lr
        
        if self.warmup_updates > 0:
            # linearly warmup for the first self.warmup_updates
            self.lr_step = (self.max_lr - self.warmup_init_lr) / self.warmup_updates
        else:
            self.lr_step = 1 # Match Uni-core's implementation

        self.t_mult = t_mult
        self.lr_shrink = lr_shrink
        
        if lr_period_updates <= 0:
            self.period = self.total_steps - self.warmup_updates
        else:
            self.period = lr_period_updates

        # Set initial learning rate
        self.lr = self.warmup_init_lr
        self._set_lr(self.lr)

    def _set_lr(self, lr):
        """Sets the learning rate for all parameter groups in the optimizer."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def step(self, epoch, val_loss=None):
        """
        Update the learning rate at the end of the given epoch.
        Following Uni-Core, this is a no-op as we update on a per-step basis.
        """
        # we don't change the learning rate at epoch boundaries
        return self._get_lr()

    def step_update(self, num_updates):
        """
        Update the learning rate after each update.
        Args:
            num_updates (int): The current training step.
        Returns:
            float: The new learning rate.
        """
        if num_updates < self.warmup_updates:
            # Linear warmup
            self.lr = self.warmup_init_lr + num_updates * self.lr_step
        else:
            # Cosine decay phase
            curr_updates = num_updates - self.warmup_updates
            
            if self.t_mult != 1:
                i = math.floor(
                    math.log(1 - curr_updates / self.period * (1 - self.t_mult), self.t_mult)
                )
                t_i = self.t_mult**i * self.period
                t_curr = curr_updates - (1 - self.t_mult**i) / (1 - self.t_mult) * self.period
                r = float(t_curr) / t_i
            else:
                # force i to zero in one-cycle
                i = 0
                t_i = self.period
                t_curr = curr_updates
                r = float(t_curr) / t_i
                r = min(1.0, r)

            lr_shrink = self.lr_shrink**i
            min_lr = self.min_lr * lr_shrink
            max_lr = self.max_lr * lr_shrink

            self.lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * r))

        self._set_lr(self.lr)
        return self.lr


class WarmupConstantLRSchedule:
    """
    A learning rate scheduler with linear warmup followed by constant learning rate.
    This is a simplified version of CosineLRSchedule for constant LR after warmup.
    """
    def __init__(self, optimizer, total_steps, max_lr, min_lr=1e-9,
                 warmup_updates=0, warmup_ratio=0.0, warmup_init_lr=-1.0):
        """
        Args:
            optimizer: The optimizer instance.
            total_steps (int): The total number of training steps.
            max_lr (float): The maximum learning rate after warmup (constant LR).
            min_lr (float): The minimum learning rate (not used in constant phase, but kept for compatibility).
            warmup_updates (int): Number of steps for the warmup phase. Overrides warmup_ratio if > 0.
            warmup_ratio (float): Fraction of total_steps to use for warmup.
            warmup_init_lr (float): Initial learning rate for warmup. Defaults to min_lr if < 0.
        """
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr

        # Determine warmup updates
        if warmup_ratio > 0 and warmup_updates == 0:
            self.warmup_updates = int(warmup_ratio * total_steps)
        else:
            self.warmup_updates = warmup_updates

        # Determine warmup initial LR
        self.warmup_init_lr = warmup_init_lr if warmup_init_lr >= 0 else self.min_lr

        # Calculate warmup step size
        if self.warmup_updates > 0:
            self.lr_step = (self.max_lr - self.warmup_init_lr) / self.warmup_updates
        else:
            self.lr_step = 0  # No warmup

        # Set initial learning rate
        self.lr = self.warmup_init_lr
        self._set_lr(self.lr)

    def _set_lr(self, lr):
        """Sets the learning rate for all parameter groups in the optimizer."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def step(self, epoch, val_loss=None):
        """
        Update the learning rate at the end of the given epoch.
        Following Uni-Core, this is a no-op as we update on a per-step basis.
        """
        # we don't change the learning rate at epoch boundaries
        return self._get_lr()

    def step_update(self, num_updates):
        """
        Update the learning rate after each update.
        Args:
            num_updates (int): The current training step.
        Returns:
            float: The new learning rate.
        """
        if num_updates < self.warmup_updates:
            # Linear warmup phase
            self.lr = self.warmup_init_lr + num_updates * self.lr_step
        else:
            # Constant learning rate phase
            self.lr = self.max_lr

        self._set_lr(self.lr)
        return self.lr