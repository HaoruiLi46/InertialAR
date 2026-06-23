from copy import deepcopy
import torch


def separate_decay_params(args, params):
    if args.weight_decay <= 0:
        return [{"params": [p for _, p in params if p.requires_grad]}]

    no_wd = (
        set(args.no_weight_decay_names.split(","))
        if args.no_weight_decay_names
        else set()
    )

    def skip_decay(name, p):
        return name.endswith(".bias") or p.ndim == 1 or any(nd in name for nd in no_wd)

    decay_params = []
    no_decay_params = []
    for name, p in params:
        if not p.requires_grad:
            continue
        elif skip_decay(name, p):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    ret = []
    if len(decay_params) > 0:
        ret.append({"params": decay_params})
    if len(no_decay_params) > 0:
        ret.append({"params": no_decay_params, "weight_decay": 0.0})
    return ret


def pad_numel(numel, multiplier=2):
    return (numel + multiplier - 1) // multiplier * multiplier


def flatten_orders(params):
    dtype_grouped_params = {}
    ordered_dtype = []  # for sort dtype
    total_param_size = 0
    for p in params:
        if p.dtype not in dtype_grouped_params:
            dtype_grouped_params[p.dtype] = []
            ordered_dtype.append(p.dtype)
        dtype_grouped_params[p.dtype].append(p)
        total_param_size += pad_numel(p.data.numel())
    return dtype_grouped_params, ordered_dtype, total_param_size



@torch.no_grad()
def flatten_parameters(params):
    dtype_grouped_params, ordered_dtype, _ = flatten_orders(params)

    flatten_params = {}
    for dtype in ordered_dtype:
        cur_params = dtype_grouped_params[dtype]
        total_param_size = sum(pad_numel(p.data.numel()) for p in cur_params)
        flatten_params[dtype] = (
            cur_params[0].new(0).type(dtype).new_zeros(total_param_size)
        )
        offset = 0
        for p in cur_params:
            numel = p.data.numel()
            flatten_params[dtype][offset : offset + numel].copy_(p.data.view(-1))
            p.data = flatten_params[dtype].data[offset : offset + numel].view(*p.shape)
            offset += pad_numel(numel)
        flatten_params[dtype] = torch.nn.Parameter(flatten_params[dtype])
        flatten_params[dtype].grad = flatten_params[dtype].data.new(total_param_size)
        offset = 0
        for p in cur_params:
            numel = p.data.numel()
            p.grad = flatten_params[dtype].grad[offset : offset + numel].view(*p.shape)
            offset += pad_numel(numel)
    torch.cuda.empty_cache()
    return [flatten_params[dtype] for dtype in ordered_dtype]


@torch.no_grad()
def flatten_parameters_fp32(params, set_to_param=False, set_grad=True):
    dtype_grouped_params, ordered_dtype, total_param_size = flatten_orders(params)

    flatten_params = torch.zeros(
        total_param_size, dtype=torch.float32, device=params[0].device
    )
    offset = 0
    for dtype in ordered_dtype:
        cur_params = dtype_grouped_params[dtype]
        for p in cur_params:
            numel = p.data.numel()
            flatten_params[offset : offset + numel].copy_(p.data.view(-1))
            if set_to_param:
                p.data = flatten_params.data[offset : offset + numel].view(*p.shape)
                # set to None here, it will throw error when using this incorrectly
                p.grad = None
            offset += pad_numel(numel)
    flatten_params = torch.nn.Parameter(flatten_params)
    if set_grad:
        flatten_params.grad = torch.zeros_like(flatten_params)
    torch.cuda.empty_cache()
    return flatten_params


def get_fp16_params(args, params):
    param_group = separate_decay_params(args, params)
    fp16_group = []
    fp32_group = []
    for param_dict in param_group:
        params = param_dict["params"]
        check_param_device(params)
        fp16_params = flatten_parameters(params)
        fp32_params = flatten_parameters_fp32(params)
        fp16_group.append({"params": fp16_params})
        param_dict["params"] = [fp32_params]
        fp32_group.append(param_dict)
    return fp16_group, fp32_group


class ExponentialMovingAverageModel:
    def __init__(self, args, model, decay, is_flattened=False):
        self.args = args
        self.model_ema = deepcopy(model)
        self.decay = decay
        self.is_flattened = is_flattened
        if not is_flattened:
            self.name2param = self.get_name2param()
        else:
            self.flatten_params = self.flatten_parameters()

    def get_name2param(self):
        name2param = dict()
        for n, p in self.model_ema.named_parameters():
            name2param[n] = p
            # use float type for ema
            p.data = p.data.float()
            p.grad = None
        return name2param

    def flatten_parameters(self):
        param_group = separate_decay_params(
            self.args, self.model_ema.named_parameters()
        )
        flatten_group = []
        for param_dict in param_group:
            params = param_dict["params"]
            flatten_param = flatten_parameters_fp32(
                params, set_to_param=True, set_grad=False
            )
            flatten_group.append(flatten_param)
        return flatten_group

    def update_one_param(self, ema_param, new_param):
        diff = ema_param - new_param
        diff *= 1 - self.decay
        ema_param -= diff

    def update(self, new_param):
        if self.is_flattened:
            with torch.no_grad():
                for i in range(len(self.flatten_params)):
                    self.update_one_param(
                        self.flatten_params[i], new_param[i]["params"][0]
                    )
        else:
            with torch.no_grad():
                for n, p in new_param:
                    if n in self.name2param:
                        self.update_one_param(self.name2param[n], p)

    def load_state_dict(self, state_dict):
        self.model_ema.load_state_dict(state_dict["params"])
        self.decay = state_dict["decay"] if "decay" in state_dict else self.decay

    def state_dict(self):
        return {
            "params": self.model_ema.state_dict(),
            "decay": self.decay,
        }
