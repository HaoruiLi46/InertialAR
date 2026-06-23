"""Configuration for the InertialAR molecular autoregressive model."""

import torch

PUBLIC_MODEL_NAME = "InertialAR"
LEGACY_MODEL_NAMES = ("InertialGPT",)
SUPPORTED_MODEL_NAMES = (PUBLIC_MODEL_NAME, *LEGACY_MODEL_NAMES)


def normalize_model_name(model_3d):
    if model_3d is None:
        return PUBLIC_MODEL_NAME
    if model_3d in SUPPORTED_MODEL_NAMES:
        return PUBLIC_MODEL_NAME
    raise ValueError(f"Invalid model_3d: {model_3d}")


def pairwise_squared_distances(x, y):
    x_norm = x.pow(2).sum(dim=1, keepdim=True)  # [N, 1]
    y_norm = y.pow(2).sum(dim=1, keepdim=True).T  # [1, M]
    dist = x_norm + y_norm - 2 * x @ y.T
    return dist

def nystrom_prepare_landmark_multi_sigma(Z, sigma_list, EPS):
    """
    Input:
        Z: (m, d)
        sigma_list: (L, 1, 1)

    Output:
        L_inv_list: inverse Cholesky factors L^{-1} for W = L @ L.T.
        Forward code uses C @ L_inv.T, yielding standard Nyström features C @ L^{-T}.
    """
    device = Z.device

    D_W = pairwise_squared_distances(Z, Z).unsqueeze(0)  # (1, m, m)
    W_all = torch.exp(-D_W / (2 * sigma_list**2))  # (L, m, m)

    L_inv_list = []
    for l in range(len(sigma_list)):
        W = W_all[l] + EPS * torch.eye(Z.shape[0], device=device, dtype=Z.dtype)
        L = torch.linalg.cholesky(W)  # (m, m)
        eye = torch.eye(Z.shape[0], device=device, dtype=Z.dtype)
        L_inv = torch.linalg.solve_triangular(L, eye, upper=False)
        L_inv_list.append(L_inv)
    L_inv_list = torch.stack(L_inv_list).to(device)
    return L_inv_list

class InertialARConfig:

    embd_pdrop = 0.0
    resid_pdrop = 0.0
    attn_pdrop = 0.0
    S_churn = 0
    S_min = 0.05
    S_max = 50
    S_noise = 1
    num_steps = 100
    EPS = 1e-5
    
    def __init__(self, vocab_size, block_size, latent_dim, max_distance, num_sigma, dropout, scale, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.max_distance = max_distance
        self.num_sigma = num_sigma
        self.scale = scale
        
        self.embd_pdrop = dropout
        self.resid_pdrop = dropout
        self.attn_pdrop = dropout
        self.diffusion_dropout = dropout

        self.num_tasks = 1

        for k,v in kwargs.items():
            setattr(self, k, v)

        self.model_3d = normalize_model_name(getattr(self, "model_3d", None))

        assert latent_dim % self.n_head == 0, "latent_dim must be divisible by n_head for RoPE"

        self.rope_dim = latent_dim
        self.nystrom_dim = latent_dim
        self.final_repr_dim = self.rope_dim + self.nystrom_dim

        self.num_anchor = latent_dim
        self.sigma_list = torch.linspace(0.1, 1, self.num_sigma).view(-1, 1, 1)
        print('sigma_list', self.sigma_list.shape, self.sigma_list.squeeze())

        self.anchor_points = (torch.rand(self.num_anchor, 3) - 0.5) * 2 * self.max_distance
        self.L_inv_list = nystrom_prepare_landmark_multi_sigma(
            self.anchor_points,
            self.sigma_list,
            self.EPS,
        )
        self.idx_repr_dim = self.rope_dim - 6

        assert self.final_repr_dim % self.n_head == 0
