# config.py
from dataclasses import dataclass

@dataclass
class Heat1DConfig:
    # Numerical dtype
    dtype: str = "float16"   # ensure robust GP sampling & AD

    # PDE: u_t = alpha * u_xx
    alpha: float = 0.01

    # Grids
    x0: float = 0.0
    x1: float = 1.0
    Nx: int = 32 # 128

    # Fine simulation (dataset generation)
    T_train: float = 1.0
    T_test: float = 5.0
    Nt_fine_train: int = 10_000
    Nt_fine_test: int = 50_000
    Nt_coarse: int = 100  # intervals (Nt_coarse+1 samples)

    # Dataset sizes
    N_train: int = 1000 #2000
    N_test: int = 200 #500

    # PI sampling per Sec. 2.2.2
    pi_ic_count: int = 500 # 1600
    pi_times: tuple = (0.0, 0.25, 0.5)  # 4800 profiles

    # Gaussian process ICs (periodic)
    gp_length_scale: float = 0.1
    gp_variance: float = 10_000.0
    gp_seed: int = 1234

    # DeepONet (Table 2: PITI)
    branch_layers: int = 4 # 8
    trunk_layers: int = 5 # 10
    hidden_dim: int = 16 # 256
    activation: str = "tanh"
    split_branch: bool = False
    split_trunk: bool = True

    # Training (pure physics)
    epochs: int = 300 # 300_000
    batch_size_profiles: int = 16
    Nr_per_profile: int = 128
    Nb_per_profile: int = 4
    Ns_per_profile: int = 128
    Nc_per_profile: int = 128
    use_data_losses: bool = False

    # Loss weights (Table 3: PITI)
    lambda_PDE: float = 1.0
    lambda_R: float = 10.0
    lambda_BC: float = 1.0
    lambda_C: float = 1.0
    lambda_u: float = 0.0
    lambda_ut: float = 0.0

    # Optimizer (Table 2)
    base_lr: float = 1e-4
    lr_decay_gamma: float = 0.9
    lr_decay_steps: int = 40_000
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0

    # Inference
    dt_infer: float = 0.01
    infer_scheme: str = "RK4"   # "Euler", "RK4", "ABM2"

    # Paths / device
    save_checkpoints: bool = True
    ckpt_path: str = "checkpoints/heat1d_piti.pt"
    device: str = "cuda"  # "cuda" or "cpu"
