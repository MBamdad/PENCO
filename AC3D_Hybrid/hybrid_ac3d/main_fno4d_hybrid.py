import os  # <- minimal addition
import torch, numpy as np, random
import config
from networks import FNO4d, TNO3d
from utilities import build_loaders, train_fno_hybrid, evaluate_stats_and_plot
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    set_seeds(config.SEED)
    device = config.DEVICE
    print("Using device:", device)
    print('PDE_WEIGHT is: ', config.PDE_WEIGHT)
    print('MODEL name is: ', config.MODEL)
    print('Problem is: ', config.PROBLEM)
    print('N_Train is: ', config.N_TRAIN)

    # (tiny clarity prints; no behavioral change)
    print(f"Grid: N={config.GRID_RESOLUTION}, L={config.L_DOMAIN:g}, dx={config.DX:g}")
    print(f"Time: dt={config.DT:g}, Nt={config.TOTAL_TIME_STEPS}, T_end={config.DT*config.TOTAL_TIME_STEPS:g}")
    print(f"ε (epsilon): {config.EPSILON_PARAM:g}")
    print(f"Data path: {config.MAT_DATA_PATH}")
    if isinstance(config.MAT_DATA_PATH, str) and (config.MAT_DATA_PATH.startswith("/") or config.MAT_DATA_PATH.startswith(".")):
        if not os.path.exists(config.MAT_DATA_PATH):
            print(f"[WARN] MAT_DATA_PATH not found: {config.MAT_DATA_PATH}")

    # Model
    if config.MODEL == 'FNO4d':
        model = FNO4d(
            modes1=config.MODES, modes2=config.MODES, modes3=config.MODES, modes4_internal=None,
            width=config.WIDTH, width_q=config.WIDTH_Q, T_in_channels=config.T_IN_CHANNELS,
            n_layers=config.N_LAYERS
        ).to(device)
    else:
        model = TNO3d(
            modes1=config.MODES,  # spectral modes in x
            modes2=config.MODES,  # spectral modes in y
            modes3=config.MODES,  # spectral modes in z
            width=config.WIDTH,   # channel width in trunk
            width_q=config.WIDTH_Q,  # width in the projection MLP (q)
            width_h=config.WIDTH_H,  # temporal memory width
            T_in=config.T_IN_CHANNELS,
            T_out=1,  # one-step output, keeps utilities unchanged
            n_layers=config.N_LAYERS
        ).to(config.DEVICE)

    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Data
    train_loader, test_loader, test_ids, normalizers = build_loaders()

    # Optim
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # Train (hybrid, with PINNs-style debug prints)
    train_fno_hybrid(model, train_loader, test_loader, optimizer, scheduler, device, pde_weight=config.PDE_WEIGHT)

    # Evaluate: stats + 3×len(times) plot
    evaluate_stats_and_plot(model, config.MAT_DATA_PATH, test_ids, times=config.EVAL_TIME_FRAMES)

if __name__ == "__main__":
    main()
