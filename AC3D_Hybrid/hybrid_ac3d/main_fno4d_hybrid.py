import torch, numpy as np, random
import config
from networks import FNO4d
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
    # Model
    model = FNO4d(
        modes1=config.MODES, modes2=config.MODES, modes3=config.MODES, modes4_internal=None,
        width=config.WIDTH, width_q=config.WIDTH_Q, T_in_channels=config.T_IN_CHANNELS,
        n_layers=config.N_LAYERS
    ).to(device)
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
