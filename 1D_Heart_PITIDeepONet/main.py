# main.py
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from config import Heat1DConfig
from dataset import Heat1DGenerated
from trainer import PITITrainer
from utilities import make_uniform_grid

def collate_pi(batch):
    out = {"mode": [], "u": [], "t": [], "traj_id": []}
    for b in batch:
        out["mode"].append(b["mode"])
        out["u"].append(b["u"])
        out["t"].append(b["t"])
        out["traj_id"].append(b["traj_id"])
    return out

def plot_xt_contour(x, t, U, fname, title):
    """
    x: (Nx,), t: (Nt+1,), U: (Nt+1, Nx)
    Saves contour image.
    """
    X, T = torch.meshgrid(x, t, indexing="ij")  # (Nx,Nt+1)
    Z = U.T  # (Nt+1, Nx) -> transpose to (Nx,Nt+1) for pcolormesh/contourf
    plt.figure(figsize=(7, 4))
    cs = plt.contourf(T.cpu().numpy(), X.cpu().numpy(), Z.cpu().numpy(), levels=100)
    plt.colorbar(cs, label="T")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

def main():
    cfg = Heat1DConfig()
    # Optional (global) default dtype:
    torch.set_default_dtype(getattr(torch, cfg.dtype))

    # Datasets
    train_ds = Heat1DGenerated(cfg, split="train")
    val_ds = Heat1DGenerated(cfg, split="val")
    test_ds = Heat1DGenerated(cfg, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size_profiles,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_pi,
        num_workers=0,
    )

    trainer = PITITrainer(cfg)
    x_grid = make_uniform_grid(cfg.x0, cfg.x1, cfg.Nx, device=trainer.device, dtype=cfg.dtype)

    print("Starting training (pure physics-informed)...")
    for epoch in range(1, cfg.epochs + 1):
        loss = trainer.train_epoch(train_loader, x_grid)
        if epoch % 1000 == 0:
            print(f"[Epoch {epoch}/{cfg.epochs}] loss={loss:.6e} lr={trainer.optimizer.param_groups[0]['lr']:.3e}")
        if cfg.save_checkpoints and epoch % 50_000 == 0:
            os.makedirs(os.path.dirname(cfg.ckpt_path), exist_ok=True)
            torch.save({"model": trainer.model.state_dict(), "cfg": cfg.__dict__}, cfg.ckpt_path)

    if cfg.save_checkpoints:
        os.makedirs(os.path.dirname(cfg.ckpt_path), exist_ok=True)
        torch.save({"model": trainer.model.state_dict(), "cfg": cfg.__dict__}, cfg.ckpt_path)

    # ---- Inference evaluation at T_test and x–t contour plotting ----
    print(f"Rolling learned operator with {cfg.infer_scheme} (dt={cfg.dt_infer}) to T={cfg.T_test}...")
    # Pick first test trajectory
    item = test_ds[0]
    U_ref = item["U"].to(trainer.device).to(getattr(torch, cfg.dtype))   # (Nt_c+1, Nx)
    T_ref = item["T"]

    u0 = U_ref[0].clone()
    t_pred, U_pred = trainer.roll_and_collect(u0, cfg.T_test, cfg.dt_infer, scheme=cfg.infer_scheme)

    # Reference resampled to t_pred via linear interp on coarse U_ref
    Nt_c = U_ref.shape[0] - 1
    t_coarse = torch.linspace(0.0, T_ref, Nt_c + 1, device=trainer.device, dtype=U_ref.dtype)
    # Interp reference to t_pred
    def interp_time(A, t_src, t_tgt):
        At = torch.zeros(len(t_tgt), A.shape[1], device=A.device, dtype=A.dtype)
        idx_float = (t_tgt / t_src[-1]) * (len(t_src) - 1)
        idx0 = torch.clamp(idx_float.floor().long(), 0, len(t_src) - 2)
        idx1 = idx0 + 1
        w = (t_tgt - t_src[idx0]) / (t_src[idx1] - t_src[idx0] + 1e-20)
        w = w.unsqueeze(1)
        At = (1 - w) * A[idx0] + w * A[idx1]
        return At

    U_ref_interp = interp_time(U_ref, t_coarse, t_pred)

    # Save x–t contours
    plot_xt_contour(x_grid, t_pred, U_pred, "heat1d_xt_contour.png", "PITI-DeepONet Prediction (T)")
    plot_xt_contour(x_grid, t_pred, U_ref_interp, "heat1d_xt_contour_ref.png", "Reference (T)")
    plot_xt_contour(x_grid, t_pred, torch.abs(U_pred - U_ref_interp), "heat1d_xt_contour_diff.png", "|Pred - Ref|")

    print("Saved contour plots: heat1d_xt_contour.png, heat1d_xt_contour_ref.png, heat1d_xt_contour_diff.png")

if __name__ == "__main__":
    main()
