# dataset.py
import torch
from torch.utils.data import Dataset
from utilities import (
    make_uniform_grid,
    sample_periodic_gp_ic,
    simulate_heat_1d,
    downsample_linear_time,
)
from config import Heat1DConfig

class Heat1DGenerated(Dataset):
    """
    Generates trajectories (coarse) and PI profiles (Sec. 2.2.2).
    """
    def __init__(self, cfg: Heat1DConfig, split: str = "train"):
        super().__init__()
        assert split in ("train", "test", "val")
        self.cfg = cfg
        self.split = split
        self.device = torch.device("cpu")  # keep raw data on CPU
        self.dtype = cfg.dtype
        self._prepare_grid()
        self._generate()

    def _prepare_grid(self):
        self.x = make_uniform_grid(self.cfg.x0, self.cfg.x1, self.cfg.Nx,
                                   device=self.device, dtype=self.dtype)
        self.dx = (self.x[-1] - self.x[0]) / (self.cfg.Nx - 1)

    def _generate(self):
        cfg = self.cfg
        if self.split == "train":
            N = cfg.N_train
            T = cfg.T_train
            Nt_fine = cfg.Nt_fine_train
        else:
            N = cfg.N_test
            T = cfg.T_test if self.split == "test" else cfg.T_train
            Nt_fine = cfg.Nt_fine_test if self.split == "test" else cfg.Nt_fine_train

        dt_fine = T / Nt_fine
        self.trajectories = []
        gseed = cfg.gp_seed + (0 if self.split == "train" else (1 if self.split == "val" else 2))
        gen = torch.Generator(device=self.device).manual_seed(gseed)

        for _ in range(N):
            f0 = sample_periodic_gp_ic(
                self.x, cfg.gp_length_scale, cfg.gp_variance,
                seed=int(torch.randint(0, 10_000_000, (1,), generator=gen).item()),
                device=self.device, dtype=self.dtype,
            )
            U_fine, Ut_fine = simulate_heat_1d(f0, cfg.alpha, self.x, Nt_fine, dt_fine)
            Uc, Utc = downsample_linear_time(U_fine, Ut_fine, T, cfg.Nt_coarse)
            self.trajectories.append({"U": Uc, "Ut": Utc, "T": T})

        # PI profiles from first 1600 trajectories at t in {0, 0.25, 0.5}
        self.pi_profiles = []
        if self.split == "train":
            use_count = min(cfg.pi_ic_count, len(self.trajectories))
            target_times = torch.tensor(cfg.pi_times, dtype=getattr(torch, self.dtype))
            for i in range(use_count):
                U = self.trajectories[i]["U"]
                T = self.trajectories[i]["T"]
                t_coarse = torch.linspace(0.0, T, U.shape[0], device=U.device, dtype=U.dtype)
                for t in target_times:
                    idx = torch.argmin(torch.abs(t_coarse - t))
                    self.pi_profiles.append({"u": U[idx], "t": float(t), "idx": int(idx), "traj_id": i})

    def __len__(self):
        if self.split == "train" and len(self.pi_profiles) > 0:
            return len(self.pi_profiles)
        return len(self.trajectories)

    def __getitem__(self, idx):
        if self.split == "train" and len(self.pi_profiles) > 0:
            item = self.pi_profiles[idx]
            return {"mode": "pi_profile", "u": item["u"], "t": item["t"], "traj_id": item["traj_id"]}
        traj = self.trajectories[idx]
        return {"mode": "trajectory", "U": traj["U"], "Ut": traj["Ut"], "T": traj["T"]}
