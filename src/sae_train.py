"""
Training loop cho Sparse Autoencoder.

Theo đề cương §6.3:
  - Optimizer Adam, lr=1e-3, betas=(0.9, 0.999)
  - Batch size 256
  - Epoch tối đa 200, EarlyStopping(patience=10) theo val MSE
  - Seed {0, 1, 42, 123, 2026} cho 5 lần chạy độc lập
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sae_dataset import make_dataloaders
from sae_losses import sae_loss
from sae_model import SparseAutoencoder


@dataclass
class TrainConfig:
    # Data
    train_path: str = "/home/kali/sae-lof-ids-pi5/data/processed/train_benign.parquet"
    val_path: str = "/home/kali/sae-lof-ids-pi5/data/processed/val.parquet"
    benign_label: int = 0
    max_train_samples: int | None = None   # None = dùng full 878,556 benign
    max_val_samples: int | None = None
    # Model
    input_dim: int = 46
    hidden_dims: tuple[int, ...] = (32, 16)
    latent_dim: int = 8
    # Loss
    rho: float = 0.05
    beta: float = 3.0
    # Optim
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 10
    # Infra
    num_workers: int = 2
    device: str = "auto"  # auto -> cuda/cpu
    seed: int = 0
    # Logging
    run_name: str = "sae_v0"
    ckpt_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    verbose_every: int = 1  # print mỗi N epoch


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic trade-off: bật nếu cần reproducibility tuyệt đối
    # torch.use_deterministic_algorithms(True)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    rho: float,
    beta: float,
    device: str,
) -> dict[str, float]:
    """Evaluate trên val loader. Trả về MSE trung bình (và các chỉ số phụ)."""
    model.eval()
    total_mse = 0.0
    total_kl = 0.0
    n_batches = 0
    rho_hat_accum = None

    for x in loader:
        x = x.to(device, non_blocking=True)
        x_hat, z = model(x)
        _, comp = sae_loss(x, x_hat, z, rho=rho, beta=beta)
        total_mse += comp["mse"]
        total_kl += comp["kl"]
        n_batches += 1
        rh = z.mean(dim=0).cpu().numpy()
        rho_hat_accum = rh if rho_hat_accum is None else rho_hat_accum + rh

    rho_hat_mean = (rho_hat_accum / n_batches) if n_batches > 0 else np.zeros(1)

    return {
        "mse": total_mse / max(n_batches, 1),
        "kl": total_kl / max(n_batches, 1),
        "rho_hat_mean_overall": float(rho_hat_mean.mean()),
        "rho_hat_std_overall": float(rho_hat_mean.std()),
        "n_batches": n_batches,
    }


def train(cfg: TrainConfig) -> dict:
    """Main training function. Trả về dict với history và best metrics."""

    # Resolve device
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(cfg.seed)

    # Dirs
    ckpt_dir = Path(cfg.ckpt_dir)
    log_dir = Path(cfg.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{cfg.run_name}_best.pt"
    log_path = log_dir / f"{cfg.run_name}_history.json"

    # Data
    print(f"[{cfg.run_name}] Loading data...")
    train_loader, val_loader, n_feat = make_dataloaders(
        train_path=cfg.train_path,
        val_path=cfg.val_path,
        batch_size=cfg.batch_size,
        max_train_samples=cfg.max_train_samples,
        max_val_samples=cfg.max_val_samples,
        benign_label=cfg.benign_label,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    # Auto-adjust input_dim nếu lệch (vd EDA giữ 46)
    if n_feat != cfg.input_dim:
        print(f"[{cfg.run_name}] input_dim override: cfg={cfg.input_dim} -> data={n_feat}")
        cfg.input_dim = n_feat

    # Model
    model = SparseAutoencoder(
        input_dim=cfg.input_dim,
        hidden_dims=cfg.hidden_dims,
        latent_dim=cfg.latent_dim,
    ).to(cfg.device)
    print(f"[{cfg.run_name}] Model params: {model.count_parameters():,}")

    optim = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
    )

    # Training state
    history: list[dict] = []
    best_val_mse = float("inf")
    best_epoch = -1
    no_improve = 0
    t_start = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        t_ep = time.time()

        train_mse_sum = 0.0
        train_kl_sum = 0.0
        n_seen = 0

        for x in train_loader:
            x = x.to(cfg.device, non_blocking=True)
            x_hat, z = model(x)
            loss, comp = sae_loss(x, x_hat, z, rho=cfg.rho, beta=cfg.beta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            bs = x.size(0)
            train_mse_sum += comp["mse"] * bs
            train_kl_sum += comp["kl"] * bs
            n_seen += bs

        train_mse = train_mse_sum / max(n_seen, 1)
        train_kl = train_kl_sum / max(n_seen, 1)

        val_metrics = evaluate(model, val_loader, cfg.rho, cfg.beta, cfg.device)
        val_mse = val_metrics["mse"]

        epoch_time = time.time() - t_ep

        entry = {
            "epoch": epoch,
            "train_mse": train_mse,
            "train_kl": train_kl,
            "val_mse": val_mse,
            "val_kl": val_metrics["kl"],
            "val_rho_hat_mean": val_metrics["rho_hat_mean_overall"],
            "val_rho_hat_std": val_metrics["rho_hat_std_overall"],
            "epoch_time_s": epoch_time,
        }
        history.append(entry)

        if epoch % cfg.verbose_every == 0 or epoch == 1:
            print(
                f"[{cfg.run_name}] ep{epoch:03d} "
                f"tr_mse={train_mse:.5f} tr_kl={train_kl:.4f} "
                f"val_mse={val_mse:.5f} "
                f"rho_hat={val_metrics['rho_hat_mean_overall']:.3f}"
                f"±{val_metrics['rho_hat_std_overall']:.3f} "
                f"({epoch_time:.1f}s)"
            )

        # Early stopping check
        if val_mse < best_val_mse - 1e-6:
            best_val_mse = val_mse
            best_epoch = epoch
            no_improve = 0
            # Save best
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "cfg": asdict(cfg),
                "val_mse": val_mse,
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"[{cfg.run_name}] EarlyStopping at epoch {epoch} "
                      f"(no improvement in {cfg.patience} epochs, best ep{best_epoch})")
                break

    total_time = time.time() - t_start

    summary = {
        "cfg": asdict(cfg),
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "total_epochs_run": len(history),
        "total_time_s": total_time,
        "history": history,
        "model_params": model.count_parameters(),
    }

    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[{cfg.run_name}] DONE. Best val_mse={best_val_mse:.5f} "
          f"@ ep{best_epoch} | total {total_time:.1f}s")
    return summary


if __name__ == "__main__":
    # Smoke test: chạy 2 epoch trên dummy data
    import tempfile

    # Tạo dummy parquet
    rng = np.random.default_rng(0)
    n_train, n_val = 2000, 500
    cols = [f"feat_{i}" for i in range(46)] + ["Label"]

    import pandas as pd
    df_train = pd.DataFrame(
        np.column_stack([rng.standard_normal((n_train, 46)).astype(np.float32),
                         np.zeros(n_train, dtype=int)]),
        columns=cols,
    )
    df_val = pd.DataFrame(
        np.column_stack([rng.standard_normal((n_val, 46)).astype(np.float32),
                         rng.integers(0, 8, size=n_val)]),
        columns=cols,
    )

    tmpdir = tempfile.mkdtemp()
    tp = Path(tmpdir) / "tr.parquet"
    vp = Path(tmpdir) / "va.parquet"
    df_train.to_parquet(tp)
    df_val.to_parquet(vp)

    cfg = TrainConfig(
        train_path=str(tp),
        val_path=str(vp),
        max_epochs=3,
        patience=5,
        batch_size=64,
        num_workers=0,
        run_name="smoke_test",
        ckpt_dir=tmpdir + "/ckpt",
        log_dir=tmpdir + "/log",
    )
    summary = train(cfg)
    assert summary["total_epochs_run"] == 3
    assert summary["best_val_mse"] < float("inf")
    print("✓ Training smoke test OK")
