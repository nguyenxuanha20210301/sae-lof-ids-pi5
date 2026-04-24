"""
Training loop Week 4 — upgrade của src/sae_train.py với:
  1. Resume-from-checkpoint giữa epoch (RNG state save/load đúng)
  2. Done-flag pattern: skip run đã hoàn thành
  3. Checkpoint mỗi epoch (không chỉ khi best) để recover sau interrupt
  4. Atomic write: dùng write-then-rename để tránh corrupted checkpoint nếu Colab kill giữa save

Sử dụng:
    from sae_train_v1 import TrainConfigV1, train_with_resume

    cfg = TrainConfigV1(
        train_path='...', val_path='...', benign_label=0,
        rho=0.05, beta=3.0, seed=42,
        run_name='sae_v1_rho=0.05_beta=3.0_seed=42',
        ckpt_dir='/content/drive/MyDrive/sae-lof-ids-pi5/checkpoints',
        log_dir='/content/drive/MyDrive/sae-lof-ids-pi5/logs',
        ...
    )
    summary = train_with_resume(cfg)
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import từ Week 3 codebase (giả định src/ có các module này)
from sae_model import SparseAutoencoder
from sae_losses import sae_loss
from sae_dataset import BenignFlowDataset, make_dataloaders


@dataclass
class TrainConfigV1:
    """Config cho training v1 với resume support."""
    # Data
    train_path: str
    val_path: str
    benign_label: int = 0
    max_train_samples: Optional[int] = None  # None = full
    max_val_samples: Optional[int] = None

    # Model
    input_dim: int = 46
    hidden_dims: tuple = (32, 16)
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
    num_workers: int = 4
    device: str = "auto"  # 'auto' | 'cuda' | 'cpu'
    seed: int = 0

    # Run identifiers
    run_name: str = "sae_v1"
    ckpt_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    verbose_every: int = 5


# ---------------------------------------------------------------------------
# Utilities: deterministic setup, atomic write, RNG state
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set seed cho tất cả RNG (torch, numpy, python). Gọi MỘT LẦN ở đầu training.

    Note: để resume đúng, phải SAVE/LOAD RNG state ở mỗi epoch, không chỉ set_seed
    ở đầu. Hàm này chỉ dùng cho run mới (fresh start).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic nhưng chấp nhận hit performance ~5-10%
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_state() -> dict:
    """Capture toàn bộ RNG state để lưu vào checkpoint.

    Torch RNG state MUST be torch.ByteTensor on CPU để torch.set_rng_state() chấp nhận
    khi restore. Sau torch.save/load, dtype đôi khi bị convert — ta ép sẵn ở đây.
    """
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state().cpu().to(torch.uint8),
    }
    if torch.cuda.is_available():
        # CUDA RNG là list[Tensor] — một per GPU
        state["torch_cuda_random"] = [t.cpu().to(torch.uint8) for t in torch.cuda.get_rng_state_all()]
    return state


def set_rng_state(state: dict) -> None:
    """Restore RNG state từ checkpoint (gọi khi resume).

    Defensive: ép kiểu về ByteTensor để phòng trường hợp torch.load deserialize
    sang dtype khác (vd CUDA tensor hoặc int64).
    """
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])

    torch_state = state["torch_random"]
    # Ép về torch.ByteTensor trên CPU (yêu cầu của torch.set_rng_state)
    if not isinstance(torch_state, torch.Tensor):
        torch_state = torch.tensor(torch_state, dtype=torch.uint8)
    torch_state = torch_state.cpu().to(torch.uint8)
    torch.set_rng_state(torch_state)

    if "torch_cuda_random" in state and torch.cuda.is_available():
        cuda_states = state["torch_cuda_random"]
        # Ép từng tensor về ByteTensor CPU
        fixed = []
        for t in cuda_states:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, dtype=torch.uint8)
            fixed.append(t.cpu().to(torch.uint8))
        torch.cuda.set_rng_state_all(fixed)


def atomic_torch_save(obj, path: str | Path) -> None:
    """Write-then-rename để tránh corrupted checkpoint nếu process bị kill.

    Colab thường bị kill đột ngột; nếu đang write torch.save và bị cắt giữa chừng,
    file .pt sẽ corrupted và resume sẽ crash. Pattern này ghi sang file tmp rồi
    rename — rename trên POSIX là atomic, không thể bị interrupt nửa chừng.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic rename


def atomic_json_save(obj, path: str | Path) -> None:
    """Atomic JSON save."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Core training with resume
# ---------------------------------------------------------------------------

def train_with_resume(cfg: TrainConfigV1) -> dict:
    """Train một SAE với resume-from-checkpoint support.

    Returns
    -------
    summary : dict với các key: run_name, best_val_mse, best_epoch, total_epochs_run,
              total_time_s, model_params, history, status ('completed' | 'resumed_completed')

    Behavior
    --------
    1. Check {run_name}.done flag trong ckpt_dir. Nếu có → skip, return cached summary.
    2. Check {run_name}_last.pt. Nếu có → resume từ epoch + 1 với RNG state restore.
    3. Nếu không có → fresh start với set_seed(cfg.seed).
    4. Mỗi epoch: lưu {run_name}_last.pt (atomic). Nếu val MSE cải thiện → lưu thêm
       {run_name}_best.pt.
    5. Khi kết thúc: touch {run_name}.done và lưu {run_name}_history.json.
    """
    ckpt_dir = Path(cfg.ckpt_dir)
    log_dir = Path(cfg.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    done_flag = ckpt_dir / f"{cfg.run_name}.done"
    last_ckpt = ckpt_dir / f"{cfg.run_name}_last.pt"
    best_ckpt = ckpt_dir / f"{cfg.run_name}_best.pt"
    hist_file = log_dir / f"{cfg.run_name}_history.json"
    state_file = ckpt_dir / f"{cfg.run_name}_state.json"

    # --- 1. Skip if already done ---
    if done_flag.exists():
        print(f"[SKIP] {cfg.run_name} already completed")
        if hist_file.exists():
            with open(hist_file) as f:
                history = json.load(f)
        else:
            history = []
        # Reconstruct summary từ state file
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            return {
                "run_name": cfg.run_name,
                "best_val_mse": state.get("best_val_mse"),
                "best_epoch": state.get("best_epoch"),
                "total_epochs_run": state.get("epoch", 0) + 1,
                "total_time_s": state.get("total_time_s", 0),
                "model_params": state.get("model_params"),
                "history": history,
                "status": "skipped_done",
            }
        return {"run_name": cfg.run_name, "status": "skipped_done_no_state"}

    device = resolve_device(cfg.device)
    print(f"[{cfg.run_name}] device={device}  rho={cfg.rho}  beta={cfg.beta}  seed={cfg.seed}")

    # --- 2. Setup data ---
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

    # Auto-adjust input_dim nếu lệch (vd data có 46 thay vì 47)
    if n_feat != cfg.input_dim:
        print(f"[{cfg.run_name}] input_dim override: cfg={cfg.input_dim} -> data={n_feat}")
        cfg.input_dim = n_feat

    # --- 3. Setup model + optim ---
    model = SparseAutoencoder(
        input_dim=cfg.input_dim,
        hidden_dims=tuple(cfg.hidden_dims),
        latent_dim=cfg.latent_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n_params = model.count_parameters()

    # --- 4. Attempt resume ---
    start_epoch = 0
    best_val_mse = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = []
    prev_elapsed_s = 0.0  # Thời gian đã chạy ở các phiên trước (trước khi resume)

    if last_ckpt.exists():
        print(f"[RESUME] Found {last_ckpt.name}")
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optim_state_dict"])
        set_rng_state(ckpt["rng_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_mse = ckpt["best_val_mse"]
        best_epoch = ckpt["best_epoch"]
        patience_counter = ckpt["patience_counter"]
        prev_elapsed_s = ckpt.get("total_time_s", 0.0)
        # Load history và cắt về start_epoch (tránh double entries sau resume)
        if hist_file.exists():
            with open(hist_file) as f:
                history = json.load(f)
            history = [h for h in history if h["epoch"] < start_epoch]
        print(f"[RESUME] from epoch {start_epoch}/{cfg.max_epochs}, "
              f"best_val_mse={best_val_mse:.5f} (prev elapsed: {prev_elapsed_s:.0f}s)")
    else:
        set_seed(cfg.seed)
        print(f"[FRESH] seed={cfg.seed}")

    # --- 5. Train loop ---
    t_start = time.time()
    for epoch in range(start_epoch, cfg.max_epochs):
        # ---- Train epoch ----
        model.train()
        train_mse_sum = 0.0
        train_kl_sum = 0.0
        train_n = 0
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad()
            x_hat, z = model(x)
            loss, comp = sae_loss(x, x_hat, z, rho=cfg.rho, beta=cfg.beta)
            loss.backward()
            optimizer.step()
            b = x.size(0)
            # comp["mse"] và comp["kl"] đã là Python float (detach().cpu())
            train_mse_sum += comp["mse"] * b
            train_kl_sum += comp["kl"] * b
            train_n += b
        train_mse = train_mse_sum / train_n
        train_kl = train_kl_sum / train_n

        # ---- Validate ----
        model.eval()
        val_mse_sum = 0.0
        val_kl_sum = 0.0
        rho_hat_accum = torch.zeros(cfg.latent_dim, device=device)
        val_n = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                x_hat, z = model(x)
                _, comp = sae_loss(x, x_hat, z, rho=cfg.rho, beta=cfg.beta)
                b = x.size(0)
                val_mse_sum += comp["mse"] * b
                val_kl_sum += comp["kl"] * b
                rho_hat_accum += z.sum(dim=0)
                val_n += b
        val_mse = val_mse_sum / val_n
        val_kl = val_kl_sum / val_n
        val_rho_hat = (rho_hat_accum / val_n).cpu().numpy()
        val_rho_hat_mean = float(val_rho_hat.mean())
        val_rho_hat_std = float(val_rho_hat.std())

        # ---- EarlyStopping (on pure val_mse, không bao gồm KL) ----
        improved = val_mse < best_val_mse
        if improved:
            best_val_mse = val_mse
            best_epoch = epoch
            patience_counter = 0
            # Save best-only checkpoint (compact, không optim state)
            atomic_torch_save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "cfg": asdict(cfg),
                "val_mse": val_mse,
                "val_rho_hat_mean": val_rho_hat_mean,
            }, best_ckpt)
        else:
            patience_counter += 1

        # ---- Log ----
        elapsed_s_now = prev_elapsed_s + (time.time() - t_start)

        hist_entry = {
            "epoch": epoch,
            "train_mse": train_mse, "train_kl": train_kl,
            "val_mse": val_mse, "val_kl": val_kl,
            "val_rho_hat_mean": val_rho_hat_mean,
            "val_rho_hat_std": val_rho_hat_std,
            "improved": improved,
            "patience_counter": patience_counter,
            "elapsed_s": elapsed_s_now,
        }
        history.append(hist_entry)

        if epoch % cfg.verbose_every == 0 or improved:
            tag = "✓ BEST" if improved else f"pat={patience_counter}"
            print(f"  ep {epoch:3d}/{cfg.max_epochs}  "
                  f"train MSE={train_mse:.5f}  val MSE={val_mse:.5f}  "
                  f"ρ̂={val_rho_hat_mean:.4f}  {tag}")

        # ---- Save last checkpoint MỖI EPOCH (for resume) ----
        atomic_torch_save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "rng_state": get_rng_state(),
            "cfg": asdict(cfg),
            "best_val_mse": best_val_mse,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "total_time_s": elapsed_s_now,
        }, last_ckpt)
        atomic_json_save(history, hist_file)
        atomic_json_save({
            "epoch": epoch,
            "best_val_mse": best_val_mse,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "total_time_s": elapsed_s_now,
            "model_params": n_params,
        }, state_file)

        # ---- EarlyStopping check ----
        if patience_counter >= cfg.patience:
            print(f"[EARLY STOP] patience exceeded at epoch {epoch}")
            break

    final_elapsed_s = prev_elapsed_s + (time.time() - t_start)

    # --- 6. Mark done ---
    done_flag.touch()
    summary = {
        "run_name": cfg.run_name,
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "total_epochs_run": len(history),
        "total_time_s": final_elapsed_s,
        "model_params": n_params,
        "history": history,
        "status": "completed",
    }
    atomic_json_save({k: v for k, v in summary.items() if k != "history"},
                     state_file)
    return summary
