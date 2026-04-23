"""
Grid search sơ bộ ρ × β cho SAE v0 (Tuần 3).

Theo RQ5 của đề cương: ablation rho ∈ {0.01, 0.05, 0.1, 0.2}, beta ∈ {1, 3, 10}.
Tuần 3: thu hẹp còn 3×3 = 9 combo trên sub-sample 50k benign, 50 epoch để:
  - Verify pipeline chạy ổn trên data thật
  - Chọn 1–2 điểm đẹp để làm base cho Tuần 4 grid đầy đủ + 5 seed

Chạy:
    python3 sae_grid_search.py --config ../configs/sae_v0.yaml

Output:
    ../logs/grid_<timestamp>/
        ├── result_rho=0.01_beta=1.0.json
        ├── ...
        └── summary.csv             # tổng hợp để đọc nhanh
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

# Import từ cùng thư mục src/
sys.path.insert(0, str(Path(__file__).parent))
from sae_train import TrainConfig, train


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cfg_from_yaml(y: dict, rho: float, beta: float, run_name: str,
                  log_dir: str, ckpt_dir: str) -> TrainConfig:
    """Build TrainConfig từ YAML, override rho/beta/run_name/dirs."""
    return TrainConfig(
        train_path=y["data"]["train_path"],
        val_path=y["data"]["val_path"],
        benign_label=y["data"]["benign_label"],
        max_train_samples=y["data"].get("max_train_samples"),
        max_val_samples=y["data"].get("max_val_samples"),
        input_dim=y["model"]["input_dim"],
        hidden_dims=tuple(y["model"]["hidden_dims"]),
        latent_dim=y["model"]["latent_dim"],
        rho=rho,
        beta=beta,
        lr=y["optim"]["lr"],
        batch_size=y["optim"]["batch_size"],
        max_epochs=y["optim"]["max_epochs"],
        patience=y["optim"]["patience"],
        num_workers=y["infra"]["num_workers"],
        device=y["infra"]["device"],
        seed=y["infra"]["seed"],
        run_name=run_name,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir,
        verbose_every=y["logging"].get("verbose_every", 5),
    )


def main(args):
    y = load_yaml(args.config)
    rhos = y["grid_search"]["rho_values"]
    betas = y["grid_search"]["beta_values"]

    ts = time.strftime("%Y%m%d_%H%M%S")
    grid_root = Path(y["logging"]["log_dir"]) / f"grid_{ts}"
    ckpt_root = Path(y["logging"]["ckpt_dir"]) / f"grid_{ts}"
    grid_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    results = []
    combos = list(itertools.product(rhos, betas))
    print(f"=== Grid search: {len(combos)} combos ===")
    print(f"  rho values : {rhos}")
    print(f"  beta values: {betas}")
    print(f"  log dir    : {grid_root}")
    print()

    t_total = time.time()
    for i, (rho, beta) in enumerate(combos, 1):
        run_name = f"rho={rho}_beta={beta}"
        print(f"--- [{i}/{len(combos)}] {run_name} ---")
        cfg = cfg_from_yaml(
            y, rho=rho, beta=beta, run_name=run_name,
            log_dir=str(grid_root), ckpt_dir=str(ckpt_root),
        )
        summary = train(cfg)

        # Trích các chỉ số chính cho CSV
        last = summary["history"][-1] if summary["history"] else {}
        row = {
            "rho": rho,
            "beta": beta,
            "best_val_mse": summary["best_val_mse"],
            "best_epoch": summary["best_epoch"],
            "epochs_run": summary["total_epochs_run"],
            "total_time_s": summary["total_time_s"],
            "final_rho_hat_mean": last.get("val_rho_hat_mean"),
            "final_rho_hat_std": last.get("val_rho_hat_std"),
            "model_params": summary["model_params"],
        }
        results.append(row)
        print()

    # Ghi CSV tổng hợp
    csv_path = grid_root / "summary.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

    # Ghi JSON tổng hợp cho đầy đủ
    with open(grid_root / "summary.json", "w") as f:
        json.dump({
            "config": y,
            "results": results,
            "total_time_s": time.time() - t_total,
        }, f, indent=2, default=str)

    # In bảng console đẹp
    print("\n" + "=" * 70)
    print("GRID SEARCH SUMMARY")
    print("=" * 70)
    print(f"{'rho':>6} {'beta':>6} {'best_val_mse':>14} "
          f"{'best_ep':>8} {'epochs':>7} {'rho_hat':>10} {'time_s':>8}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["best_val_mse"]):
        rh = r["final_rho_hat_mean"] or 0.0
        print(f"{r['rho']:>6.3f} {r['beta']:>6.1f} {r['best_val_mse']:>14.6f} "
              f"{r['best_epoch']:>8d} {r['epochs_run']:>7d} "
              f"{rh:>10.4f} {r['total_time_s']:>8.1f}")
    print("=" * 70)
    print(f"\nAll logs: {grid_root}")
    print(f"CSV summary: {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../configs/sae_v0.yaml")
    args = p.parse_args()
    main(args)
