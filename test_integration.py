"""
Integration test: chạy logic chính của notebook 03 trên dữ liệu tổng hợp.
Không chạy notebook thật, nhưng chạy các đoạn code quan trọng để đảm bảo:
- Pipeline dataset -> model -> train -> eval hoạt động end-to-end
- Recon error benign < attack (điều kiện cần cho SAE đúng)
- Mini grid 2x2 nhỏ chạy được
"""
import sys
from pathlib import Path

# Sử dụng src của project
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import torch
import tempfile
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

from sae_model import SparseAutoencoder
from sae_losses import sae_loss
from sae_train import TrainConfig, train


def make_fake_cic_iot():
    """Tạo dữ liệu giả lập giống CIC-IoT2023 sau EDA:
    - 46 features đã scale
    - 8 classes, benign = 0
    - benign có distribution khác attack (để SAE học được manifold)
    """
    rng = np.random.default_rng(42)
    n_benign_train = 5000
    n_benign_val = 500
    n_attack_val = 500

    # Benign = phân phối trên 2-D manifold nhúng trong 46-D
    def benign_gen(n):
        t = rng.uniform(0, 2 * np.pi, n)
        z1 = np.cos(t); z2 = np.sin(t)
        # Lift lên 46-D bằng random projection + noise nhỏ
        W = rng.standard_normal((2, 46)) * 0.5
        X = np.column_stack([z1, z2]) @ W + 0.1 * rng.standard_normal((n, 46))
        return X.astype(np.float32)

    # Attack = noise gaussian rộng, ngoài manifold
    def attack_gen(n):
        return (2.0 * rng.standard_normal((n, 46))).astype(np.float32)

    Xb_tr = benign_gen(n_benign_train)
    Xb_va = benign_gen(n_benign_val)
    Xa_va = attack_gen(n_attack_val)

    # Scale theo benign_train (như EDA)
    scaler = StandardScaler().fit(Xb_tr)
    Xb_tr_s = scaler.transform(Xb_tr).astype(np.float32)
    Xb_va_s = scaler.transform(Xb_va).astype(np.float32)
    Xa_va_s = scaler.transform(Xa_va).astype(np.float32)

    cols = [f"f{i}" for i in range(46)]
    df_tr = pd.DataFrame(Xb_tr_s, columns=cols)
    df_tr["Label"] = 0  # benign

    df_va = pd.DataFrame(
        np.vstack([Xb_va_s, Xa_va_s]),
        columns=cols,
    )
    df_va["Label"] = [0] * n_benign_val + [1] * n_attack_val

    return df_tr, df_va


def main():
    tmp = tempfile.mkdtemp()
    print(f"Temp dir: {tmp}")

    # 1. Tạo và lưu parquet giả lập
    df_tr, df_va = make_fake_cic_iot()
    tp = Path(tmp) / "train_benign.parquet"
    vp = Path(tmp) / "val.parquet"
    df_tr.to_parquet(tp)
    df_va.to_parquet(vp)
    print(f"train_benign: {len(df_tr):,} rows")
    print(f"val         : {len(df_va):,} rows (benign={sum(df_va.Label==0)}, attack={sum(df_va.Label!=0)})")

    # 2. Train SAE base config (ρ=0.05, β=3.0) — như cell 9 của notebook
    cfg = TrainConfig(
        train_path=str(tp), val_path=str(vp),
        benign_label=0,
        max_train_samples=None, max_val_samples=None,
        input_dim=46, hidden_dims=(32, 16), latent_dim=8,
        rho=0.05, beta=3.0,
        lr=1e-3, batch_size=256,
        max_epochs=20, patience=5,
        num_workers=0, device="cpu", seed=0,
        run_name="integration_test",
        ckpt_dir=tmp + "/ckpt", log_dir=tmp + "/log",
        verbose_every=5,
    )
    summary = train(cfg)
    print(f"\n=== Base training done: val_mse={summary['best_val_mse']:.5f} ===")

    # 3. Sanity check: recon error benign vs attack — như §3 của notebook
    ckpt = torch.load(tmp + "/ckpt/integration_test_best.pt", weights_only=False)
    model = SparseAutoencoder(input_dim=46, hidden_dims=(32, 16), latent_dim=8)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    feat_cols = [c for c in df_va.columns if c != "Label"]
    Xb = torch.from_numpy(df_va[df_va.Label == 0][feat_cols].to_numpy(np.float32))
    Xa = torch.from_numpy(df_va[df_va.Label != 0][feat_cols].to_numpy(np.float32))

    @torch.no_grad()
    def recon_err(X):
        xh, _ = model(X)
        return ((xh - X) ** 2).mean(dim=1).numpy()

    eB = recon_err(Xb)
    eA = recon_err(Xa)
    print(f"Recon MSE benign: mean={eB.mean():.4f}  median={np.median(eB):.4f}")
    print(f"Recon MSE attack: mean={eA.mean():.4f}  median={np.median(eA):.4f}")
    ratio = np.median(eA) / np.median(eB)
    print(f"Ratio attack/benign (median): {ratio:.2f}×")

    y_true = np.concatenate([np.zeros(len(eB)), np.ones(len(eA))])
    y_score = np.concatenate([eB, eA])
    auc = roc_auc_score(y_true, y_score)
    print(f"AUC recon-MSE: {auc:.4f}")
    assert auc > 0.70, f"AUC quá thấp ({auc:.3f}); SAE không học đúng manifold"
    assert ratio > 1.2, f"Attack err không cao hơn benign đáng kể ({ratio:.2f}×)"
    print("✓ Sanity check PASSED")

    # 4. Mini grid 2x2 — như §4 của notebook
    print("\n=== Mini grid 2x2 ===")
    grid_results = []
    for rho in [0.05, 0.10]:
        for beta in [1.0, 3.0]:
            cfg2 = TrainConfig(
                train_path=str(tp), val_path=str(vp),
                benign_label=0,
                max_train_samples=None, max_val_samples=None,
                input_dim=46, hidden_dims=(32, 16), latent_dim=8,
                rho=rho, beta=beta,
                lr=1e-3, batch_size=256,
                max_epochs=10, patience=5,
                num_workers=0, device="cpu", seed=0,
                run_name=f"rho={rho}_beta={beta}",
                ckpt_dir=tmp + "/grid_ckpt", log_dir=tmp + "/grid_log",
                verbose_every=20,  # silent
            )
            s = train(cfg2)
            grid_results.append({"rho": rho, "beta": beta,
                                 "val_mse": s["best_val_mse"]})
    gdf = pd.DataFrame(grid_results)
    print(gdf.to_string())
    print("\n✓ Integration test ALL PASSED")


if __name__ == "__main__":
    main()
