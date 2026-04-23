"""
Dataset cho Sparse Autoencoder trên CIC-IoT2023 benign flows.

Input: parquet file đã được:
  - Stratified split 80/10/10
  - Scaled bằng StandardScaler fit trên benign-train
  - Giữ 46 features (bỏ ts và Label)

Kỳ vọng 2 file từ EDA:
  /home/kali/sae-lof-ids-pi5/data/processed/train_benign.parquet
  /home/kali/sae-lof-ids-pi5/data/processed/val.parquet        (dùng để validation recon loss)

Lưu ý validation:
- `train_benign.parquet` chỉ chứa benign — dùng cho training.
- `val.parquet` chứa cả benign và attack (validation tổng).
  Với SAE pure-benign training, validation loss = MSE trên benign-val-only.
  Ta lọc Label == 0 (benign) khi build val loader.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Cột Label trong parquet — EDA đã map 34 lớp -> 8 categories.
# Sẽ verify tự động khi load. Benign thường là label = 0.
# LABEL_CANDIDATES = ("Label", "label", "y", "target")

# Thứ tự ưu tiên: y8 (default cho SAE, đã map về 8 category theo đề cương),
# y_bin (tương đương cho binary NIDS), y34 (fine-grained multi-class).
# Giữ các tên legacy ở cuối để tương thích dataset khác.
LABEL_CANDIDATES = ("y8", "y_bin", "y34", "Label", "label", "y", "target")


def _find_label_col(df: pd.DataFrame) -> str | None:
    """Tìm cột label; trả None nếu không có (vd train_benign thuần)."""
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    return None


class BenignFlowDataset(Dataset):
    """Dataset các flow benign đã scale, trả về tensor float32.

    Tham số
    -------
    parquet_path : str | Path
        Đường dẫn parquet.
    benign_label : int | None
        Nếu file chứa cột Label (vd val.parquet), lọc về giá trị này.
        None nghĩa là dùng toàn bộ (cho train_benign.parquet thuần benign).
    max_samples : int | None
        Nếu set, sample ngẫu nhiên tối đa N dòng (cho quick iteration).
    seed : int
        Seed cho sampling.
    feature_cols : list[str] | None
        Nếu None, tự động chọn mọi cột số ngoại trừ cột label.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        benign_label: int | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        feature_cols: list[str] | None = None,
    ) -> None:
        self.path = Path(parquet_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Parquet không tồn tại: {self.path}")

        df = pd.read_parquet(self.path)

        # Xác định cột label
        label_col = _find_label_col(df)

        # Lọc benign nếu cần
        if benign_label is not None:
            if label_col is None:
                raise ValueError(
                    f"Yêu cầu lọc benign_label={benign_label} nhưng file không có cột Label. "
                    f"Columns: {list(df.columns)[:10]}..."
                )
            df = df[df[label_col] == benign_label].reset_index(drop=True)

        # Sub-sample
        if max_samples is not None and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

        # Chọn feature columns — LOẠI TẤT CẢ các cột label (y_bin, y8, y34),
        # không chỉ cột đang dùng làm label_col cho filter.
        # Lý do: EDA lưu cả 3 resolution trong cùng parquet; nếu lọt vào feature
        # sẽ làm input_dim sai và rò rỉ label vào SAE.
        ALL_LABEL_COLS = set(LABEL_CANDIDATES)
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in ALL_LABEL_COLS]
        self.feature_cols = feature_cols

        X = df[feature_cols].to_numpy(dtype=np.float32)

        # Handle NaN/Inf còn sót (EDA nên đã xử lý, nhưng phòng hờ)
        n_bad = (~np.isfinite(X)).sum()
        if n_bad > 0:
            print(f"[BenignFlowDataset] Warn: {n_bad} non-finite -> thay 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        self.X = torch.from_numpy(X.copy())
        self.n_samples, self.n_features = self.X.shape

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Pure autoencoder: target = input
        return self.X[idx]

    def summary(self) -> str:
        return (
            f"BenignFlowDataset({self.path.name}): "
            f"{self.n_samples:,} rows × {self.n_features} features"
        )


def make_dataloaders(
    train_path: str | Path,
    val_path: str | Path,
    batch_size: int = 256,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    benign_label: int = 0,
    num_workers: int = 2,
    seed: int = 0,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """Build train & val dataloaders. Val chỉ chứa benign (để đo MSE recon).

    Returns
    -------
    train_loader, val_loader, n_features
    """
    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = BenignFlowDataset(
        train_path,
        benign_label=None,  # train_benign.parquet là thuần benign
        max_samples=max_train_samples,
        seed=seed,
    )
    val_ds = BenignFlowDataset(
        val_path,
        benign_label=benign_label,  # lọc benign từ val tổng
        max_samples=max_val_samples,
        seed=seed,
        feature_cols=train_ds.feature_cols,  # đảm bảo cùng schema
    )

    print(train_ds.summary())
    print(val_ds.summary())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader, train_ds.n_features


if __name__ == "__main__":
    # Smoke test với parquet giả lập (khi chạy thật sẽ dùng path EDA)
    import tempfile

    rng = np.random.default_rng(0)
    n = 1000
    cols = [f"feat_{i}" for i in range(46)] + ["Label"]
    df = pd.DataFrame(
        np.column_stack([rng.standard_normal((n, 46)).astype(np.float32),
                         rng.integers(0, 8, size=n)]),
        columns=cols,
    )
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.to_parquet(f.name)
        ds = BenignFlowDataset(f.name, benign_label=0, max_samples=100)
        print(ds.summary())
        print(f"Shape of sample 0: {ds[0].shape} dtype={ds[0].dtype}")
        assert ds[0].dtype == torch.float32
        print("✓ Dataset smoke test OK")
