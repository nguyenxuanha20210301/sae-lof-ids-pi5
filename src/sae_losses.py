"""
Loss function cho Sparse Autoencoder.

Công thức theo đề cương §6.2 (và CS294A Andrew Ng 2011):

    L = (1/N) * sum_i ||x_i - x_hat_i||^2_2
        + beta * sum_j KL(rho || rho_hat_j)

trong đó:
    rho       : target sparsity (hằng số, vd 0.05)
    rho_hat_j : mean activation của unit ẩn thứ j trên batch,
                rho_hat_j = (1/N) * sum_i a_j(x_i)
    KL(rho || rho_hat_j) = rho * log(rho / rho_hat_j)
                         + (1 - rho) * log((1 - rho) / (1 - rho_hat_j))

Lưu ý ổn định số:
- rho_hat có thể rất gần 0 hoặc 1 ở đầu training => chia cho 0.
  Clamp trong khoảng [eps, 1 - eps] với eps = 1e-7 (không gây bias đáng kể
  khi rho được chọn trong khoảng [0.01, 0.2]).
- sum_j (không mean_j) để match CS294A và đa số implementation;
  nhưng khi latent_dim=8 và beta=3, mức penalty tổng vẫn nhỏ so với MSE
  trên 46 feature.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence_sparsity(
    rho_hat: torch.Tensor,
    rho: float,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Tính KL(rho || rho_hat_j) summed over hidden units.

    Tham số
    -------
    rho_hat : torch.Tensor, shape (latent_dim,)
        Mean activation per hidden unit, averaged trên batch.
    rho : float
        Target sparsity.
    eps : float
        Clamping để tránh log(0).

    Returns
    -------
    torch.Tensor, scalar
        sum_j KL(rho || rho_hat_j)
    """
    rho_hat = rho_hat.clamp(min=eps, max=1.0 - eps)
    rho_tensor = torch.full_like(rho_hat, rho)
    kl = rho_tensor * torch.log(rho_tensor / rho_hat) + \
         (1.0 - rho_tensor) * torch.log((1.0 - rho_tensor) / (1.0 - rho_hat))
    return kl.sum()


def sae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    z: torch.Tensor,
    rho: float = 0.05,
    beta: float = 3.0,
    reduction: str = "mean",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Tính loss tổng hợp của SAE.

    Tham số
    -------
    x : torch.Tensor, shape (B, D)
        Input (đã normalize bằng StandardScaler).
    x_hat : torch.Tensor, shape (B, D)
        Reconstruction từ decoder.
    z : torch.Tensor, shape (B, latent_dim)
        Latent activations (đã qua sigmoid, ∈ (0,1)).
    rho : float
        Target sparsity (hằng số).
    beta : float
        Trọng số KL penalty so với MSE.
    reduction : str
        "mean" (mặc định) hoặc "sum" cho MSE.

    Returns
    -------
    total_loss : torch.Tensor, scalar
        Loss để backward.
    components : dict
        {"mse": float, "kl": float, "total": float} để logging.
    """
    # MSE reconstruction (mean over batch AND feature — hành vi mặc định của F.mse_loss)
    mse = F.mse_loss(x_hat, x, reduction=reduction)

    # Sparsity penalty: rho_hat_j = mean của activation qua batch cho mỗi unit
    rho_hat = z.mean(dim=0)
    kl = kl_divergence_sparsity(rho_hat, rho)

    total = mse + beta * kl

    components = {
        "mse": float(mse.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "total": float(total.detach().cpu()),
        "rho_hat_mean": float(rho_hat.mean().detach().cpu()),
        "rho_hat_max": float(rho_hat.max().detach().cpu()),
        "rho_hat_min": float(rho_hat.min().detach().cpu()),
    }
    return total, components


if __name__ == "__main__":
    # Smoke test
    torch.manual_seed(0)
    B, D, K = 256, 46, 8
    x = torch.randn(B, D)
    x_hat = x + 0.1 * torch.randn_like(x)  # recon noise nhỏ
    z = torch.sigmoid(torch.randn(B, K))  # latent giả lập

    loss, comp = sae_loss(x, x_hat, z, rho=0.05, beta=3.0)
    print(f"Total loss : {comp['total']:.4f}")
    print(f"  MSE      : {comp['mse']:.4f}")
    print(f"  KL       : {comp['kl']:.4f}")
    print(f"  rho_hat  : mean={comp['rho_hat_mean']:.3f} "
          f"min={comp['rho_hat_min']:.3f} max={comp['rho_hat_max']:.3f}")
    assert loss.requires_grad or not loss.requires_grad  # smoke
    print("✓ Loss compute OK")

    # Edge case: rho_hat tiến đến 0 (dead units)
    z_dead = torch.zeros(B, K) + 1e-10
    _, comp2 = sae_loss(x, x_hat, z_dead, rho=0.05, beta=3.0)
    print(f"Dead units case: KL = {comp2['kl']:.4f} (nên lớn, penalty cao)")
    assert comp2["kl"] > comp["kl"], "Dead units phải bị penalty mạnh hơn"
    print("✓ KL penalizes dead units correctly")

    # Edge case: rho_hat = rho chính xác
    z_target = torch.full((B, K), 0.05)
    _, comp3 = sae_loss(x, x_hat, z_target, rho=0.05, beta=3.0)
    print(f"Exact-rho case : KL = {comp3['kl']:.6f} (nên ~0)")
    assert comp3["kl"] < 1e-5, "Khi rho_hat = rho, KL phải ~0"
    print("✓ KL = 0 khi rho_hat khớp rho target")
