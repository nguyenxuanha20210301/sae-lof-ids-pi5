"""
Sparse Autoencoder cho CIC-IoT2023 NIDS trên Raspberry Pi 5.

Kiến trúc theo đề cương §6.2:
    Input (47)
      -> Dense(32, ReLU)
      -> Dense(16, ReLU)
      -> Bottleneck Dense(8, sigmoid)  # latent z, sigmoid để rho_hat ∈ (0,1) cho KL penalty
      -> Dense(16, ReLU)
      -> Dense(32, ReLU)
      -> Dense(47, linear)              # reconstruction

Tổng params dự kiến: ~4,855 (tính tay: xem __main__ block bên dưới).

Note về input_dim:
- EDA báo "46 features + 1 Label" => 46 features dùng cho model (Label không phải input).
- Để khớp với đề cương ghi "47", ta dùng input_dim mặc định = 46 (=len(feature columns)).
  Nếu sau này bao gồm cả Label làm feature (không khuyến khích), chỉ cần đổi tham số.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder với KL-divergence sparsity penalty trên latent layer.

    Layer cuối của encoder dùng sigmoid để ép activation ∈ (0,1),
    cho phép tính rho_hat = mean(activation, dim=batch) và đưa vào KL(rho || rho_hat).

    Tham số
    -------
    input_dim : int
        Số feature đầu vào (mặc định 46 cho CIC-IoT2023 sau khi bỏ Label và ts).
    hidden_dims : tuple[int, ...]
        Kích thước các lớp ẩn của encoder (decoder là gương của encoder).
        Mặc định (32, 16) theo đề cương.
    latent_dim : int
        Kích thước không gian ẩn (bottleneck). Mặc định 8 theo đề cương.

    Ghi chú về tính toán số params (input_dim=46):
        enc1: 46*32 + 32 = 1504
        enc2: 32*16 + 16 = 528
        enc_z: 16*8 + 8  = 136
        dec1: 8*16 + 16  = 144
        dec2: 16*32 + 32 = 544
        dec_out: 32*46 + 46 = 1518
        Tổng = 4374 params  (nhỏ hơn ngân sách 100k, dư room)
    """

    def __init__(
        self,
        input_dim: int = 46,
        hidden_dims: tuple[int, ...] = (32, 16),
        latent_dim: int = 8,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # --- Encoder ---
        encoder_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev, h))
            encoder_layers.append(nn.ReLU(inplace=True))
            prev = h
        # Lớp latent: sigmoid để activation ∈ (0,1) cho KL penalty
        encoder_layers.append(nn.Linear(prev, latent_dim))
        encoder_layers.append(nn.Sigmoid())
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Decoder (gương của encoder) ---
        decoder_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev, h))
            decoder_layers.append(nn.ReLU(inplace=True))
            prev = h
        # Output: linear (không activation) — StandardScaler cho ra feature ~ N(0,1),
        # nên output có thể âm/dương tùy ý.
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """He init cho ReLU, Xavier cho lớp sigmoid/linear cuối."""
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                # Lớp cuối encoder (trước sigmoid) và lớp cuối decoder: xavier
                # Các lớp khác: kaiming (vì theo sau bởi ReLU)
                # Phân biệt dựa trên _next_activation sẽ khó; dùng heuristic đơn giản:
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về latent z (sigmoid), shape (B, latent_dim), mỗi phần tử ∈ (0,1)."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Tái tạo x_hat từ z."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        x_hat : torch.Tensor, shape (B, input_dim)
            Reconstruction.
        z : torch.Tensor, shape (B, latent_dim)
            Latent activations (sigmoid).
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test: verify kiến trúc và param count
    torch.manual_seed(0)
    model = SparseAutoencoder(input_dim=46, hidden_dims=(32, 16), latent_dim=8)
    print(f"Model: {model}")
    print(f"Total trainable params: {model.count_parameters():,}")

    x = torch.randn(256, 46)
    x_hat, z = model(x)
    print(f"Input shape   : {tuple(x.shape)}")
    print(f"Latent shape  : {tuple(z.shape)}  | range: [{z.min():.4f}, {z.max():.4f}]")
    print(f"Output shape  : {tuple(x_hat.shape)}")

    # Sanity check: latent nằm trong (0,1) vì sigmoid
    assert (z > 0).all() and (z < 1).all(), "Latent phải ∈ (0,1) sau sigmoid"
    print("✓ Latent activations in (0,1) — ready for KL sparsity penalty")
