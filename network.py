import torch
from torch import nn


class Encoder(nn.Sequential):
    def __init__(self, hidden_dim=64, kernel_size=5, padding=2):
        super().__init__(
            nn.Conv2d(3, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )


class Decoder(nn.Sequential):
    def __init__(
        self,
        hidden_dim=64,
        output_dim=4,
        kernel_size=5,
        padding=2,
        stride=2,
        output_kernel_size=3,
        output_padding=1,
    ):
        super().__init__(
            nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                output_padding=output_padding,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                output_padding=0,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_dim,
                output_dim,
                kernel_size=output_kernel_size,
                stride=1,
                padding=0,
            ),
        )


class PositionEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(4, hidden_dim)

    def forward(self, x):
        spatial_shape = x.shape[-3:-1]
        grid = torch.stack(
            torch.meshgrid(*[torch.linspace(0.0, 1.0, r, device=x.device) for r in spatial_shape]), dim=-1
        )
        grid = torch.cat([grid, 1 - grid], dim=-1)
        return x + self.dense(grid)
