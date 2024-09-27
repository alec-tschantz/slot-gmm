import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, num_iter, num_slots, input_size, epsilon=1e-8):
        super().__init__()
        self.num_iter = num_iter
        self.num_slots = num_slots
        self.input_size = input_size
        self.epsilon = epsilon

        self.slots_mu = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, num_slots, input_size)))
        self.slots_log_sigma = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, num_slots, input_size)))
        self.mixing_coefficients = nn.Parameter(torch.full((1, num_slots), 1.0 / num_slots))

    def forward(self, inputs):
        B, N, D = inputs.size()
        K = self.num_slots

        mu = self.slots_mu.expand(B, -1, -1)  # Shape: (B, K, D)
        log_sigma = self.slots_log_sigma.expand(B, -1, -1)  # Shape: (B, K, D)
        sigma = torch.exp(log_sigma)  # Variances, Shape: (B, K, D)
        pi = self.mixing_coefficients.expand(B, -1)  # Shape: (B, K)

        for _ in range(self.num_iter):
            x_expanded = inputs.unsqueeze(2)  # Shape: (B, N, 1, D)
            mu_expanded = mu.unsqueeze(1)  # Shape: (B, 1, K, D)
            sigma_expanded = sigma.unsqueeze(1)  # Shape: (B, 1, K, D)

            diff = x_expanded - mu_expanded  # Shape: (B, N, K, D)
            exponent = -0.5 * ((diff**2) / (sigma_expanded + self.epsilon))
            log_coeff = -0.5 * torch.log(2 * torch.pi * sigma_expanded + self.epsilon)
            logits = torch.sum(log_coeff + exponent, dim=-1)  # Shape: (B, N, K)

            log_pi = torch.log(pi + self.epsilon).unsqueeze(1)  # Shape: (B, 1, K)
            logits = logits + log_pi  # Shape: (B, N, K)
            gamma = F.softmax(logits, dim=-1)  # Shape: (B, N, K)
            pi = torch.mean(gamma, dim=1)  # Shape: (B, K)

            gamma_sum = gamma.sum(dim=1, keepdim=True) + self.epsilon  # Shape: (B, 1, K)
            mu = torch.bmm(gamma.transpose(1, 2), inputs) / gamma_sum.transpose(1, 2)  # Shape: (B, K, D)

            diff = inputs.unsqueeze(2) - mu.unsqueeze(1)  # Shape: (B, N, K, D)
            gamma_expanded = gamma.unsqueeze(-1)  # Shape: (B, N, K, 1)
            sigma = torch.sum(gamma_expanded * diff**2, dim=1) / gamma_sum.transpose(1, 2)  # Shape: (B, K, D)

        slots = mu + sigma * torch.randn(B, K, D)
        return slots


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


class Model(nn.Module):

    def __init__(
        self,
        resolution=(35, 35),
        num_slots=4,
        num_iter=3,
        hidden_dim=32,
        decoder_initial_size=(6, 6),
    ):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iter = num_iter
        self.hidden_dim = hidden_dim
        self.decoder_initial_size = decoder_initial_size

        self.encoder_cnn = Encoder(hidden_dim)
        self.encoder_pos = PositionEmbedding(hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.gmm_attention = Attention(num_iter=num_iter, num_slots=num_slots, input_size=hidden_dim)

        self.decoder_pos = PositionEmbedding(hidden_dim)
        self.decoder_cnn = Decoder(hidden_dim, output_dim=4)

    def forward(self, image):
        x = self.encoder_cnn(image).movedim(1, -1)
        x = self.encoder_pos(x)
        x = self.mlp(self.layer_norm(x))

        slots = self.gmm_attention(x.flatten(start_dim=1, end_dim=2))
        x = slots.reshape(-1, 1, 1, slots.shape[-1]).expand(-1, *self.decoder_initial_size, -1)
        x = self.decoder_pos(x)
        x = self.decoder_cnn(x.movedim(-1, 1))

        x = F.interpolate(x, image.shape[-2:], mode="bilinear")
        x = x.unflatten(0, (len(image), len(x) // len(image)))

        recons, masks = x.split((3, 1), dim=2)
        masks = masks.softmax(dim=1)
        recon_combined = (recons * masks).sum(dim=1)

        return recon_combined, recons, masks, slots
