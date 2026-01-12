import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE_cost(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden_dim=128, encoder_layers=3, decoder_layers=3, cost_predictor_layers=2):
        """
        input_dim: 2 * D (v_norm + is_zero concat한 차원)
        latent_dim: latent space 차원
        hidden_dim: MLP hidden 크기
        """
        super().__init__()

        # Encoder
        encoder_modules = []
        encoder_current_dim = input_dim
        for layer in range(encoder_layers):
            encoder_modules.append(nn.Linear(encoder_current_dim, hidden_dim))
            encoder_modules.append(nn.ReLU())
            encoder_current_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_modules)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        decoder_modules = []
        decoder_current_dim = latent_dim
        for layer in range(decoder_layers):
            decoder_modules.append(nn.Linear(decoder_current_dim, hidden_dim))
            decoder_modules.append(nn.ReLU())
            decoder_current_dim = hidden_dim
        decoder_modules.append(nn.Linear(hidden_dim, input_dim))
        # 출력은 연속값이니까 activation 없이 그대로
        self.decoder = nn.Sequential(*decoder_modules)


        # Cost Predictor
        cost_predictor_modules = []
        cost_predictor_current_dim = latent_dim
        for layer in range(cost_predictor_layers):
            cost_predictor_modules.append(nn.Linear(cost_predictor_current_dim, hidden_dim))
            cost_predictor_modules.append(nn.ReLU())
            cost_predictor_current_dim = hidden_dim
        cost_predictor_modules.append(nn.Linear(hidden_dim, 1))  # 비용 예측은 스칼라 값
        self.cost_predictor = nn.Sequential(*cost_predictor_modules)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def predict_cost(self, z):
        return self.cost_predictor(z).squeeze(-1)  # (B,)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        cost_pred = self.cost_predictor(z).squeeze(-1)  # (B,)
        return x_recon, mu, logvar, z, cost_pred