"""
Hierarchical VAE for drum pattern generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalDrumVAE(nn.Module):
    def __init__(self, z_high_dim=4, z_low_dim=12):
        """
        Two-level VAE for drum patterns.
        
        The architecture uses a hierarchy of latent variables where z_high
        encodes style/genre information and z_low encodes pattern variations.
        
        Args:
            z_high_dim: Dimension of high-level latent (style)
            z_low_dim: Dimension of low-level latent (variation)
        """
        super().__init__()
        self.z_high_dim = z_high_dim
        self.z_low_dim = z_low_dim
        
        # Encoder: pattern → z_low → z_high
        # We use 1D convolutions treating the pattern as a sequence
        self.encoder_low = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=3, padding=1),  # [16, 9] → [16, 32]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # → [8, 64]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # → [4, 128]
            nn.ReLU(),
            nn.Flatten()  # → [512]
        )
        
        # Low-level latent parameters
        self.fc_mu_low = nn.Linear(512, z_low_dim)
        self.fc_logvar_low = nn.Linear(512, z_low_dim)
        
        # Encoder from z_low to z_high
        self.encoder_high = nn.Sequential(
            nn.Linear(z_low_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # High-level latent parameters
        self.fc_mu_high = nn.Linear(32, z_high_dim)
        self.fc_logvar_high = nn.Linear(32, z_high_dim)
        
        # Decoder: z_high → z_low → pattern
        # TODO: Implement decoder architecture
        # Mirror the encoder structure
        # Use transposed convolutions for upsampling

        self.prior_low_given_high = nn.Sequential(
            nn.Linear(z_high_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * z_low_dim)   # splits into (mu_p, logvar_p)
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(z_low_dim + z_high_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128 * 4),
            nn.ReLU()
        )
        # Upsample 4 -> 8 -> 16 timesteps and project to 9 instruments
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (128, 4)),  # -> [B, 128, 4]
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # -> [B, 64, 8]
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # -> [B, 32, 16]
            nn.ReLU(),
            nn.Conv1d(32, 9, kernel_size=1)                                   # -> [B, 9, 16]
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling.
        
        TODO: Implement
        z = mu + eps * std where eps ~ N(0,1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def encode_hierarchy(self, x):
        """
        Encode pattern to both latent levels.
        
        Args:
            x: Drum patterns [batch_size, 16, 9]
            
        Returns:
            mu_low, logvar_low: Parameters for q(z_low|x)
            mu_high, logvar_high: Parameters for q(z_high|z_low)
        """
        # Reshape for Conv1d: [batch, 16, 9] → [batch, 9, 16]
        x = x.transpose(1, 2).float()
        
        # TODO: Encode to z_low parameters
        # TODO: Sample z_low using reparameterization
        # TODO: Encode z_low to z_high parameters
        h = self.encoder_low(x)        # [B, 512]
        mu_low = self.fc_mu_low(h)
        logvar_low = self.fc_logvar_low(h)
        z_low = self.reparameterize(mu_low, logvar_low)

        h_high = self.encoder_high(z_low)
        mu_high = self.fc_mu_high(h_high)
        logvar_high = self.fc_logvar_high(h_high)
        z_high = self.reparameterize(mu_high, logvar_high)

        return z_low, (mu_low, logvar_low), z_high, (mu_high, logvar_high)
        
    def prior_low_params(self, z_high):
        """Return (mu_p, logvar_p) of p(z_low | z_high)."""
        p = self.prior_low_given_high(z_high)
        mu_p, logvar_p = torch.chunk(p, 2, dim=-1)
        return mu_p, logvar_p    

    def decode_hierarchy(self, z_high, z_low=None, temperature=1.0):
        """
        Decode from latent variables to pattern.
        
        Args:
            z_high: High-level latent code
            z_low: Low-level latent code (if None, sample from prior)
            temperature: Temperature for binary output (lower = sharper)
            
        Returns:
            pattern_logits: Logits for binary pattern [batch, 16, 9]
        """
        # TODO: If z_low is None, sample from conditional prior p(z_low|z_high)
        # TODO: Decode z_high and z_low to pattern logits
        # TODO: Apply temperature scaling before sigmoid
        if z_low is None:
            # Sample z_low ~ p(z_low | z_high)
            mu_p, logvar_p = self.prior_low_params(z_high)
            z_low = self.reparameterize(mu_p, logvar_p)

        z = torch.cat([z_low, z_high], dim=-1)
        h = self.decoder_fc(z)
        logits = self.decoder_conv(h) / max(1e-6, float(temperature))  # [B, 9, 16]
        return logits.transpose(1, 2) 
    
    @staticmethod
    def kl_standard_normal(mu, logvar):
        """KL(q||p) where p=N(0, I). Sum over latent dims, mean over batch."""
        kl = 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar)
        return kl.sum(dim=-1).mean()

    @staticmethod
    def kl_diag_normals(mu_q, logvar_q, mu_p, logvar_p):
        """
        KL( N(mu_q, diag(sigma_q^2)) || N(mu_p, diag(sigma_p^2)) ).
        Sum over latent dims, mean over batch.
        """
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        term = (logvar_p - logvar_q) + (var_q + (mu_q - mu_p)**2) / (var_p + 1e-8) - 1.0
        kl = 0.5 * term
        return kl.sum(dim=-1).mean()
    
    def forward(self, x, beta=1.0, temperature=1.0, reduction="mean"):
        """
        Full forward pass with loss computation.
        
        Args:
            x: Input patterns [batch_size, 16, 9]
            beta: KL weight for beta-VAE (use < 1 to prevent collapse)
            
        Returns:
            recon: Reconstructed patterns
            mu_low, logvar_low, mu_high, logvar_high: Latent parameters
        """
        # TODO: Encode, decode, compute losses
        z_low, (mu_low, logvar_low), z_high, (mu_high, logvar_high) = self.encode_hierarchy(x)
        logits = self.decode_hierarchy(z_high, z_low=z_low, temperature=temperature)

        # Reconstruction: Bernoulli likelihood, use BCE with logits
        bce = F.binary_cross_entropy_with_logits(
            logits, x.float(), reduction='none'
        )  # [B, 16, 9]
        if reduction == "mean":
            recon_loss = bce.mean()
        else:
            recon_loss = bce.sum()

        # KL terms
        mu_p_low, logvar_p_low = self.prior_low_params(z_high)
        kl_low = self.kl_diag_normals(mu_low, logvar_low, mu_p_low, logvar_p_low)
        kl_high = self.kl_standard_normal(mu_high, logvar_high)

        total = recon_loss + beta * (kl_low + kl_high)

        with torch.no_grad():
            recon = (torch.sigmoid(logits) > 0.5).float()

        return {
            "recon": recon,            # [B, 16, 9]
            "logits": logits,          # [B, 16, 9]
            "z_low": z_low,
            "z_high": z_high,
            "losses": {
                "total": total,
                "recon": recon_loss,
                "kl_low": kl_low,
                "kl_high": kl_high,
            },
        }

    @torch.no_grad()
    def sample(self, n=8, temperature=1.0):
        """
        Unconditional sampling from the hierarchy.
        """
        device = next(self.parameters()).device
        z_high = torch.randn(n, self.z_high_dim, device=device)  # p(z_high)
        logits = self.decode_hierarchy(z_high, z_low=None, temperature=temperature)
        return (torch.sigmoid(logits) > 0.5).float()  # [n, 16, 9]

    @torch.no_grad()
    def conditional_sample(self, z_high, n=1, temperature=1.0):
        """
        Sample patterns conditioned on a provided z_high.
        """
        z_high = z_high.expand(n, -1)
        logits = self.decode_hierarchy(z_high, z_low=None, temperature=temperature)
        return (torch.sigmoid(logits) > 0.5).float()
