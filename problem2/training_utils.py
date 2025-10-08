"""
Training implementations for hierarchical VAE with posterior collapse prevention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def kl_anneal_schedule(epoch, num_epochs=100, cycles=4, min_beta=0.0, max_beta=1.0, method='cyclical'):
    """
    TODO: Implement KL annealing schedule
    Start with beta ≈ 0, gradually increase to 1.0
    Consider cyclical annealing for better results
    """
    if method == 'linear':
        beta = min_beta + (max_beta - min_beta) * (epoch / num_epochs)
    elif method == 'cyclical':
        cycle_length = max(1, num_epochs // max(1, cycles))
        progress = (epoch % cycle_length) / float(cycle_length)
        beta = min_beta + progress * (max_beta - min_beta)
    elif method == 'sigmoid':
        import math
        progress = epoch / num_epochs
        beta = min_beta + (max_beta - min_beta) / (1 + math.exp(-12 * (progress - 0.5)))
    else:
        raise ValueError(f"Unknown KL anneal method: {method}")
    return float(np.clip(beta, min_beta, max_beta))

def temp_anneal_schedule(epoch, start_temp=2.0, end_temp=0.7, num_epochs=100):
    t = start_temp - (start_temp - end_temp) * (epoch / num_epochs)
    return max(end_temp, t)
    

def train_hierarchical_vae(model, data_loader, num_epochs=100, device='mps'):
    """
    Train hierarchical VAE with KL annealing and other tricks.
    
    Implements several techniques to prevent posterior collapse:
    1. KL annealing (gradual beta increase)
    2. Free bits (minimum KL per dimension)
    3. Temperature annealing for discrete outputs
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    free_bits = 0.5
    
    history = defaultdict(list)
    
    for epoch in range(num_epochs):
        model.train()
        beta = kl_anneal_schedule(epoch, num_epochs=num_epochs, method='linear', min_beta=0.2, max_beta=0.5)
        temperature = temp_anneal_schedule(epoch, num_epochs=num_epochs)
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            patterns = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            optimizer.zero_grad()
            # TODO: Implement training step
            # 1. Forward pass through hierarchical VAE
            z_low, (mu_low, logvar_low), z_high, (mu_high, logvar_high) = model.encode_hierarchy(patterns)
            if hasattr(model, "prior_low_params"):
                mu_p_low, logvar_p_low = model.prior_low_params(z_high)
            else:
                mu_p_low = torch.zeros_like(mu_low)
                logvar_p_low = torch.zeros_like(logvar_low)
            logits = model.decode_hierarchy(z_high, z_low=z_low, temperature=temperature)
            # 2. Compute reconstruction loss
            recon = F.binary_cross_entropy_with_logits(logits, patterns.float(), reduction="mean")
            # 3. Compute KL divergences (both levels) and Apply free bits to prevent collapse
            kl_high_dim = 0.5 * (torch.exp(logvar_high) + mu_high**2 - 1.0 - logvar_high)
            var_q = torch.exp(logvar_low); var_p = torch.exp(logvar_p_low)
            kl_low_dim = 0.5 * (logvar_p_low - logvar_low + (var_q + (mu_low - mu_p_low) ** 2) / var_p - 1.0)
            kl_high = torch.clamp(kl_high_dim, min=free_bits).sum(-1).mean()
            kl_low  = torch.clamp(kl_low_dim,  min=free_bits).sum(-1).mean()
            # 4. Total loss = recon_loss + beta * kl_loss
            total_loss = recon + beta * (kl_low + kl_high)
            # 5. Backward and optimize
            total_loss.backward()
            optimizer.step()

            epoch_losses["recon"] += recon.item()
            epoch_losses["kl_low"] += kl_low.item()
            epoch_losses["kl_high"] += kl_high.item()
            epoch_losses["total"] += total_loss.item()
            num_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            history[k].append(epoch_losses[k])
        history["beta"].append(beta)
        history["temperature"].append(temperature)

        print(
            f"[Epoch {epoch+1:03d}] "
            f"Loss={epoch_losses['total']:.4f} | "
            f"Recon={epoch_losses['recon']:.4f} | "
            f"KL_low={epoch_losses['kl_low']:.4f} | "
            f"KL_high={epoch_losses['kl_high']:.4f} | "
            f"β={beta:.3f} | T={temperature:.2f}"
        )
            
    return history

@torch.no_grad()
def sample_diverse_patterns(model, n_styles=5, n_variations=10, device='mps'):
    """
    Generate diverse drum patterns using the hierarchy.
    
    TODO:
    1. Sample n_styles from z_high prior
    2. For each style, sample n_variations from conditional p(z_low|z_high)
    3. Decode to patterns
    4. Organize in grid showing style consistency
    """
    model.eval()
    model.to(device)

    styles = []
    for _ in range(n_styles):
        z_high = torch.randn(1, model.z_high_dim, device=device)
        style_patterns = []
        for _ in range(n_variations):
            if hasattr(model, "conditional_sample"):
                sample = model.conditional_sample(z_high, n=1)
            else:
                logits = model.decode_hierarchy(z_high, z_low=None, temperature=1.0)
                probs = torch.sigmoid(logits)
                sample = torch.bernoulli(probs).float().cpu()
            style_patterns.append(sample.cpu().numpy())
        styles.append(np.concatenate(style_patterns, axis=0))
    return styles
    
@torch.no_grad()
def analyze_posterior_collapse(model, data_loader, device='mps'):
    """
    Diagnose which latent dimensions are being used.
    
    TODO:
    1. Encode validation data
    2. Compute KL divergence per dimension
    3. Identify collapsed dimensions (KL ≈ 0)
    4. Return utilization statistics
    """
    model.eval()
    model.to(device)

    kl_low_vals = []
    kl_high_vals = []

    for batch in data_loader:
        patterns = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        _, (mu_low, logvar_low), _, (mu_high, logvar_high) = model.encode_hierarchy(patterns)

        kl_low = 0.5 * (torch.exp(logvar_low) + mu_low**2 - 1 - logvar_low)
        kl_high = 0.5 * (torch.exp(logvar_high) + mu_high**2 - 1 - logvar_high)

        kl_low_vals.append(kl_low.mean(dim=0).cpu().numpy())
        kl_high_vals.append(kl_high.mean(dim=0).cpu().numpy())

    kl_low_mean = np.mean(kl_low_vals, axis=0)
    kl_high_mean = np.mean(kl_high_vals, axis=0)

    collapse_threshold = 0.05
    collapsed_low = (kl_low_mean < collapse_threshold).sum()
    collapsed_high = (kl_high_mean < collapse_threshold).sum()

    print("Posterior Collapse Analysis:")
    print(f"Low-level latent dims: {model.z_low_dim} | Active: {model.z_low_dim - collapsed_low} | Collapsed: {collapsed_low}")
    print(f"High-level latent dims: {model.z_high_dim} | Active: {model.z_high_dim - collapsed_high} | Collapsed: {collapsed_high}")

    return {
        "kl_low_mean": kl_low_mean,
        "kl_high_mean": kl_high_mean,
        "collapsed_low": collapsed_low,
        "collapsed_high": collapsed_high,
    }
