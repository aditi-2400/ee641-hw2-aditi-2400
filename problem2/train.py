"""
Main training script for hierarchical VAE experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from pathlib import Path
import numpy as np

from dataset import DrumPatternDataset
from hierarchical_vae import HierarchicalDrumVAE
from training_utils import kl_anneal_schedule, temp_anneal_schedule

def train_epoch(model, data_loader, optimizer, epoch, device, config):
    """
    Train model for one epoch with annealing schedules.
    
    Returns:
        Dictionary of average metrics for the epoch
    """
    model.train()
    
    # Metrics tracking
    metrics = {
        'total_loss': 0,
        'recon_loss': 0,
        'kl_low': 0,
        'kl_high': 0
    }
    
    # Get annealing parameters for this epoch
    beta = kl_anneal_schedule(epoch, method=config['kl_anneal_method'])
    temperature = temp_anneal_schedule(epoch)
    
    n_seen = 0
    for batch_idx, (patterns, styles, densities) in enumerate(data_loader):
        patterns = patterns.to(device)
        optimizer.zero_grad()
        # Forward pass
        out = model(patterns, beta=beta, temperature=temperature)
        
        # Compute loss
        losses = out["losses"]
        total = losses["total"]
        
        # Backward pass
        total.backward()
        optimizer.step()

        bsz = patterns.size(0)
        n_seen += bsz
        # Update metrics
        metrics['total_loss'] += total.item()
        metrics['recon_loss'] += losses["recon"].item()
        metrics['kl_low'] += losses["kl_low"].item()
        metrics['kl_high'] += losses["kl_high"].item()
        
        # Log progress
        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch:3d} [{batch_idx:3d}/{len(data_loader)}] "
                f"Loss: {total.item()/bsz:.4f} "
                f"Beta: {beta:.3f} Temp: {temperature:.2f}"
            )
    
    # Average metrics
    num_batches = max(1, len(data_loader))
    for k in metrics:
        metrics[k] /= num_batches
    
    return metrics

@torch.no_grad()
def validate_epoch(model, data_loader, device, epoch):
    """
    Validation pass. Uses the model's internal ELBO terms.
    """
    model.eval()
    metrics = {
        "total_loss": 0.0,
        "recon_loss": 0.0,
        "kl_low": 0.0,
        "kl_high": 0.0,
    }

    for patterns, styles, densities in data_loader:
        patterns = patterns.to(device)
        out = model(patterns)  
        losses = out["losses"]

        metrics["total_loss"] += losses["total"].item()
        metrics["recon_loss"] += losses["recon"].item()
        metrics["kl_low"] += losses["kl_low"].item()
        metrics["kl_high"] += losses["kl_high"].item()

    num_batches = max(1, len(data_loader))
    for k in metrics:
        metrics[k] /= num_batches

    print(
        f"Epoch {epoch:3d} Validation - "
        f"Loss: {metrics['total_loss']:.4f} "
        f"KL_high: {metrics['kl_high']:.4f} "
        f"KL_low: {metrics['kl_low']:.4f}"
    )
    return metrics

def main():
    """
    Main training entry point for hierarchical VAE experiments.
    """
    # Configuration
    config = {
        'device': torch.device('mps' if torch.backends.mps.is_available() else 'cpu'),
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'z_high_dim': 4,
        'z_low_dim': 12,
        'kl_anneal_method': 'sigmoid',  # 'linear', 'cyclical', or 'sigmoid'
        'data_dir': '../data/drums',
        'checkpoint_dir': 'checkpoints',
        'results_dir': 'results'
    }
    
    # Create directories
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset and dataloader
    train_dataset = DrumPatternDataset(config['data_dir'], split='train')
    val_dataset = DrumPatternDataset(config['data_dir'], split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model and optimizer
    model = HierarchicalDrumVAE(
        z_high_dim=config['z_high_dim'],
        z_low_dim=config['z_low_dim']
    ).to(config['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training history
    history = {
        'train': [],
        'val': [],
        'config': config
    }
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, epoch, 
            config['device'], config
        )
        history['train'].append(train_metrics)
        
        # Validate every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            val_metrics = validate_epoch(model, val_loader, config["device"], epoch)
            history["val"].append(val_metrics)
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch+1}.pth")
    
    # Save final model and history
    torch.save(model.state_dict(), f"{config['results_dir']}/best_model.pth")
    
    with open(f"{config['results_dir']}/training_log.json", 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    print(f"Training complete. Results saved to {config['results_dir']}/")

if __name__ == '__main__':
    main()