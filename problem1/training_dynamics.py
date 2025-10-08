"""
GAN training implementation with mode collapse analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os, sys
PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
from provided.metrics import mode_coverage_score

def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Create one-hot from integer labels on the same device."""
    b = labels.size(0)
    out = torch.zeros(b, num_classes, device=labels.device, dtype=torch.float32)
    out.scatter_(1, labels.view(-1, 1), 1.0)
    return out


def _sample_noise(batch_size: int, z_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, z_dim, device=device)


def _maybe_to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """Assumes x in [0,1]; map to [-1,1] to match Tanh generator output."""
    return x * 2.0 - 1.0


def train_gan(generator, discriminator, data_loader, num_epochs=100, device='mps', checkpoint_dir=None):
    """
    Standard GAN training implementation.
    
    Uses vanilla GAN objective which typically exhibits mode collapse.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        data_loader: DataLoader for training data
        num_epochs: Number of training epochs
        device: Device for computation
        
    Returns:
        dict: Training history and metrics
    """
    device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()

    # Initialize optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training history
    history = defaultdict(list)
    z_dim = getattr(generator, 'z_dim', 100)
    g_conditional = getattr(generator, 'conditional', False)
    d_conditional = getattr(discriminator, 'conditional', False)
    num_classes = getattr(generator, 'num_classes', 26)
    
    for epoch in range(num_epochs):
        for batch_idx, (real_images, labels) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device).long() 
            
            # Labels for loss computation
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ========== Train Discriminator ==========
            # TODO: Implement discriminator training step
            # 1. Zero gradients
            d_optimizer.zero_grad(set_to_none=True)

            # 2. Forward pass on real image
            real_images_scaled = _maybe_to_minus1_1(real_images)

            if d_conditional:
                labels_oh = _one_hot(labels, num_classes)
                d_out_real = discriminator(real_images_scaled, class_label=labels_oh)
            else:
                d_out_real = discriminator(real_images_scaled)

            # 3. Compute real loss
            d_loss_real = criterion(d_out_real, real_labels)

            # 4. Generate fake images from random z
            z = _sample_noise(batch_size, z_dim, device)
            if g_conditional:
                g_labels_oh = _one_hot(labels, num_classes)
                fake_images = generator(z, class_label=g_labels_oh)
            else:
                fake_images = generator(z)

            # 5. Forward pass on fake images (detached)
            if d_conditional:
                d_out_fake = discriminator(fake_images.detach(), class_label=g_labels_oh if g_conditional else None)
            else:
                d_out_fake = discriminator(fake_images.detach())

            # 6. Compute fake loss
            d_loss_fake = criterion(d_out_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            # 7. Backward and optimize
            d_loss.backward()
            d_optimizer.step()
            
            # ========== Train Generator ==========
            # TODO: Implement generator training step
            # 1. Zero gradients
            g_optimizer.zero_grad(set_to_none=True)

            # 2. Generate fake images
            z = _sample_noise(batch_size, z_dim, device)
            if g_conditional:
                g_labels_oh = _one_hot(labels, num_classes)
                fake_images = generator(z, class_label=g_labels_oh)
                d_out_fake_for_g = discriminator(fake_images, class_label=g_labels_oh if d_conditional else None)
            else:
                fake_images = generator(z)
                d_out_fake_for_g = discriminator(fake_images)

            # 3. Forward pass through discriminator

            # 4. Compute adversarial loss
            g_loss = criterion(d_out_fake_for_g, real_labels)

            # 5. Backward and optimize

            g_loss.backward()
            g_optimizer.step()

            # Log metrics
            if batch_idx % 10 == 0:
                history['d_loss'].append(d_loss.item())
                history['g_loss'].append(g_loss.item())
                history['epoch'].append(epoch + batch_idx/len(data_loader))
        
        # Analyze mode collapse every 10 epochs
        if epoch % 10 == 0:
            mode_coverage = analyze_mode_coverage(generator, device)
            history['coverage_score'].append(mode_coverage['coverage_score'])
            history['letter_counts'].append(mode_coverage['letter_counts'])
            history['missing_letters'].append(mode_coverage['missing_letters'])
            history['mode_cov_epoch'].append(epoch)
            print(f"Epoch {epoch}: coverage={mode_coverage['coverage_score']:.2f} "
          f"unique={mode_coverage['n_unique']} missing={mode_coverage['missing_letters']}")
            
        if (epoch+1) in {10,30,50,100}:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1:04d}.pth")
            torch.save(generator.state_dict(), ckpt_path)
    
    return history

def analyze_mode_coverage(generator, device, n_samples=1000):
    """
    Measure mode coverage by counting unique letters in generated samples.
    
    Args:
        generator: Trained generator network
        device: Device for computation
        n_samples: Number of samples to generate
        
    Returns:
        float: Coverage score (unique letters / 26)
    """
    # TODO: Generate n_samples images
    generator.eval()
    samples = []

    z_dim = getattr(generator, 'z_dim', 100)
    conditional = getattr(generator, 'conditional', False)
    num_classes = getattr(generator, 'num_classes', 26)

    with torch.no_grad():
        remaining = n_samples
        while remaining > 0:
            b = min(256, remaining)
            z = torch.randn(b, z_dim, device=device)

            if conditional:
                y = torch.randint(0, num_classes, (b,), device=device)
                y_oh = torch.zeros(b, num_classes, device=device, dtype=torch.float32)
                y_oh.scatter_(1, y.view(-1, 1), 1.0)
                imgs = generator(z, class_label=y_oh)
            else:
                imgs = generator(z)
            
            imgs01 = (imgs + 1.0) / 2.0
            imgs01 = imgs01.clamp_(0.0, 1.0).cpu()

            samples.extend([imgs01[i] for i in range(imgs01.size(0))])
            remaining -= b

    generator.train()

    # Use provided letter classifier to identify generated letters
    all_imgs = torch.stack(samples, dim=0) # [N,1,28,28]
    stats = mode_coverage_score(all_imgs, classifier_fn=None)

    # Count unique letters produced
    # Return coverage score (0 to 1)
    return stats

def visualize_mode_collapse(history, save_path):
    """
    Visualize mode collapse progression over training.
    
    Args:
        history: Training metrics dictionary
        save_path: Output path for visualization
    """

    cov = history.get('coverage_score', history.get('mode_coverage', []))
    if not cov:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, 'No coverage data in history.', ha='center', va='center')
        plt.axis('off')
        plt.savefig(save_path, dpi=150)
        plt.close()
        return

    epochs = history.get('mode_cov_epoch', list(range(len(cov))))

    last_counts = None
    if 'letter_counts' in history and len(history['letter_counts']) > 0:
        last_counts = history['letter_counts'][-1]   # dict: int->count
    last_missing = set()
    if 'missing_letters' in history and len(history['missing_letters']) > 0:
        last_missing = set(int(i) for i in history['missing_letters'][-1])

    counts = np.zeros(26, dtype=np.int32)
    if isinstance(last_counts, dict):
        for k, v in last_counts.items():
            k = int(k)
            if 0 <= k < 26:
                counts[k] = int(v)

    letters = [chr(65 + i) for i in range(26)]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 3]})

    # TODO: Plot mode coverage over time
    axes[0].plot(epochs, cov, marker='o')
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Coverage (unique letters / 26)')
    axes[0].set_title('Mode Coverage Over Time')

    # Show which letters survive and which disappear
    bars = axes[1].bar(letters, counts)
    # highlight missing letters
    for i in range(26):
        if i in last_missing:
            bars[i].set_alpha(0.3)
    axes[1].set_ylabel('Count in last probe')
    axes[1].set_title('Letter Frequency (missing letters are faded)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()