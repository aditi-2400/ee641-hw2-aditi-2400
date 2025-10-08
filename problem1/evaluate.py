"""
Analysis and evaluation experiments for trained GAN models.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
from provided.metrics import mode_coverage_score
from provided.visualize import plot_alphabet_grid
from training_dynamics import analyze_mode_coverage
import re
from models import Generator
import glob


def _one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    oh = torch.zeros(y.size(0), num_classes, device=y.device, dtype=torch.float32)
    oh.scatter_(1, y.view(-1, 1), 1.0)
    return oh

def _to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp_(0.0, 1.0)

def _classify_single(img01: torch.Tensor) -> int:
    """
    Classify a single image tensor [1,28,28] in [0,1] to a letter index 0..25,
    using the provided simple classifier via mode_coverage_score on a batch of size 1.
    """
    if mode_coverage_score is None:
        raise RuntimeError("mode_coverage_score is not available; cannot classify letters.")
    batch = img01.unsqueeze(0)  # [1,1,28,28]
    stats = mode_coverage_score(batch, classifier_fn=None)
    (k, _) = next(iter(stats['letter_counts'].items()))
    return int(k)

def _tile_gray(images01: torch.Tensor, rows: int, cols: int) -> np.ndarray:
    """
    images01: [N,1,H,W] in [0,1]
    returns a H*rows by W*cols uint8 mosaic
    """
    n = images01.shape[0]
    h, w = images01.shape[-2:]
    canvas = np.zeros((rows * h, cols * w), dtype=np.float32)
    for i in range(min(n, rows * cols)):
        r, c = divmod(i, cols)
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = images01[i, 0].cpu().numpy()
    return (canvas * 255.0).clip(0, 255).astype(np.uint8)



def interpolation_experiment(generator, device, steps_per_pair: int = 8, samples_per_letter: int = 256,
                             save_path: str = None):
    """
    Interpolate between latent codes to generate smooth transitions.
    
    TODO:
    1. Find latent codes for specific letters (via optimization)
    2. Interpolate between them
    3. Visualize the path from A to Z
    """
    
    """
    Interpolate between latent codes to generate smooth transitions.

    Strategy:
    - If generator.conditional:
        * Sample one latent z_i per letter i (0..25), and linearly interpolate BOTH z and one-hot y.
          Soft labels during interpolation are okay at inference.
    - If generator is unconditional:
        * Sample many z, keep the first z_i that generates letter i (via the provided heuristic classifier),
          then interpolate z_i → z_{i+1}.

    Args:
        generator: trained nn.Module
        device: torch.device
        steps_per_pair: interpolation steps between successive letters (A→B, B→C, ...)
        samples_per_letter: (unconditional only) how many z to try per letter to find a match
        save_path: optional path to save a big interpolation grid (PNG)

    Returns:
        dict with:
          - 'grid': uint8 image mosaic
          - 'rows': number of rows
          - 'cols': number of cols
    """
    model_device = next(generator.parameters()).device
    device = model_device 
    generator.eval()
    z_dim = getattr(generator, 'z_dim', 100)
    conditional = getattr(generator, 'conditional', False)
    num_classes = getattr(generator, 'num_classes', 26)

    with torch.no_grad():
        # Collect latent anchors z_0..z_25 for letters A..Z (0..25)
        z_list = []
        if conditional:
            # Just sample one z per letter (label enforces letter identity)
            for _ in range(26):
                z_list.append(torch.randn(1, z_dim, device=device))
        else:
            if mode_coverage_score is None:
                raise RuntimeError("Unconditional interpolation needs the classifier (mode_coverage_score).")
            for target in range(26):
                found = None
                trials = 0
                while trials < samples_per_letter and found is None:
                    z = torch.randn(32, z_dim, device=device)  
                    imgs = generator(z)                         # [B,1,28,28] in [-1,1]
                    imgs01 = _to_01(imgs)
                    for i in range(z.size(0)):
                        pred = _classify_single(imgs01[i])      
                        if pred == target:
                            found = z[i:i+1]
                            break
                    trials += z.size(0)
                if found is None:
                    found = torch.randn(1, z_dim, device=device)
                z_list.append(found)

        rows = 25                          # pairs: A→B, B→C, ..., Y→Z
        cols = steps_per_pair
        all_imgs = []

        for i in range(25):
            z0 = z_list[i]
            z1 = z_list[i + 1]
            for s in range(steps_per_pair):
                t = s / max(1, steps_per_pair - 1)
                zt = (1 - t) * z0 + t * z1  

                if conditional:
                    y0 = torch.zeros(1, num_classes, device=device); y0[0, i] = 1.0
                    y1 = torch.zeros(1, num_classes, device=device); y1[0, i + 1] = 1.0
                    yt = (1 - t) * y0 + t * y1
                    img = generator(zt, class_label=yt)
                else:
                    img = generator(zt)
                all_imgs.append(_to_01(img))

        all_imgs = torch.cat(all_imgs, dim=0)
        grid = _tile_gray(all_imgs, rows=rows, cols=cols)

    generator.train()

    if save_path is not None:
        plt.figure(figsize=(cols * 0.6, rows * 0.6))
        plt.imshow(grid, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        if not os.path.splitext(save_path)[1]:
            save_path = save_path + ".png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()

    return {'grid': grid, 'rows': rows, 'cols': cols}

def style_consistency_experiment(conditional_generator, device='mps', bins: int = 16, save_path: str = None):
    """
    Test if conditional GAN maintains style across letters.
    
    TODO:
    1. Fix a latent code z
    2. Generate all 26 letters with same z
    3. Measure style consistency
    """
    model_device = next(conditional_generator.parameters()).device
    device = model_device
    G = conditional_generator
    assert getattr(G, 'conditional', False), "style_consistency_experiment requires a conditional generator"

    G.eval()
    z_dim = getattr(G, 'z_dim', 100)
    num_classes = getattr(G, 'num_classes', 26)

    with torch.no_grad():
        z = torch.randn(1, z_dim, device=device)
        imgs = []
        for cls in range(26):
            y = torch.zeros(1, num_classes, device=device)
            y[0, cls] = 1.0
            x = G(z, class_label=y)         
            imgs.append(_to_01(x))          
        images01 = torch.cat(imgs, dim=0)   

    # 16-bin intensity hist per letter
    histos = []
    for i in range(26):
        arr = images01[i, 0].cpu().numpy().ravel()  
        h, _ = np.histogram(arr, bins=bins, range=(0.0, 1.0), density=False)
        h = h.astype(np.float32)
        h = h / (np.linalg.norm(h) + 1e-8)  
        histos.append(h)
    H = np.stack(histos, axis=0) 

    C = H @ H.T  
    score = (np.sum(C) - np.trace(C)) / (C.shape[0] * (C.shape[0] - 1))

    # Save grid if asked
    if save_path is not None:
        grid = _tile_gray(images01, rows=2, cols=13)  # 2x13 grid = 26 letters
        plt.figure(figsize=(13 * 0.6, 2 * 0.6))
        plt.imshow(grid, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        if not os.path.splitext(save_path)[1]:
            save_path = save_path + ".png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()

    G.train()
    return {
        'score': float(score),
        'images01': images01,
        'histograms': H,
        'cosine_matrix': C,
    }

def mode_recovery_experiment(generator_checkpoints, n_samples, device='mps', save_path = None):
    """
    Analyze how mode collapse progresses and potentially recovers.
    
    TODO:
    1. Load checkpoints from different epochs
    2. Measure mode coverage at each checkpoint
    3. Identify when specific letters disappear/reappear
    """
    from models import Generator

    results_epochs = []
    results_cov = []
    results_missing = []

    # Try to extract epoch numbers from filenames like "...epoch_030.pth"
    def _infer_epoch(path: str) -> int:
        m = re.search(r'(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else len(results_epochs)

    # Probe each checkpoint
    for ckpt_path in generator_checkpoints:
        # Build fresh generator (adjust if your model is conditional)
        G = Generator(z_dim=100, conditional=getattr(Generator(), 'conditional', False))
        sd = torch.load(ckpt_path, map_location='cpu')
        # Accept either raw state_dict or a dict with a key
        if isinstance(sd, dict) and 'state_dict' in sd:
            G.load_state_dict(sd['state_dict'])
        elif isinstance(sd, dict) and 'generator_state_dict' in sd:
            G.load_state_dict(sd['generator_state_dict'])
        else:
            G.load_state_dict(sd)
        G.to(device)

        # Measure coverage (prefer your function if available)
        if analyze_mode_coverage is not None:
            stats = analyze_mode_coverage(G, device, n_samples=n_samples)
            cov = float(stats.get('coverage_score', stats.get('coverage', 0.0)))
            missing = stats.get('missing_letters', [])
        else:
            if mode_coverage_score is None:
                raise RuntimeError("Need either analyze_mode_coverage or mode_coverage_score.")
            # fallback: generate and evaluate here
            G.eval()
            batches = []
            z_dim = getattr(G, 'z_dim', 100)
            with torch.no_grad():
                remaining = n_samples
                while remaining > 0:
                    b = min(256, remaining)
                    z = torch.randn(b, z_dim, device=device)
                    x = G(z)
                    x01 = _to_01(x).cpu()
                    batches.append(x01)
                    remaining -= b
            all_imgs = torch.cat(batches, dim=0)
            stats = mode_coverage_score(all_imgs, classifier_fn=None)
            cov = float(stats['coverage_score'])
            missing = stats['missing_letters']
            G.train()

        results_epochs.append(_infer_epoch(ckpt_path))
        results_cov.append(cov)
        results_missing.append(missing)

    # Build presence heatmap [T, 26]
    T = len(results_epochs)
    presence = np.ones((T, 26), dtype=np.float32)
    for t in range(T):
        for mi in results_missing[t]:
            presence[t, int(mi)] = 0.0

    # Plot heatmap if asked
    if save_path is not None:
        order = np.argsort(results_epochs)
        ep_sorted = np.array(results_epochs)[order]
        pres_sorted = presence[order]

        plt.figure(figsize=(10, 4))
        plt.imshow(pres_sorted.T, aspect='auto', cmap='Greys', vmin=0.0, vmax=1.0)
        plt.yticks(range(26), [chr(65+i) for i in range(26)])
        plt.xticks(range(T), ep_sorted, rotation=45)
        plt.xlabel('Epoch / Checkpoint')
        plt.title('Letter Presence Across Checkpoints (1=present, 0=missing)')
        plt.tight_layout()
        if not os.path.splitext(save_path)[1]:
            save_path = save_path + ".png"
        plt.savefig(save_path, dpi=150)
        plt.close()

    return {
        'epochs': results_epochs,
        'coverage_score': results_cov,
        'missing_letters': results_missing,
        'presence': presence,
    }

def main():
    results_dir = "results"
    viz_dir = os.path.join(results_dir, "visualizations")
    best_path = os.path.join(results_dir, "best_generator.pth")
    conditional = getattr(Generator(), "conditional", False)
    num_classes = getattr(Generator(), "num_classes", 26)
    z_dim = getattr(Generator(), "z_dim", 100) or 100
    device = (torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
              else torch.device('cpu'))
    G = Generator(z_dim=z_dim, conditional=conditional, num_classes=num_classes)
    sd = torch.load(best_path, map_location="cpu")
    G.load_state_dict(sd["generator_state_dict"])
    G.to(device).eval()
    interpolation_experiment(G, device, steps_per_pair=8,
                                 save_path=os.path.join(viz_dir, "interpolation_fix.png"))
    ckpts = sorted(glob.glob("checkpoints/epoch_*.pth"))
    out = mode_recovery_experiment(ckpts, device=device,
                               n_samples=1000,
                               save_path="results/visualizations/mode_recovery_heatmap.png")
    if getattr(G, 'conditional', False):
        style_consistency_experiment(G, device, save_path=os.path.join(viz_dir, "style_consistency.png"))

    fig = plot_alphabet_grid(G, device=device, z_dim=100, seed=42)
    fig.savefig("results/visualizations/alphabet_grid_best.png", dpi=150, bbox_inches='tight')

if __name__ == '__main__':
    main()