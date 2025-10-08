"""
Latent space analysis tools for hierarchical VAE.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import os
PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
from provided.visualize import plot_drum_pattern, plot_latent_space_2d
from provided.metrics import drum_pattern_validity, sequence_diversity
import json

@torch.no_grad()
def _encode_dataset(model, data_loader, device):
    """Encode the whole loader once; return dict of arrays/tensors."""
    model.eval().to(device)
    zs_high, mus_high, logvars_high = [], [], []
    zs_low,  mus_low,  logvars_low  = [], [], []
    labels = []

    for patterns, styles, _dens in data_loader:
        x = patterns.to(device)
        # Use model's hierarchical encoder to get means/vars too
        z_low, (mu_low, logvar_low), z_high, (mu_high, logvar_high) = model.encode_hierarchy(x)

        zs_high.append(z_high.cpu())
        zs_low.append(z_low.cpu())
        mus_high.append(mu_high.cpu()); logvars_high.append(logvar_high.cpu())
        mus_low.append(mu_low.cpu());   logvars_low.append(logvar_low.cpu())
        labels.append(styles.clone())

    return {
        "z_high": torch.cat(zs_high, dim=0),
        "z_low":  torch.cat(zs_low,  dim=0),
        "mu_high": torch.cat(mus_high, dim=0),
        "mu_low":  torch.cat(mus_low,  dim=0),
        "logvar_high": torch.cat(logvars_high, dim=0),
        "logvar_low":  torch.cat(logvars_low,  dim=0),
        "labels": torch.cat(labels, dim=0).numpy(),
    }

@torch.no_grad()
def visualize_latent_hierarchy(model, data_loader, device='mps', save_dir='results/latent_analysis', max_points=3000, per_style_show=4):
    """
    Visualize the two-level latent space structure.
    
    TODO:
    1. Encode all data to get z_high and z_low
    2. Use t-SNE to visualize z_high (colored by genre)
    3. For each z_high cluster, show z_low variations
    4. Create hierarchical visualization
    """
    os.makedirs(save_dir, exist_ok=True)
    # --- 1) encode ---
    enc = _encode_dataset(model, data_loader, device)

    # --- 2) z_high t-SNE ---
    z_high = enc["mu_high"].numpy()
    labels = enc["labels"]
    if len(z_high) > max_points:
        idx = np.random.choice(len(z_high), size=max_points, replace=False)
        z_high = z_high[idx]
        labels = labels[idx]

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=0)
    z_high_2d = tsne.fit_transform(z_high)
    fig = plot_latent_space_2d(z_high_2d, labels=labels, title="z_high (style) t-SNE")
    fig_path = os.path.join(save_dir, "latent_tsne_z_high.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- 2) Per-style variations decoded from conditional prior ---
    device_t = next(model.parameters()).device
    unique_styles = np.unique(enc["labels"])
    for s in unique_styles:
        style_dir = os.path.join(save_dir, f"style_{int(s)}")
        os.makedirs(style_dir, exist_ok=True)

        mask = enc["labels"] == s
        if mask.sum() == 0:
            continue
        proto = torch.from_numpy(enc["mu_high"].numpy()[mask].mean(axis=0)).to(device_t).unsqueeze(0)

        for i in range(per_style_show):
            logits = model.decode_hierarchy(z_high=proto, z_low=None, temperature=1.0)
            patt = (torch.sigmoid(logits) > 0.5).float().squeeze(0).cpu()
            fig = plot_drum_pattern(patt, title=f"Style {int(s)} sample {i+1}")
            plt.savefig(os.path.join(style_dir, f"pattern_style{s}_sample{i+1}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

def interpolate_styles(model, pattern1, pattern2, n_steps=10, device='mps', temperature=1.0, save_dir='results/generated_patterns/interpolations'):
    """
    Interpolate between two drum patterns at both latent levels.
    
    TODO:
    1. Encode both patterns to get latents
    2. Interpolate z_high (style transition)
    3. Interpolate z_low (variation transition)
    4. Decode and visualize both paths
    5. Compare smooth vs abrupt transitions
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval().to(device)

    x1 = pattern1.unsqueeze(0).to(device)
    x2 = pattern2.unsqueeze(0).to(device)

    _, (mu_low1, _), _, (mu_high1, _) = model.encode_hierarchy(x1)
    _, (mu_low2, _), _, (mu_high2, _) = model.encode_hierarchy(x2)

    mu_low1 = mu_low1.squeeze(0);   mu_low2 = mu_low2.squeeze(0)
    mu_high1 = mu_high1.squeeze(0); mu_high2 = mu_high2.squeeze(0)
    alphas = torch.linspace(0, 1, steps=n_steps, device=device)
    # Interpolate z_high (style)
    for i, a in enumerate(alphas):
        z_high = (1 - a) * mu_high1 + a * mu_high2
        logits = model.decode_hierarchy(z_high.unsqueeze(0), z_low=mu_low1.unsqueeze(0), temperature=temperature)
        patt = (torch.sigmoid(logits) > 0.5).float().squeeze(0).cpu()
        fig = plot_drum_pattern(patt, title=f"Style interp step {i+1}/{n_steps}")
        plt.savefig(os.path.join(save_dir, f"style_interp_step_{i+1}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Interpolate z_low (variation)
    for i, a in enumerate(alphas):
        z_low = (1 - a) * mu_low1 + a * mu_low2
        logits = model.decode_hierarchy(mu_high1.unsqueeze(0), z_low=z_low.unsqueeze(0), temperature=temperature)
        patt = (torch.sigmoid(logits) > 0.5).float().squeeze(0).cpu()
        fig = plot_drum_pattern(patt, title=f"Variation interp step {i+1}/{n_steps}")
        plt.savefig(os.path.join(save_dir, f"variation_interp_step_{i+1}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

@torch.no_grad()
def measure_disentanglement(model, data_loader, device='mps', save_path='results/latent_analysis/disentanglement.json'):
    """
    Measure how well the hierarchy disentangles style from variation.
    
    TODO:
    1. Group patterns by genre
    2. Compute z_high variance within vs across genres
    3. Compute z_low variance for same genre
    4. Return disentanglement metrics
    """
    enc = _encode_dataset(model, data_loader, device)
    mu_h = enc["mu_high"].numpy()
    mu_l = enc["mu_low"].numpy()
    y    = enc["labels"]

    # Group indices per genre
    groups = {g: np.where(y == g)[0] for g in np.unique(y)}
    # z_high: within-genre variance
    within_h = []
    means_h = []
    for g, idx in groups.items():
        X = mu_h[idx]
        means_h.append(X.mean(axis=0, keepdims=True))
        if len(X) > 1:
            within_h.append(X.var(axis=0).mean())
    within_h_var = float(np.mean(within_h)) if within_h else 0.0
    means_h = np.vstack(means_h)
    between_h_var = float(means_h.var(axis=0).mean())

    within_l = []
    for g, idx in groups.items():
        X = mu_l[idx]
        if len(X) > 1:
            within_l.append(X.var(axis=0).mean())
    within_l_var = float(np.mean(within_l)) if within_l else 0.0

    eps = 1e-8
    style_separation_ratio = between_h_var / (within_h_var + eps)  
    variation_within_style = within_l_var                            

    report =  {
        "z_high_within_var": within_h_var,
        "z_high_between_var": between_h_var,
        "style_separation_ratio": style_separation_ratio,
        "z_low_within_var": variation_within_style,
    }

    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


@torch.no_grad()
def controllable_generation(model, data_loader, n_per_genre=16, temperature=1.0, device='mps', save_dir='results/generated_patterns/samples', metrics_path='results/generated_patterns/metrics.json'):
    """
    Test controllable generation using the hierarchy.
    
    TODO:
    1. Learn genre embeddings in z_high space
    2. Generate patterns with specified genre
    3. Control complexity via z_low sampling temperature
    4. Evaluate genre classification accuracy
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval().to(device)

    enc = _encode_dataset(model, data_loader, device)
    mu_h = enc["mu_high"]
    labels = enc["labels"]
    device_t = next(model.parameters()).device

    # Build prototypes in z_high
    prototypes = {}
    for g in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == g)[0]
        proto = mu_h[idx].mean(dim=0).to(device_t)
        prototypes[int(g)] = proto

    # Generate + save
    all_metrics, all_patterns = {}, []
    for s, proto in prototypes.items():
        style_dir = os.path.join(save_dir, f"style_{s}")
        os.makedirs(style_dir, exist_ok=True)

        samples = []
        for i in range(1, n_per_genre + 1):
            logits = model.decode_hierarchy(proto.unsqueeze(0), z_low=None, temperature=temperature)
            patt = (torch.sigmoid(logits) > 0.5).float().squeeze(0).cpu()
            fig = plot_drum_pattern(patt, title=f"Style {s} • sample {i}")
            plt.savefig(os.path.join(style_dir, f"style_{s}_sample_{i:02d}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
            samples.append(patt)

        batch = torch.stack(samples, dim=0)
        all_patterns.append(batch)
        all_metrics[s] = {
            "validity": float(drum_pattern_validity(batch)),
            "diversity": float(sequence_diversity(batch)),
        }

    # Overall metrics
    if all_patterns:
        all_cat = torch.cat(all_patterns, dim=0)
        overall = {
            "validity": float(drum_pattern_validity(all_cat)),
            "diversity": float(sequence_diversity(all_cat)),
        }
        all_metrics["overall"] = overall

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    return {"metrics": all_metrics}

@torch.no_grad()
def save_style_transfer_examples(
    model,
    data_loader,
    n_examples=8,
    device='mps',
    out_dir='results/generated_patterns/style_transfer'
):
    """
    Style transfer: keep z_low of a source pattern, swap z_high with a target
    style prototype. Saves images per transfer pair.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval().to(device)

    enc = _encode_dataset(model, data_loader, device)
    mu_h, labels = enc["mu_high"], enc["labels"]
    device_t = next(model.parameters()).device

    # z_high prototypes
    prototypes = {}
    for s in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == s)[0]
        prototypes[s] = mu_h[idx].mean(dim=0).to(device_t)

    # gather a few source patterns
    collected = 0
    for x, y, _ in data_loader:
        for i in range(x.size(0)):
            if collected >= n_examples: break
            src = x[i:i+1].to(device_t)
            src_style = int(y[i].item())
            _, (mu_l, _), _, (_mu_h, _) = model.encode_hierarchy(src)

            for tgt_style, proto in prototypes.items():
                logits = model.decode_hierarchy(proto.unsqueeze(0), z_low=mu_l, temperature=1.0)
                patt = (torch.sigmoid(logits) > 0.5).float().squeeze(0).cpu()
                fig = plot_drum_pattern(patt, title=f"transfer {src_style} → {tgt_style} • ex {collected+1}")
                fname = f"transfer_src{src_style}_to{tgt_style}_ex{collected+1}.png"
                plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
                plt.close(fig)

            collected += 1
            if collected >= n_examples: break
        if collected >= n_examples: break

@torch.no_grad()
def save_dimension_traversals(
    model,
    data_loader,
    device='mps',
    out_dir='results/latent_analysis/traversals',
    n_dims_high=4,
    n_dims_low=6,
    steps=7,
    width=2.5,
):
    """
    Interpret latent dimensions by varying one dim at a time (±width stds).
    Saves PNGs under results/latent_analysis/traversals/.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval().to(device)

    for x, _, _ in data_loader:
        seed = x[0:1].to(device)
        break

    _, (mu_l, logvar_l), _, (mu_h, logvar_h) = model.encode_hierarchy(seed)
    mu_l, mu_h = mu_l.squeeze(0), mu_h.squeeze(0)
    std_l, std_h = torch.exp(0.5 * logvar_l).squeeze(0), torch.exp(0.5 * logvar_h).squeeze(0)

    def _traverse(which, dim, mu, std, fixed_other, is_high):
        vals = torch.linspace(-width, width, steps, device=device) * std[dim] + mu[dim]
        for i, v in enumerate(vals, start=1):
            z_h = mu_h.clone()
            z_l = mu_l.clone()
            if is_high: z_h[dim] = v
            else:       z_l[dim] = v
            logits = model.decode_hierarchy(z_h.unsqueeze(0), z_low=z_l.unsqueeze(0), temperature=1.0)
            patt = (torch.sigmoid(logits) > 0.5).float().squeeze(0).cpu()
            fig = plot_drum_pattern(patt, title=f"{which} dim {dim} • step {i}/{steps}")
            plt.savefig(os.path.join(out_dir, f"{which}_dim{dim}_step{i:02d}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

    for d in range(min(n_dims_high, mu_h.numel())):
        _traverse("z_high", d, mu_h, std_h, mu_l, True)

    for d in range(min(n_dims_low, mu_l.numel())):
        _traverse("z_low", d, mu_l, std_l, mu_h, False)

def _auto_device(pref="auto"):
    if isinstance(pref, torch.device):
        return pref
    if pref and str(pref).lower() != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    import argparse
    from pathlib import Path
    from dataset import DrumPatternDataset
    from hierarchical_vae import HierarchicalDrumVAE
    from torch.utils.data import DataLoader

    # ---- Config dict (edit as you like) ----
    config = {
        "data_dir": "../data/drums",
        "results_dir": "results",
        "checkpoint": "results/best_model.pth",
        "z_high_dim": 4,
        "z_low_dim": 12,
        "batch_size": 64,
        "num_workers": 0,
        "device": "auto",              
        "samples_per_style": 10,
        "per_style_show": 4,
        "temperature": 1.0,
        "interp_pairs": 5,
        "interp_steps": 10,
        "traversal_steps": 7,
        "traversal_high_dims": 4,
        "traversal_low_dims": 6,
        "tsne_max_points": 3000,
    }

    device = _auto_device(config["device"])


    # --- IO setup
    base = Path(config["results_dir"])
    gen_dir = base / "generated_patterns"
    lat_dir = base / "latent_analysis"
    (gen_dir / "samples").mkdir(parents=True, exist_ok=True)
    (gen_dir / "interpolations").mkdir(parents=True, exist_ok=True)
    (gen_dir / "style_transfer").mkdir(parents=True, exist_ok=True)
    lat_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data
    train_ds = DrumPatternDataset(config["data_dir"], split="train")
    val_ds   = DrumPatternDataset(config["data_dir"], split="val")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    # ---- Model
    model = HierarchicalDrumVAE(z_high_dim=config["z_high_dim"], z_low_dim=config["z_low_dim"])
    ckpt_path = Path(config["checkpoint"])
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    # support either raw state_dict or dict with 'model_state_dict'
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
    model.to(device).eval()

    # ---------------- Generated patterns ----------------
    # 1) N samples per style + metrics
    controllable_generation(
        model,
        data_loader=train_loader,
        n_per_genre=config["samples_per_style"],
        temperature=config["temperature"],
        device=device,
        save_dir=str(gen_dir / "samples"),
        metrics_path=str(gen_dir / "metrics.json"),
    )

    # 2) Style transfer examples
    save_style_transfer_examples(
        model,
        data_loader=val_loader,
        n_examples=8,
        device=device,
        out_dir=str(gen_dir / "style_transfer"),
    )

    # 3) Interpolation sequences (choose random pairs from val set)
    import numpy as np
    rng = np.random.default_rng(0)
    if len(val_ds) >= 2:
        for k in range(1, config["interp_pairs"] + 1):
            i, j = rng.integers(0, len(val_ds), size=2)
            p1, _, _ = val_ds[i]
            p2, _, _ = val_ds[j]
            pair_dir = gen_dir / "interpolations" / f"pair_{k:02d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            interpolate_styles(
                model,
                p1, p2,
                n_steps=config["interp_steps"],
                device=device,
                temperature=config["temperature"],
                save_dir=str(pair_dir),
            )

    # ---------------- Latent analysis ----------------
    # 4) t-SNE + per-style visuals
    visualize_latent_hierarchy(
        model,
        data_loader=val_loader,
        device=device,
        save_dir=str(lat_dir),
        max_points=config["tsne_max_points"],
        per_style_show=config["per_style_show"],
    )

    # 5) Disentanglement report (JSON)
    measure_disentanglement(
        model,
        data_loader=val_loader,
        device=device,
        save_path=str(lat_dir / "disentanglement.json"),
    )

    # 6) Dimension traversals (saved PNGs)
    save_dimension_traversals(
        model,
        data_loader=val_loader,
        device=device,
        out_dir=str(lat_dir / "traversals"),
        n_dims_high=config["traversal_high_dims"],
        n_dims_low=config["traversal_low_dims"],
        steps=config["traversal_steps"],
    )

    print(f"All artifacts written under: {base.resolve()}")


if __name__ == "__main__":
    main()