"""
Dataset loader for drum pattern generation task.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
DEFAULT_INSTRS = ['kick','snare','hihat_closed','hihat_open','tom_low','tom_high','crash','ride','clap']
DEFAULT_STYLES = ['rock','jazz','hiphop','electronic','latin']

class DrumPatternDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Initialize drum pattern dataset.
        
        Args:
            data_dir: Path to drum dataset directory
            split: 'train' or 'val'
        """
        self.data_dir = data_dir
        self.split = split
        # Load patterns from drum_patterns.npz
        data_path = os.path.join(data_dir, 'patterns.npz')
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"patterns.npz not found in {data_dir}")
        data = np.load(data_path, allow_pickle=True)
        
        if split == 'train':
            if not all(k in data.files for k in ['train_patterns', 'train_styles']):
                raise KeyError(f"Expected 'train_patterns' and 'train_styles' in {data.files}")
            patterns = data['train_patterns']
            styles   = data['train_styles']
        elif split == 'val':
            if not all(k in data.files for k in ['val_patterns', 'val_styles']):
                raise KeyError(f"Expected 'val_patterns' and 'val_styles' in {data.files}")
            patterns = data['val_patterns']
            styles   = data['val_styles']
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split}")
        
        # Load metadata
        json_meta = os.path.join(data_dir, 'patterns.json')
        if os.path.isfile(json_meta):
            with open(json_meta, 'r') as f:
                meta = json.load(f)
            self.instrument_names = meta['instruments']
            self.style_names = meta['styles']
            self.timesteps  = int(meta.get('timesteps', 16))

        elif 'metadata' in data.files:
            meta = data['metadata'].item()
            self.instrument_names = meta.get('instruments', DEFAULT_INSTRS)
            self.style_names      = meta.get('styles',      DEFAULT_STYLES)
            self.timesteps        = int(meta.get('timesteps', 16))
        else:
            self.instrument_names = DEFAULT_INSTRS
            self.style_names      = DEFAULT_STYLES
            self.timesteps        = 16

        self.patterns = patterns.astype(np.float32)
        self.styles   = styles.astype(np.int64)
    
    def __len__(self):
        return len(self.patterns)
    
    def __getitem__(self, idx):
        """
        Return a drum pattern sample.
        
        Returns:
            pattern: Binary tensor of shape [16, 9]
            style: Integer style label (0-4)
            density: Float indicating pattern density (for analysis)
        """
        pattern = self.patterns[idx]
        style = int(self.styles[idx])
        
        # Convert to tensor
        pattern_tensor = torch.from_numpy(pattern).float()
        
        # Compute density metric (fraction of active hits)
        density = float(pattern_tensor.sum().item()) / (self.timesteps * len(self.instrument_names))
        
        return pattern_tensor, style, density
    
    def pattern_to_pianoroll(self, pattern):
        """
        Convert pattern to visual piano roll representation.
        
        Args:
            pattern: Binary array [16, 9] or tensor
            
        Returns:
            pianoroll: Visual representation for plotting
        """
        if torch.is_tensor(pattern):
            pattern = pattern.cpu().numpy()
        
        # Create visual representation with instrument labels
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each active hit
        for t in range(16):
            for i in range(9):
                if pattern[t, i] > 0.5:
                    rect = patches.Rectangle((t, i), 1, 1, 
                                            linewidth=1, 
                                            edgecolor='black',
                                            facecolor='blue')
                    ax.add_patch(rect)
        
        # Add grid
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.set_xticks(range(17))
        ax.set_yticks(range(10))
        ax.set_yticklabels([''] + self.instrument_names)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Instrument')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        
        return fig