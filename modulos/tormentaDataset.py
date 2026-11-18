import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class TormentaDataset(Dataset):
    def __init__(self, split_dir, stats_path):
        self.files = sorted(Path(split_dir).glob("*.npz"))
        stats = np.load(stats_path)
        # Shape (3,1,1) para que el broadcast funcione sobre (3,297,297)
        self.min_X = torch.tensor(stats["min_X"], dtype=torch.float32).view(-1, 1, 1)
        self.max_X = torch.tensor(stats["max_X"], dtype=torch.float32).view(-1, 1, 1)
        self.min_y = float(stats["min_y"])
        self.max_y = float(stats["max_y"])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        X = torch.tensor(data["X"], dtype=torch.float32)
        y = torch.tensor(data["y"], dtype=torch.float32).unsqueeze(0)

        # Normalización Min-Max al rango [0, 1] usando los extremos globales
        X = (X - self.min_X) / (self.max_X - self.min_X + 1e-8)  # 1e-8 evita división por cero

        # Normalización logarítmica
        y_log = torch.log(y + 0.1)

        # [0,1] para la sigmoide (SOLO si se va a usar FACL como loss)
        y = (y_log - self.min_y) / (self.max_y - self.min_y + 1e-8)

        return X, y

    def desnormalizar_y(self, y_norm):
        """Inversa completa: [0,1] → log(y+0.1) → LIGHTDENS real."""
        y_log = y_norm * (self.max_y - self.min_y) + self.min_y
        return torch.clamp(torch.exp(y_log) - 0.1, min=0)

    def __repr__(self):
        return f'Dataset con {len(self.files)} archivos'
