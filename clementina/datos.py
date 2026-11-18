"""Carga de datasets. Cada worker importa este modulo una vez."""
import os
from modulos.tormentaDataset import TormentaDataset

DATASET_DIR = os.environ.get("DATASET_DIR", os.path.expanduser("~/data/dataset_Camilo"))

train_dataset = TormentaDataset(split_dir=os.path.join(DATASET_DIR, "train"),
                                stats_path=os.path.join(DATASET_DIR, "stats.npz"))
val_dataset   = TormentaDataset(split_dir=os.path.join(DATASET_DIR, "val"),
                                stats_path=os.path.join(DATASET_DIR, "stats.npz"))
print(f"[datos] Train: {len(train_dataset)} muestras | Val: {len(val_dataset)} muestras", flush=True)

_n = int(os.environ.get("SUBSET", 0))
if _n:
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, range(_n))
    val_dataset   = Subset(val_dataset, range(_n // 4))
