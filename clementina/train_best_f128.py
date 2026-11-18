import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from modelo import UNet_WRF, FACL
from datos import train_dataset, val_dataset

SEED = 42
device = torch.device("xpu")

n_niveles = 3
n_filtros_base = 128
dropout_ps = [0.03, 0.08, 0.26]
kernel_sizes = [5, 5, 3]

EPOCAS = 50
BATCH_SIZE = 16
LR = 2.5e-5
ALPHA = 0.1

def main():
    print(f"Iniciando entrenamiento final con n_filtros_base={n_filtros_base}...")
    tiempo_inicio = time.time()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, generator=generator,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=False)

    modelo = UNet_WRF(
        in_channels=3,
        n_niveles=n_niveles,
        n_filtros_base=n_filtros_base,
        kernel_sizes=kernel_sizes,
        dropout_ps=dropout_ps
    ).to(device)
    
    n_parametros = sum(p.numel() for p in modelo.parameters())
    print(f"Arquitectura: {n_niveles} niveles, kernels {kernel_sizes}, dropouts {dropout_ps}")
    print(f"Cantidad de parámetros: {n_parametros:,}")

    criterion = FACL(total_steps=EPOCAS * len(train_loader), alpha=ALPHA).to(device)
    optimizer = optim.Adam(modelo.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCAS, eta_min=1e-7)

    mejor_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)
    ruta_modelo = "checkpoints/best_model_f128.pth"

    for epoch in range(EPOCAS):
        modelo.train()
        train_loss_acc = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = modelo(batch_X)
            loss = criterion(pred, batch_y, mode='train')
            loss.backward()
            optimizer.step()
            train_loss_acc += loss.item()

        train_loss_avg = train_loss_acc / len(train_loader)

        modelo.eval()
        criterion.val_reset()
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                criterion.val_acumular(modelo(batch_X), batch_y)
        
        val_loss = criterion.val_resultado()
        scheduler.step()

        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            torch.save(modelo.state_dict(), ruta_modelo)
            marca_mejor = " (¡Nuevo mejor! Guardado)"
        else:
            marca_mejor = ""

        print(f"Epoch {epoch+1:02d}/{EPOCAS} | Train Loss: {train_loss_avg:.4f} | "
              f"Val Loss: {val_loss:.4f} (FAL: {criterion.val_fal:.5f} FCL: {criterion.val_fcl:.4f}){marca_mejor}")

    tiempo_total = (time.time() - tiempo_inicio) / 3600
    print(f"\nEntrenamiento finalizado en {tiempo_total:.2f} horas.")
    print(f"Mejor Val Loss: {mejor_val_loss:.4f}")
    print(f"Modelo guardado exitosamente en: {ruta_modelo}")

if __name__ == '__main__':
    main()
