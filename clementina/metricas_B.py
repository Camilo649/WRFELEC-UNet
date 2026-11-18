"""CSI y FSS de validación por época para la Etapa B, con las MISMAS definiciones que el notebook de resultados
(Entrenamiento_MemoFatShortUnetFACL.ipynb, celda de funciones), pero acumulando por lotes en el dispositivo:

  - LIGHTDENS real = round(dataset.desnormalizar_y(.)), igual para el objetivo y para la predicción.
  - CSI (calcular_metricas_timestep + graficar_csi): evento si LIGHTDENS > 0. Por timestep,
    CSI = hits / (hits + misses + falsas alarmas) sobre los píxeles con actividad observada o predicha;
    si no hay actividad en ninguno de los dos el timestep es NaN.
      csi_media    = promedio de los CSI por timestep ignorando NaN (el "Overall Mean" de graficar_csi)
      csi_agregado = sum(hits) / sum(hits + misses + falsas alarmas) sobre todo el set
  - FSS (calcular_matriz_fss_dataset): umbrales (1, 5, 15, 30), ventanas (1, 3, 5, 15); fracciones con
    avg_pool2d(kernel=v, stride=1, padding=v//2) sobre el campo completo; el error se suma solo en los píxeles
    con actividad observada o predicha; se acumula sobre todo el set y se divide al final (den == 0 -> 1).
    Columna fss_u{umbral}_v{ventana}.
"""
import math
import torch
import torch.nn.functional as F

UMBRALES_FSS = (1, 5, 15, 30)
VENTANAS_FSS = (1, 3, 5, 15)


def columnas_metricas(umbrales=UMBRALES_FSS, ventanas=VENTANAS_FSS):
    return (["csi_media", "csi_agregado", "n_timesteps_csi"]
            + [f"fss_u{u}_v{v}" for v in ventanas for u in umbrales])


class AcumuladorMetricas:
    def __init__(self, desnormalizar_y, device, umbrales=UMBRALES_FSS, ventanas=VENTANAS_FSS):
        assert all(v % 2 == 1 for v in ventanas), "Las ventanas deben ser impares (padding simétrico)."
        self.desnormalizar_y = desnormalizar_y
        self.umbrales, self.ventanas = tuple(umbrales), tuple(ventanas)
        U, V = len(self.umbrales), len(self.ventanas)
        self.umb = torch.tensor(self.umbrales, device=device, dtype=torch.float32).view(1, U, 1, 1)
        self.num = torch.zeros(V, U, device=device, dtype=torch.float64)
        self.den = torch.zeros(V, U, device=device, dtype=torch.float64)
        self.suma_csi = torch.zeros((), device=device, dtype=torch.float64)    # suma de los CSI por timestep no NaN
        self.n_csi = torch.zeros((), device=device, dtype=torch.float64)       # cantidad de timesteps no NaN
        self.hits = torch.zeros((), device=device, dtype=torch.float64)
        self.total = torch.zeros((), device=device, dtype=torch.float64)       # hits + misses + falsas alarmas

    @torch.no_grad()
    def acumular(self, pred, y):
        """pred, y: valores normalizados, forma (B, 1, H, W)."""
        if pred.shape != y.shape:
            raise ValueError(f"forma de la predicción {tuple(pred.shape)} != forma del objetivo {tuple(y.shape)}")
        y_real = torch.round(self.desnormalizar_y(y))
        y_pred = torch.round(self.desnormalizar_y(pred))
        obs, prd = y_real > 0, y_pred > 0

        # --- CSI por timestep ---
        dims = tuple(range(1, y.dim()))
        hits = (obs & prd).sum(dim=dims).double()
        total = (obs | prd).sum(dim=dims).double()          # cada píxel con actividad es hit, miss o falsa alarma
        hay = total > 0
        self.suma_csi += torch.where(hay, hits / total.clamp(min=1), torch.zeros_like(hits)).sum()
        self.n_csi += hay.double().sum()
        self.hits += hits.sum()
        self.total += total.sum()

        # --- FSS ---
        mask = (obs | prd).float()
        obs_bin = (y_real >= self.umb).float()
        pred_bin = (y_pred >= self.umb).float()
        for i, v in enumerate(self.ventanas):
            obs_frac = F.avg_pool2d(obs_bin, kernel_size=v, stride=1, padding=v // 2)
            pred_frac = F.avg_pool2d(pred_bin, kernel_size=v, stride=1, padding=v // 2)
            self.num[i] += (((obs_frac - pred_frac) ** 2) * mask).sum(dim=(0, 2, 3)).double()
            self.den[i] += ((obs_frac ** 2 + pred_frac ** 2) * mask).sum(dim=(0, 2, 3)).double()

    def resultados(self):
        n_csi = float(self.n_csi.item())
        total = float(self.total.item())
        res = {
            "csi_media": float(self.suma_csi.item()) / n_csi if n_csi > 0 else math.nan,
            "csi_agregado": float(self.hits.item()) / total if total > 0 else math.nan,
            "n_timesteps_csi": int(n_csi),
        }
        fss = torch.where(self.den == 0, torch.ones_like(self.den), 1.0 - self.num / self.den).cpu()
        for i, v in enumerate(self.ventanas):
            for j, u in enumerate(self.umbrales):
                res[f"fss_u{u}_v{v}"] = float(fss[i, j])
        return res
