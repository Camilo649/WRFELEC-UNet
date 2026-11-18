"""Pérdida Comb V2.1 (clase HybridFourierLossV2 de Entrenamiento_unetCombv2.ipynb, sin uncertainty weighting).

  Q(t): prob. de usar el WMSE. Se replica EL CÓDIGO del notebook:
        prob_t = min{ (t * prob_slope) * (max_prob - min_prob) + min_prob , max_prob },
        prob_slope = (max_prob - min_prob) / (epocas_para_max_prob * pasos_por_epoca)
      OJO: (max_prob - min_prob) entra dos veces, así que Q llega a max_prob a las epocas_para_max_prob/(max_prob-min_prob)
      épocas (con 0,7 y 60: ~86), no a las 60 nominales. Con rampa_exacta=True se usa la fórmula de la monografía,
      Q = min{min_prob + prob_slope * t, max_prob} (llega a las epocas_para_max_prob).
  train: con prob. Q -> WMSE; si no, FAL o FCL con prob. 1/2 cada una.
      WMSE: error cuadrático sobre LIGHTDENS desnormalizado, peso w+ si y_real > 0 y 1 si no, promedio sobre píxeles.
      FAL = mean((|F_pred| - |F_gt|)^2) * sqrt(H*W);   FCL = mean_imagen(1 - <F_pred,F_gt>/(|F_pred||F_gt| + eps)) * sqrt(H*W)
      (FFT ortonormalizada, sobre los valores normalizados; FCL por imagen, como en el notebook).
  val : (WMSE + FAL*sqrt(HW) + FCL*sqrt(HW)) / 3. Como las tres son medias sobre píxeles o imágenes, acumular por lotes
      (suma ponderada por tamaño de lote) da exactamente el valor sobre todo el set.
"""
import numpy as np
import torch
import torch.nn as nn


class CombV21(nn.Module):
    def __init__(self, desnormalizar_y, peso_positivo, max_prob, epocas_para_max_prob, pasos_por_epoca,
                 min_prob=0.0, eps=1e-8, rampa_exacta=False):
        super().__init__()
        assert 0.0 <= min_prob < max_prob <= 1.0, (min_prob, max_prob)
        self.desnormalizar_y = desnormalizar_y
        self.peso_positivo = float(peso_positivo)
        self.min_prob, self.max_prob = float(min_prob), float(max_prob)
        self.prob_slope = (self.max_prob - self.min_prob) / (max(epocas_para_max_prob, 1e-9) * pasos_por_epoca)
        self.rampa_exacta = rampa_exacta
        self.eps = eps
        self.step = 0
        self.ultimo_termino = None
        self.val_reset()

    def prob_wmse(self, t=None):
        t = self.step if t is None else t
        if self.rampa_exacta:
            p = self.min_prob + self.prob_slope * t
        else:
            p = (t * self.prob_slope) * (self.max_prob - self.min_prob) + self.min_prob
        return min(p, self.max_prob)

    @staticmethod
    def _fft(x):
        return torch.fft.fftn(x.float().squeeze(1), dim=[-1, -2], norm='ortho')

    def fal(self, fp, fg):
        return ((fp.abs() - fg.abs()) ** 2).mean()

    def fcl(self, fp, fg):
        dims = tuple(range(1, fp.dim()))
        num = torch.sum((fp * torch.conj(fg)).real, dim=dims)
        n_p = torch.sqrt(torch.sum(fp.abs() ** 2, dim=dims))
        n_g = torch.sqrt(torch.sum(fg.abs() ** 2, dim=dims))
        return (1 - num / (n_p * n_g + self.eps)).mean()

    def wmse(self, pred, gt):
        p, g = self.desnormalizar_y(pred.float()), self.desnormalizar_y(gt.float())
        w = torch.where(g > 0, torch.full_like(g, self.peso_positivo), torch.ones_like(g))
        return ((p - g) ** 2 * w).mean()

    @staticmethod
    def _peso_fourier(pred):
        H, W = pred.shape[-2:]
        return float(np.sqrt(H * W))

    # ---- validación sobre todo el set ----
    def val_reset(self):
        self._v = {'wmse': 0.0, 'fal': 0.0, 'fcl': 0.0, 'n': 0, 'w': 1.0}

    @torch.no_grad()
    def val_acumular(self, pred, gt):
        fp, fg = self._fft(pred), self._fft(gt)
        n = pred.shape[0]
        self._v['wmse'] += float(self.wmse(pred, gt)) * n
        self._v['fal'] += float(self.fal(fp, fg)) * n
        self._v['fcl'] += float(self.fcl(fp, fg)) * n
        self._v['n'] += n
        self._v['w'] = self._peso_fourier(pred)

    def val_resultado(self):
        v = self._v
        self.val_wmse, self.val_fal, self.val_fcl = (v[k] / v['n'] for k in ('wmse', 'fal', 'fcl'))   # FAL y FCL sin escalar
        return float((self.val_wmse + v['w'] * self.val_fal + v['w'] * self.val_fcl) / 3.0)

    def forward(self, pred, gt, mode='train'):
        if mode == 'val':      # un solo lote (para el set completo: val_reset / val_acumular / val_resultado)
            fp, fg = self._fft(pred), self._fft(gt)
            w = self._peso_fourier(pred)
            return (self.wmse(pred, gt) + w * self.fal(fp, fg) + w * self.fcl(fp, fg)) / 3.0
        if np.random.rand() < self.prob_wmse():
            loss, self.ultimo_termino = self.wmse(pred, gt), 'WMSE'
        else:
            fp, fg = self._fft(pred), self._fft(gt)
            w = self._peso_fourier(pred)
            if np.random.rand() < 0.5:
                loss, self.ultimo_termino = self.fal(fp, fg) * w, 'FAL'
            else:
                loss, self.ultimo_termino = self.fcl(fp, fg) * w, 'FCL'
        self.step += 1
        return loss
