"""Pérdida FACL copiada TEXTUAL del código oficial (Yan et al., 2024, arXiv:2410.23159).

Origen: https://github.com/argenycw/FACL , archivo utilspp.py, clase RandomScheduling (líneas 563-613),
commit f46009f119459c9811d239f370c9d7ae1d6148ba. La clase NO tiene ningún cambio: está copiada carácter
por carácter. Este módulo solo agrega los imports que la clase necesita (en utilspp.py están arriba del
archivo, junto con dependencias que acá no hacen falta: skimage, lpips, torchmetrics, etc.).

Uso (igual que train_meteo.py, función get_loss, con la pérdida "facl-<alpha>"):
    criterion = RandomScheduling(total_step, micro_batch, const_ratio=alpha)
    loss = criterion(y_pred, y)        # y_pred, y con forma (B, T, C, H, W)
La forward desempaqueta `_, _, _, H, W = pred.shape`, o sea que espera tensores 5D. Para mapas 4D
(B, C, H, W) se agrega una dimensión temporal de largo 1 con .unsqueeze(1) antes de llamarla: la FFT es
sobre las dos últimas dimensiones y las sumas/medias son sobre todo el tensor, así que el resultado es el
mismo que con T = 1 en el código oficial.

Licencia del código original (MIT):

MIT License

Copyright (c) 2024 Argen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import random

import numpy as np
import torch
from torch import nn


class RandomScheduling(nn.Module):
    def __init__(self, total_step, micro_batch, const_ratio=0.4):
        super(RandomScheduling, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        const_step = int(total_step*const_ratio)
        self.prob_thres = torch.linspace(1,0, int(total_step-const_step)).to(device)
        # dec = torch.linspace(1,0, int(total_step-const_step)).to(device)
        # if(const_ratio!=1):
        #     const = dec[-1] * torch.ones(const_step, device=device)
        # else:
        #     const = torch.zeros(int(total_step*const_ratio), device=device)
        # self.prob_thres = torch.cat((dec, const), dim=0)
        self.micro_batch = micro_batch
        self.step = 0
        self.out = 0

    def get_thres(self):
        if self.step % self.micro_batch == 0:
            prob = self.prob_thres[self.step//self.micro_batch] if self.step//self.micro_batch < len(self.prob_thres) else self.prob_thres[-1]
            self.out = 1 if random.random() > prob else 0
        self.step += 1
        return self.out

    def fcl(self, fft_pred, fft_truth):
        # In general, FFTs here must be shifted to the center; but here we use the whole fourier space, so it is okay to no need have fourier shift operation
        conj_pred = torch.conj(fft_pred)
        numerator = (conj_pred*fft_truth).sum().real
        denominator = torch.sqrt(((fft_truth).abs()**2).sum()*((fft_pred).abs()**2).sum())
        return 1. - numerator/denominator

    def fal(self, fft_pred, fft_truth):
        return nn.MSELoss()(fft_pred.abs(), fft_truth.abs())

    def cos_loss(self, fft_pred, fft_truth):
        # Cosine Similarity Loss
        numerator = fft_pred.real*fft_truth.real + fft_pred.imag*fft_truth.imag
        denominator = fft_pred.abs()*fft_truth.abs() + 1E-7 
        return (1. - numerator/denominator).mean()

    def forward(self, pred, gt):
        fft_pred = torch.fft.fftn(pred, dim=[-1,-2], norm='ortho')
        fft_gt = torch.fft.fftn(gt, dim=[-1,-2], norm='ortho')
        # prob = 1 if random.random() > self.prob_thres[self.step] else 0
        prob = self.get_thres()

        _, _, _, H, W = pred.shape
        weight = np.sqrt(H*W)
        loss = prob*self.fal(fft_pred, fft_gt) + (1-prob) * self.fcl(fft_pred, fft_gt)
        loss = loss*weight
        # self.step += 1
        return loss
