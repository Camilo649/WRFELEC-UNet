import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_WRF(nn.Module):
    """
    U-Net con profundidad y ancho configurables.

    Parámetros
    ----------
    in_channels : int, default 6
        Cantidad de canales de entrada (variables WRF).
    n_niveles : int, default 2
        Cantidad de niveles de encoder/decoder (pooling + upsampling).
        La arquitectura original tenía 2 niveles.
    n_filtros_base : int, default 32
        Cantidad de filtros en el primer nivel del encoder. Cada nivel
        siguiente duplica esta cantidad (32→64→128→...), igual que el
        bottleneck, que duplica una vez más el último nivel.
    kernel_sizes : list[int] o None
        Un kernel_size por nivel del encoder (y su espejo en el decoder).
        Si es None, usa kernel_size=3 en todos los niveles.
        Debe tener longitud == n_niveles.
    dropout_ps : list[float] o None
        Un dropout_p por nivel del encoder (y su espejo en el decoder),
        aplicado también en el bottleneck (usa el último valor de la lista).
        Si es None, no aplica dropout en ningún nivel (comportamiento original).
        Debe tener longitud == n_niveles.
    input_size : int, default 297
        Tamaño espacial (H=W) de la imagen de entrada, usado para calcular
        el padding necesario según la profundidad elegida.
    """
    def __init__(self, in_channels=6, n_niveles=2, n_filtros_base=32,
                 kernel_sizes=None, dropout_ps=None, input_size=297):
        super(UNet_WRF, self).__init__()

        if kernel_sizes is None:
            kernel_sizes = [3] * n_niveles
        assert len(kernel_sizes) == n_niveles, \
            f"kernel_sizes debe tener {n_niveles} elementos, tiene {len(kernel_sizes)}"

        if dropout_ps is None:
            dropout_ps = [0.0] * n_niveles
        assert len(dropout_ps) == n_niveles, \
            f"dropout_ps debe tener {n_niveles} elementos, tiene {len(dropout_ps)}"
        
        self.n_niveles = n_niveles
        self.dropout_ps = dropout_ps

        # --- Padding espacial dinámico según profundidad ---
        divisor = 2 ** n_niveles
        size_ajustado = ((input_size // divisor) + 1) * divisor
        pad_total = size_ajustado - input_size
        # Repartimos el padding: menos a la izquierda/arriba, más a la derecha/abajo
        self.pad_izq_arr = pad_total // 2
        self.pad_der_aba = pad_total - self.pad_izq_arr
        self.input_size = input_size

        # --- Canales por nivel
        canales = [n_filtros_base * (2 ** i) for i in range(n_niveles)]
        canal_bottleneck = n_filtros_base * (2 ** n_niveles)
        # Kernel del bottleneck: reusamos el del nivel más profundo
        k_bottleneck = kernel_sizes[-1]
        dropout_bottleneck = dropout_ps[-1]

        def bloque_conv(canal_in, canal_out, k, dropout_p):
            pad = k // 2
            capas = [
                nn.Conv2d(canal_in, canal_out, kernel_size=k, padding=pad),
                nn.ReLU(inplace=True),
                nn.Conv2d(canal_out, canal_out, kernel_size=k, padding=pad),
                nn.ReLU(inplace=True),
            ]
            if dropout_p > 0:
                capas.append(nn.Dropout2d(p=dropout_p))
            return nn.Sequential(*capas)
        
        # --- ENCODER (dinámico) ---
        self.enc_blocks = nn.ModuleList()
        self.pools      = nn.ModuleList()
        canal_in = in_channels
        for canal_out, k, p in zip(canales, kernel_sizes, dropout_ps):
            self.enc_blocks.append(bloque_conv(canal_in, canal_out, k, p))
            self.pools.append(nn.MaxPool2d(2, 2))
            canal_in = canal_out

        # --- BOTTLENECK ---
        self.bottleneck = bloque_conv(canal_in, canal_bottleneck, k_bottleneck, dropout_bottleneck)

        # --- DECODER (dinámico, en orden inverso al encoder) ---
        self.ups        = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        canal_in = canal_bottleneck
        for canal_skip, k, p in zip(reversed(canales), reversed(kernel_sizes), reversed(dropout_ps)):
            self.ups.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            self.dec_blocks.append(bloque_conv(canal_in + canal_skip, canal_skip, k, p))
            canal_in = canal_skip

        # --- SALIDA ---
        self.salida  = nn.Conv2d(canal_in, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # SOLO para FACL

    def forward(self, x):
        x_pad = F.pad(x, (self.pad_izq_arr, self.pad_der_aba,
                           self.pad_izq_arr, self.pad_der_aba))

        # --- Encoder ---
        skips = []
        h = x_pad
        for bloque, pool in zip(self.enc_blocks, self.pools):
            h = bloque(h)
            skips.append(h)       # guardamos para el skip connection
            h = pool(h)

        # --- Bottleneck ---
        h = self.bottleneck(h)

        # --- Decoder ---
        for up, bloque, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = bloque(h)

        out = self.salida(h)
        out = self.sigmoid(out)  # SOLO para FACL

        # Recortar el padding agregado al principio
        out = out[:, :, self.pad_izq_arr:self.pad_izq_arr + self.input_size,
                         self.pad_izq_arr:self.pad_izq_arr + self.input_size]
        return out

# ==========================================
# FUNCION DE PÉRDIDA: FACL (Yan et al., NeurIPS 2024, arXiv:2410.23159)
# ==========================================
class FACL(nn.Module):
    """
    Fourier Amplitude and Correlation Loss, según el paper y su código oficial (https://github.com/argenycw/FACL).

        FAL(X, X^) = 1/(MN) * sum_pq ( |F_pq| - |F^_pq| )^2                          (Eq. 4)
        FCL(X, X^) = 1 - [ 1/2 * sum(F F^* + F^ F^*) ] / sqrt(sum|F|^2 * sum|F^|^2)  (Eq. 5)

        FACL(t) = FAL   si p > P(t)                                                  (Eq. 6)
                  FCL   si no
        con p ~ U[0,1] muestreado en cada paso de entrenamiento.

    P(t) decrece linealmente de 1 a 0: al principio se entrena 100% con FCL (estructura)
    y después cada vez más con FAL (nitidez). alpha es la fracción final de los pasos
    totales en la que P(t) = 0 (FAL puro). El paper recomienda alpha = 0.1 o 0.2.

    FCL GLOBAL: igual que el repo oficial y que RandomFourierLoss de rainpostproc, las sumas
    de la Eq. 5 se hacen sobre TODO el tensor (batch incluido), no imagen por imagen. Por
    Parseval, FCL = 1 - coseno entre los mapas aplanados, así que es el coseno del batch
    entero tomado como un único vector: las muestras más intensas pesan más.

    Requisitos del paper que se cumplen fuera de esta clase:
      - Sigmoide en la salida del modelo (ya está en UNet_WRF).
      - Scheduler cosine annealing (ya está: CosineAnnealingLR con T_max=EPOCHS).

    Parámetros
    ----------
    total_steps : int
        Pasos de optimización totales = EPOCHS * len(train_loader).
    alpha : float
        Fracción de pasos finales con P(t) = 0. Debe estar en [0, 1).
    eps : float
        Estabilidad numérica en el denominador de FCL. NO figura en la Eq. 5; solo entra en
        juego si el batch completo tiene target vacío (0/0).

    Validación (el paper no la define)
    ----------------------------------
    0.5*FAL + 0.5*FCL, calculada sobre TODO el set de validación de una vez, como en
    rainpostproc. Como FCL es global, promediar la pérdida por batches NO da lo mismo que
    calcularla sobre el set completo; por eso se acumulan las sumas batch a batch:

        criterion.val_reset()
        for x, y in val_loader:
            criterion.val_acumular(modelo(x), y)
        val_loss = criterion.val_resultado()

    forward(..., mode='val') sigue existiendo pero evalúa UN solo batch.
    """
    def __init__(self, total_steps, alpha=0.1, eps=1e-8):
        super().__init__()
        assert 0.0 <= alpha < 1.0, f"alpha es una fracción de los pasos totales, debe estar en [0, 1): {alpha}"
        self.total_steps = int(total_steps)
        self.alpha       = alpha
        self.eps         = eps
        self.step        = 0
        # Igual que el repo oficial: linspace(1, 0, T - int(alpha*T)) y después se queda en 0
        self.n_decay = max(int(self.total_steps - int(self.total_steps * alpha)), 2)
        self.val_reset()

    def prob_fcl(self, t=None):
        """P(t): probabilidad de elegir FCL en el paso t."""
        t = self.step if t is None else t
        return max(0.0, 1.0 - t / (self.n_decay - 1))

    @staticmethod
    def _fft(x):
        # DFT 2D ortonormalizada (Eq. 2), en float32 por estabilidad
        return torch.fft.fftn(x.float(), dim=[-2, -1], norm='ortho')

    def fal(self, fft_pred, fft_gt):
        # Eq. 4: media de (|F| - |F^|)^2 sobre todas las frecuencias (y el batch)
        return ((fft_gt.abs() - fft_pred.abs()) ** 2).mean()

    def fcl(self, fft_pred, fft_gt):
        # Eq. 5, con las sumas sobre todo el tensor (global).
        # Las dos normas van en raíces SEPARADAS: la de gt no lleva gradiente y vale 0 si el
        # target está vacío; sqrt(n_gt * n_pred) juntos daría gradiente NaN (inf * 0).
        cruzado    = 0.5 * (fft_gt * torch.conj(fft_pred) + fft_pred * torch.conj(fft_gt))
        num        = cruzado.real.sum()
        norma_pred = torch.sqrt((fft_pred.abs() ** 2).sum())
        norma_gt   = torch.sqrt((fft_gt.abs() ** 2).sum())
        return 1.0 - num / (norma_pred * norma_gt + self.eps)

    # ---- Validación sobre todo el set, acumulando sumas ----
    def val_reset(self):
        self._acc = {'num': 0.0, 'n_gt': 0.0, 'n_pred': 0.0, 'sq': 0.0, 'n': 0}

    @torch.no_grad()
    def val_acumular(self, pred, gt):
        fp, fg  = self._fft(pred), self._fft(gt)
        cruzado = 0.5 * (fg * torch.conj(fp) + fp * torch.conj(fg))
        self._acc['num']    += cruzado.real.sum(dtype=torch.float64).item()
        self._acc['n_gt']   += (fg.abs() ** 2).sum(dtype=torch.float64).item()
        self._acc['n_pred'] += (fp.abs() ** 2).sum(dtype=torch.float64).item()
        self._acc['sq']     += ((fg.abs() - fp.abs()) ** 2).sum(dtype=torch.float64).item()
        self._acc['n']      += fp.numel()

    def val_resultado(self):
        a = self._acc
        # float() a propósito: np.sqrt devuelve np.float64, y con PyTorch >= 2.6 torch.load
        # (weights_only=True) no puede leer escalares de numpy guardados en el checkpoint.
        fal = float(a['sq'] / a['n'])
        fcl = float(1.0 - a['num'] / (np.sqrt(a['n_pred']) * np.sqrt(a['n_gt']) + self.eps))
        self.val_fal, self.val_fcl = fal, fcl      # por si querés ver los dos términos por separado
        return float(0.5 * fal + 0.5 * fcl)

    def forward(self, pred, gt, mode='train'):
        fft_pred, fft_gt = self._fft(pred), self._fft(gt)

        if mode == 'val':      # un solo batch; para el set completo usar val_reset/val_acumular/val_resultado
            return 0.5 * self.fal(fft_pred, fft_gt) + 0.5 * self.fcl(fft_pred, fft_gt)

        p = np.random.rand()
        loss = self.fal(fft_pred, fft_gt) if p > self.prob_fcl() else self.fcl(fft_pred, fft_gt)
        self.step += 1
        return loss

