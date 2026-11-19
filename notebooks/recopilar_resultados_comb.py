"""Reúne en una sola carpeta los resultados de la Ext Clip U-Net Comb V2.1 que dejó el notebook
Entrenamiento_Ext_Clip_U-Net_Comb_V2.1.ipynb, con la misma estructura que los demás modelos de results/.

Uso (en el JupyterHub, desde la carpeta notebooks/, es decir, la que tiene ../images, ../modelos, ../videos&gifs):
    python recopilar_resultados_comb.py            # arma ../resultados_ext_clip_u-net_comb_v2.1/ y su .zip
    python recopilar_resultados_comb.py --sin-mp4  # deja las animaciones en GIF (no convierte)

Resultado (se copia a results/ext_clip_u-net_comb_v2.1/ del repo):
    historial_entrenamiento.txt     pérdidas por época
    curva_error.png                 curvas de pérdida
    metricas/                       MSE, RMSE, MAE, SSIM, POD, FAR, CSI, FSS, residuos, sensibilidad del SSIM,
                                    R2 (txt) y las métricas por muestra (csv)
    distribucion_curvas/            distribución real vs. estimada (global y por tormenta)
    animaciones/                    reconstrucciones de las tormentas (MP4)
    reconstrucciones/               ejemplos de reconstrucción (PNG)
Si una carpeta de origen guarda varias versiones de un archivo (nombre, nombre_01, nombre_02...), se toma la última.
"""
import os, re, sys, glob, shutil, subprocess, zipfile

NOMBRE = "ext_clip_u-net_comb_v2.1"
NOMBRE_MODELO = "Ext Clip U-Net Comb V2.1"
SALIDA = f"../resultados_{NOMBRE}"
IMG, MOD, VID, EST, RES = "../images", "../modelos", "../videos&gifs", "../estudios", f"../results/{NOMBRE}"
SIN_MP4 = "--sin-mp4" in sys.argv
faltan = []


def ultima(carpeta, base, ext):
    """Archivo base.ext o la versión con contador más alto (base_01.ext, base_02.ext...)."""
    cands = []
    for ruta in glob.glob(os.path.join(carpeta, f"{glob.escape(base)}*{ext}")):
        m = re.fullmatch(re.escape(base) + r"(?:_(\d+))?" + re.escape(ext), os.path.basename(ruta))
        if m:
            cands.append((int(m.group(1) or 0), ruta))
    return max(cands)[1] if cands else None


def copiar(origen, destino, requerido=True):
    if origen is None or not os.path.exists(origen):
        if requerido:
            faltan.append(destino)
        return False
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    shutil.copy2(origen, destino)
    print(f"OK  {destino}")
    return True


if os.path.exists(SALIDA):
    shutil.rmtree(SALIDA)
os.makedirs(SALIDA)

# --- historial y curva de error ---
copiar(os.path.join(MOD, f"historial_{NOMBRE_MODELO}.txt"), f"{SALIDA}/historial_entrenamiento.txt")
copiar(ultima(IMG, f"curva_error_{NOMBRE}", ".png"), f"{SALIDA}/curva_error.png")

# --- métricas ---
for m in ("mse", "rmse", "mae", "ssim", "pod", "far", "csi"):
    copiar(ultima(IMG, f"{m}_{NOMBRE}", ".png"), f"{SALIDA}/metricas/{m}.png")
copiar(ultima(IMG, f"fss_heatmap_{NOMBRE}", ".png"), f"{SALIDA}/metricas/fss_heatmap.png")
copiar(ultima(IMG, f"sensibilidad_ssim_ventanas_{NOMBRE}", ".png"), f"{SALIDA}/metricas/sensibilidad_ssim_ventanas.png")
for suf in ("full", "noedge10"):
    copiar(ultima(IMG, f"residuals_{NOMBRE}_{suf}", ".png"), f"{SALIDA}/metricas/residuals_{suf}.png")
    copiar(os.path.join(MOD, f"r2_{NOMBRE}_{suf}.txt"), f"{SALIDA}/metricas/r2_{suf}.txt")
for csv in ("metricas_por_muestra.csv", "ssim_ventanas.csv", "matriz_fss.csv"):
    copiar(os.path.join(RES, csv), f"{SALIDA}/metricas/{csv}")

# --- distribución de curvas ---
copiar(ultima(IMG, f"dist_curvas_{NOMBRE}", ".png"), f"{SALIDA}/distribucion_curvas/dist_curvas.png")
tormentas = sorted({re.fullmatch(rf"dist_curvas_{re.escape(NOMBRE)}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{4}})(?:_\d+)?\.png",
                                 os.path.basename(r)).group(1)
                    for r in glob.glob(os.path.join(IMG, f"dist_curvas_{NOMBRE}_*.png"))
                    if re.fullmatch(rf"dist_curvas_{re.escape(NOMBRE)}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{4}})(?:_\d+)?\.png",
                                    os.path.basename(r))})
if not tormentas:
    faltan.append("distribucion_curvas/dist_curvas_<tormenta>.png")
for t in tormentas:
    copiar(ultima(IMG, f"dist_curvas_{NOMBRE}_{t}", ".png"), f"{SALIDA}/distribucion_curvas/dist_curvas_{t}.png")

# --- reconstrucciones (PNG) ---
for ruta in sorted(glob.glob(os.path.join(IMG, f"reconstruccion_{NOMBRE}_idx_*.png"))):
    copiar(ruta, f"{SALIDA}/reconstrucciones/{os.path.basename(ruta).replace(f'reconstruccion_{NOMBRE}_', 'reconstruccion_')}",
           requerido=False)

# --- animaciones (GIF -> MP4, como en los demás modelos) ---
gifs = sorted(glob.glob(os.path.join(VID, f"reconstruction_{NOMBRE}_*.gif")))
if not gifs:
    faltan.append("animaciones/reconstruction_<tormenta>.mp4")
hay_ffmpeg = shutil.which("ffmpeg") is not None
for g in gifs:
    t = os.path.basename(g)[len(f"reconstruction_{NOMBRE}_"):-len(".gif")]
    os.makedirs(f"{SALIDA}/animaciones", exist_ok=True)
    mp4 = f"{SALIDA}/animaciones/reconstruction_{t}.mp4"
    if hay_ffmpeg and not SIN_MP4:
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", g, "-movflags", "faststart",
                            "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                            "-c:v", "libx264", "-crf", "23", mp4])
        if r.returncode == 0:
            print(f"OK  {mp4}")
            continue
        print(f"AVISO: ffmpeg falló con {g}; se deja el GIF")
    copiar(g, f"{SALIDA}/animaciones/reconstruction_{t}.gif")
if gifs and not hay_ffmpeg and not SIN_MP4:
    print("AVISO: no hay ffmpeg en este entorno; las animaciones quedaron en GIF "
          "(convertilas en tu compu: ffmpeg -i x.gif -movflags faststart -pix_fmt yuv420p "
          "-vf scale=trunc(iw/2)*2:trunc(ih/2)*2 x.mp4).")

# --- zip ---
ruta_zip = f"{SALIDA}.zip"
with zipfile.ZipFile(ruta_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for raiz, _, archivos in os.walk(SALIDA):
        for a in archivos:
            ruta = os.path.join(raiz, a)
            z.write(ruta, os.path.join(NOMBRE, os.path.relpath(ruta, SALIDA)))
print(f"\nZIP: {ruta_zip} ({os.path.getsize(ruta_zip)/1e6:.1f} MB)")
if faltan:
    print("\nFALTAN (corré en el notebook la celda que los genera y repetí):")
    for f in faltan:
        print("  -", f)
else:
    print("Está todo.")
