"""Gráficos de matplotlib para la app: onda, espectrograma mel, línea de tiempo y benchmark.

Las funciones reciben arrays de numpy (no tensores) para no depender de torch,
y devuelven una ``Figure`` que Streamlit dibuja con ``st.pyplot``. Se usa
``Figure`` directamente (y no ``pyplot``) porque Streamlit ejecuta el script en
varios hilos y ``pyplot`` guarda estado global.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure


def figura_onda(onda: np.ndarray, sr: int) -> Figure:
    """Amplitud del audio en el tiempo."""
    tiempo = np.arange(len(onda)) / sr
    fig = Figure(figsize=(8, 2.2), layout="constrained")
    ax = fig.subplots()
    ax.plot(tiempo, onda, linewidth=0.5, color="tab:blue")
    ax.set_xlim(0, tiempo[-1] if len(tiempo) else 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    return fig


def figura_mel(mel_db: np.ndarray, duracion: float) -> Figure:
    """Espectrograma mel: tiempo en X, banda de frecuencia en Y, color = volumen (dB)."""
    fig = Figure(figsize=(8, 3), layout="constrained")
    ax = fig.subplots()
    img = ax.imshow(
        mel_db, origin="lower", aspect="auto", cmap="magma",
        extent=(0, duracion, 0, mel_db.shape[0]),
    )
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Banda mel")
    fig.colorbar(img, ax=ax, label="dB")
    return fig


def figura_linea_tiempo(segmentos: list, duracion: float) -> Figure:
    """Barra horizontal con un tramo de color por ventana y el animal que suena.

    ``segmentos`` es la salida de ``src.timeline.linea_de_tiempo``. Cada animal
    tiene siempre el mismo color, y los tramos sin animal se dibujan en gris.
    La transparencia del tramo indica la confianza.
    """
    claves = sorted({s.animal.clave for s in segmentos if s.animal is not None})
    paleta = {c: f"C{i % 10}" for i, c in enumerate(claves)}

    fig = Figure(figsize=(8, 1.8), layout="constrained")
    ax = fig.subplots()
    for s in segmentos:
        ancho = s.fin - s.inicio
        if s.animal is None:
            color, alpha, texto = "lightgray", 1.0, "—"
        else:
            color = paleta[s.animal.clave]
            alpha = 0.35 + 0.65 * min(s.score, 1.0)
            texto = f"{s.animal.nombre}\n{s.score:.0%}"
        ax.barh(0, ancho, left=s.inicio, height=0.8, color=color, alpha=alpha,
                edgecolor="white", linewidth=1.5)
        ax.text(s.inicio + ancho / 2, 0, texto, ha="center", va="center", fontsize=8)

    ax.set_xlim(0, duracion if duracion > 0 else 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Tiempo (s)")
    for lado in ("left", "right", "top"):
        ax.spines[lado].set_visible(False)
    return fig


def figura_benchmark(resultados: list) -> Figure:
    """Tres paneles CPU contra GPU: latencia, throughput y VRAM.

    ``resultados`` es la salida de ``src.cuda_bench.comparar``. Los dos primeros
    paneles agrupan las barras por tamaño de lote; el de VRAM solo tiene datos
    de la GPU (la CPU usa RAM normal, que no mide ``torch.cuda``).
    """
    lotes = sorted({r.lote for r in resultados})
    dispositivos = [d for d in ("CPU", "GPU") if any(r.dispositivo == d for r in resultados)]
    colores = {"CPU": "tab:gray", "GPU": "tab:green"}
    por_clave = {(r.dispositivo, r.lote): r for r in resultados}
    x = np.arange(len(lotes))
    ancho = 0.8 / len(dispositivos)

    fig = Figure(figsize=(11, 3.6), layout="constrained")
    ax_lat, ax_thr, ax_vram = fig.subplots(1, 3)

    for i, d in enumerate(dispositivos):
        filas = [por_clave.get((d, lote)) for lote in lotes]
        pos = x + (i - (len(dispositivos) - 1) / 2) * ancho
        for ax, campo in ((ax_lat, "latencia_ms"), (ax_thr, "clips_por_s")):
            valores = [getattr(r, campo) if r else 0 for r in filas]
            barras = ax.bar(pos, valores, ancho, label=d, color=colores[d])
            ax.bar_label(barras, fmt="%.0f" if campo == "latencia_ms" else "%.1f", fontsize=8)

    for ax, titulo, eje in (
        (ax_lat, "Latencia por lote", "ms (menos es mejor)"),
        (ax_thr, "Throughput", "clips de 10 s / s (más es mejor)"),
    ):
        ax.set_title(titulo)
        ax.set_ylabel(eje)
        ax.set_xticks(x, [f"lote {lote}" for lote in lotes])
        ax.margins(y=0.15)
    ax_lat.legend()

    gpu = [por_clave[("GPU", lote)] for lote in lotes if ("GPU", lote) in por_clave]
    ax_vram.set_title("VRAM de la GPU")
    if gpu:
        ax_vram.axhline(gpu[0].vram_modelo_mb, color="tab:blue", linestyle="--",
                        label=f"pesos del modelo ({gpu[0].vram_modelo_mb:.0f} MB)")
        barras = ax_vram.bar([f"lote {r.lote}" for r in gpu], [r.vram_pico_mb for r in gpu],
                             color=colores["GPU"], alpha=0.8, label="pico en inferencia")
        ax_vram.bar_label(barras, fmt="%.0f MB", fontsize=8)
        ax_vram.set_ylabel("MB")
        ax_vram.margins(y=0.15)
        ax_vram.legend(fontsize=8, loc="lower right")
    else:
        ax_vram.text(0.5, 0.5, "Sin GPU CUDA", ha="center", va="center",
                     transform=ax_vram.transAxes, color="gray")
        ax_vram.set_xticks([])
        ax_vram.set_yticks([])
    return fig
