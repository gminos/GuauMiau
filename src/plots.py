"""Gráficos de matplotlib para la app: forma de onda y espectrograma mel.

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
