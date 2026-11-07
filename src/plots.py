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
