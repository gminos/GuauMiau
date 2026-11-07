"""Línea de tiempo para audios largos.

AST solo mira ~10 s de audio, así que en un audio largo se perdería todo lo que
suena después. Para no perderlo:

1. el audio se divide en ventanas de :data:`VENTANA_S` segundos;
2. las ventanas se clasifican por lotes con ``predict_batch``, de modo que la
   GPU procesa varias a la vez en lugar de una por una;
3. cada ventana se traduce a un animal con ``agrupar`` (igual que en la app).

El resultado dice qué animal suena en cada tramo del audio.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.animals import Animal, agrupar
from src.audio import TARGET_SR
from src.model import predict_batch

VENTANA_S = 5.0
# Ventanas por lote. Con fp16 en GPU caben de sobra; en CPU solo cambia el
# tamaño de cada paso, no el resultado.
LOTE = 16
# Un trozo final más corto que esto no se analiza por separado: se une a la
# ventana anterior (si no, un resto de 0,1 s daría predicciones sin sentido).
MIN_RESTO_S = 1.0
# Igual que en la app: se piden más etiquetas de las que se muestran para que
# ningún animal quede fuera, y por debajo del umbral no se afirma nada.
ETIQUETAS_A_PEDIR = 30
UMBRAL = 0.10


@dataclass(frozen=True)
class Segmento:
    inicio: float  # segundos
    fin: float  # segundos
    animal: Animal | None  # None si ningún animal supera el umbral
    score: float  # probabilidad del mejor animal (0 si no hay)
    etiqueta: str | None  # etiqueta de AudioSet que dio esa probabilidad


def dividir_en_ventanas(
    waveform: torch.Tensor,
    sr: int = TARGET_SR,
    ventana_s: float = VENTANA_S,
    min_resto_s: float = MIN_RESTO_S,
) -> list[tuple[float, float, torch.Tensor]]:
    """Corta un audio mono en ventanas consecutivas, sin solaparse.

    Devuelve ``[(inicio_s, fin_s, trozo), ...]``. Si el último trozo dura menos
    de ``min_resto_s``, se une a la ventana anterior en lugar de quedar solo.
    """
    if waveform.ndim != 1:
        raise ValueError(f"Se esperaba audio mono 1D, llegó forma {tuple(waveform.shape)}")
    paso = int(round(ventana_s * sr))
    if paso <= 0:
        raise ValueError(f"Ventana inválida: {ventana_s} s")

    total = waveform.numel()
    cortes = list(range(0, total, paso)) + [total]
    if len(cortes) > 2 and (cortes[-1] - cortes[-2]) < min_resto_s * sr:
        del cortes[-2]  # el resto corto se une a la ventana anterior

    return [
        (a / sr, b / sr, waveform[a:b])
        for a, b in zip(cortes[:-1], cortes[1:])
    ]


def linea_de_tiempo(
    waveform: torch.Tensor,
    ventana_s: float = VENTANA_S,
    lote: int = LOTE,
    umbral: float = UMBRAL,
) -> list[Segmento]:
    """Qué animal suena en cada ventana de ``ventana_s`` segundos del audio."""
    ventanas = dividir_en_ventanas(waveform, ventana_s=ventana_s)
    segmentos: list[Segmento] = []
    for i in range(0, len(ventanas), lote):
        grupo = ventanas[i : i + lote]
        tops = predict_batch([trozo for _, _, trozo in grupo], top_k=ETIQUETAS_A_PEDIR)
        for (inicio, fin, _), top in zip(grupo, tops):
            animales = agrupar(top)
            if animales and animales[0]["score"] >= umbral:
                mejor = animales[0]
                segmentos.append(
                    Segmento(inicio, fin, mejor["animal"], mejor["score"], mejor["etiqueta"])
                )
            else:
                segmentos.append(Segmento(inicio, fin, None, 0.0, None))
    return segmentos
