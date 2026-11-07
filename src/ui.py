"""Componentes visuales de la app: estilos, encabezado, tarjeta de resultado y barras.

Aquí solo hay presentación (HTML y CSS sencillos). Todo texto que viene de los
datos pasa por ``html.escape`` antes de entrar al HTML. La lógica (cargar audio,
clasificar) vive en ``app.py`` y en ``src/``.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from src.animals import ANIMALES

ACENTO = "#FF9F43"

_CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1200px; }}
.hero h1 {{ font-size: 3rem; font-weight: 800; margin: 0; padding: 0; }}
.hero .degradado {{ background: linear-gradient(90deg, {ACENTO}, #FF6B6B);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.hero p {{ font-size: 1.1rem; opacity: .75; margin: .2rem 0 1.2rem 0; }}
.tarjeta {{ background: #1A2029; border-radius: 16px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }}
.resultado-nombre {{ font-size: 3.2rem; font-weight: 800; line-height: 1.1; }}
.resultado-conf {{ color: {ACENTO}; font-size: 1.8rem; font-weight: 700; }}
.resultado-nota {{ opacity: .6; font-size: .85rem; margin-top: .3rem; }}
.fila {{ display: flex; align-items: center; gap: .7rem; margin: .45rem 0; }}
.fila .nombre {{ width: 8.5rem; font-weight: 600; white-space: nowrap; }}
.fila .pista {{ display: block; flex: 1; background: #0F1318; border-radius: 8px; height: .9rem; overflow: hidden; }}
.fila .relleno {{ display: block; height: 100%; border-radius: 8px; background: {ACENTO}; }}
.fila .valor {{ width: 3.6rem; text-align: right; font-variant-numeric: tabular-nums; opacity: .85; }}
.chips {{ display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .6rem; }}
.chip {{ background: #1A2029; border-radius: 999px; padding: .35rem .9rem; font-size: .95rem; }}
[data-testid="stImage"] img {{ border-radius: 16px; }}
</style>
"""


def inyectar_estilos() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def encabezado() -> None:
    st.markdown(
        '<div class="hero"><h1><span class="degradado">GuauMiau</span> 🐶🐱</h1>'
        "<p>Sube un audio o grábalo con el micrófono y te digo qué animal suena.</p></div>",
        unsafe_allow_html=True,
    )


def tarjeta_resultado(nombre: str, emoji: str, score: float, etiqueta: str) -> None:
    """El resultado principal, grande: animal, confianza y de qué etiqueta salió."""
    st.markdown(
        f'<div class="tarjeta"><div class="resultado-nombre">{escape(nombre)} {escape(emoji)}</div>'
        f'<div class="resultado-conf">{score:.0%} de confianza</div>'
        f'<div class="resultado-nota">Etiqueta del modelo: {escape(etiqueta)}</div></div>',
        unsafe_allow_html=True,
    )


def barras_top(animales: list[dict], cuantos: int = 5) -> None:
    """Barras horizontales con el emoji, el nombre y el porcentaje de cada animal."""
    filas = "".join(
        f'<div class="fila"><span class="nombre">{escape(r["animal"].emoji)} '
        f'{escape(r["animal"].nombre)}</span>'
        f'<span class="pista"><span class="relleno" style="width:{min(r["score"], 1.0) * 100:.1f}%">'
        f'</span></span><span class="valor">{r["score"]:.1%}</span></div>'
        for r in animales[:cuantos]
    )
    st.markdown(
        f'<div class="tarjeta"><b>Top-{min(cuantos, len(animales))} de animales</b>'
        f'<div class="resultado-nota">Cada probabilidad es independiente; no suman 100 %.</div>'
        f"{filas}</div>",
        unsafe_allow_html=True,
    )


def animales_soportados() -> None:
    """Estado vacío: qué animales reconoce la app."""
    chips = "".join(
        f'<span class="chip">{escape(a.emoji)} {escape(a.nombre)}</span>' for a in ANIMALES
    )
    st.markdown(
        f'<div class="tarjeta"><b>Sube un audio para empezar.</b> Reconozco {len(ANIMALES)} animales:'
        f'<div class="chips">{chips}</div></div>',
        unsafe_allow_html=True,
    )
