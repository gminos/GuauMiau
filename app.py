"""Interfaz Streamlit de GuauMiau.

Flujo: el usuario sube un audio (.wav/.mp3) o graba con el micrófono ->
``preprocess`` lo deja mono a 16 kHz -> ``predict`` (AST) devuelve el top-5.

Ejecutar con:  uv run streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st
import torch

from src.animals import agrupar
from src.audio import TARGET_SR, mel_spectrogram, preprocess
from src.cuda_bench import aceleracion, comparar
from src.model import load_model, predict, select_device
from src.plots import figura_benchmark, figura_linea_tiempo, figura_mel, figura_onda
from src.timeline import MIN_RESTO_S, VENTANA_S, linea_de_tiempo
from src.ui import animales_soportados, barras_top, encabezado, inyectar_estilos, tarjeta_resultado

st.set_page_config(page_title="GuauMiau", page_icon="🐶", layout="wide")

# Se piden más etiquetas de las que se muestran: un animal puede estar descrito
# por varias (Dog, Bark...) y, si se corta en 5, alguna podría quedar fuera.
ETIQUETAS_A_PEDIR = 30
# Por debajo de esta probabilidad no afirmamos qué animal es.
UMBRAL = 0.10
# Lotes del benchmark en la app: pocos y pequeños, porque la CPU es lenta.
BENCH_LOTES = (1, 4)


@st.cache_resource(show_spinner="Cargando el modelo AST (la primera vez se descarga)...")
def get_model():
    # Streamlit vuelve a ejecutar todo el script en cada interacción.
    # cache_resource evita recargar el modelo (cientos de MB) cada vez.
    return load_model()


def classify(audio_file) -> tuple[torch.Tensor, list[dict]]:
    """Guarda el audio en un archivo temporal, lo preprocesa y lo clasifica.

    ``preprocess`` lee desde una ruta, y Streamlit entrega los archivos
    subidos (o grabados) en memoria; por eso se escriben a disco un momento.
    """
    suffix = Path(audio_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_file.getvalue())
        tmp_path = Path(tmp.name)
    try:
        waveform = preprocess(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return waveform, predict(waveform, top_k=ETIQUETAS_A_PEDIR)


inyectar_estilos()
encabezado()

# --- Barra lateral: la app avisa si usa GPU (CUDA) o CPU ---
device, dtype = select_device()
with st.sidebar:
    st.header("⚙️ Sistema")
    if device.type == "cuda":
        st.success(f"GPU: {torch.cuda.get_device_name(0)}\n\nCUDA · fp16")
    else:
        st.warning("No se detectó GPU: se usa la CPU (fp32). Funciona, pero más lento.")
    st.divider()
    st.caption("Modelo: AST de MIT (Hugging Face) · 86,6 M de parámetros · entrenado con AudioSet.")

tab_detectar, tab_tiempo, tab_bench = st.tabs(
    ["🔍 Detectar", "⏱️ Línea de tiempo", "⚡ Benchmark CUDA"]
)

waveform = None  # lo llena la pestaña "Detectar"; la línea de tiempo lo reutiliza

# --- Pestaña 1: subir o grabar un audio y ver el resultado ---
with tab_detectar:
    sub_archivo, sub_mic = st.tabs(["📁 Subir archivo", "🎤 Grabar"])
    with sub_archivo:
        uploaded = st.file_uploader("Archivo de audio", type=["wav", "mp3"])
    with sub_mic:
        recorded = st.audio_input("Graba un sonido")
    audio_file = uploaded or recorded

    if audio_file is None:
        animales_soportados()
    else:
        st.audio(audio_file)
        get_model()  # fuerza la carga (con spinner) antes de clasificar
        top = None
        with st.spinner("Analizando..."):
            try:
                waveform, top = classify(audio_file)
            except (ValueError, RuntimeError, FileNotFoundError) as err:
                st.error(f"No se pudo procesar el audio: {err}")

        if top is not None:
            seconds = waveform.numel() / TARGET_SR
            if seconds > 10.24:
                st.info(
                    f"El audio dura {seconds:.1f} s. El resultado usa solo los primeros ~10 s; "
                    "la pestaña **Línea de tiempo** analiza el audio completo."
                )

            animales = agrupar(top)
            if not animales or animales[0]["score"] < UMBRAL:
                st.subheader("No parece un animal que conozca 🤔")
            else:
                mejor = animales[0]
                animal = mejor["animal"]
                col_foto, col_res = st.columns([1, 1.4], gap="large")
                with col_foto:
                    if animal.foto:
                        st.image(str(animal.foto), width="stretch")
                with col_res:
                    tarjeta_resultado(animal.nombre, animal.emoji, mejor["score"], mejor["etiqueta"])
                    barras_top(animales)

            with st.expander("Etiquetas originales del modelo (AudioSet)"):
                for item in top[:5]:
                    st.write(f"{item['label']}: {item['score']:.1%}")

            # --- Lo que el modelo "ve": onda y espectrograma, lado a lado ---
            st.subheader("Lo que el modelo ve")
            col_onda, col_mel = st.columns(2)
            with col_onda:
                st.write("**Forma de onda**")
                st.pyplot(figura_onda(waveform.numpy(), TARGET_SR))
            with col_mel:
                # El espectrograma mel se calcula en la GPU si hay CUDA (ver src/audio.py).
                st.write("**Espectrograma mel**")
                mel_db = mel_spectrogram(waveform)
                st.pyplot(figura_mel(mel_db.numpy(), seconds))

# --- Pestaña 2: qué animal suena en cada ventana de 5 s ---
with tab_tiempo:
    if waveform is None:
        st.info("Sube un audio en la pestaña **Detectar** para ver su línea de tiempo.")
    elif waveform.numel() / TARGET_SR < VENTANA_S + MIN_RESTO_S:
        st.info(
            f"La línea de tiempo necesita un audio de más de {VENTANA_S + MIN_RESTO_S:g} s "
            f"(este dura {waveform.numel() / TARGET_SR:.1f} s)."
        )
    else:
        duracion = waveform.numel() / TARGET_SR
        st.write(f"**Ventanas de {VENTANA_S:g} s, procesadas por lotes en la GPU**")
        with st.spinner("Analizando el audio por ventanas..."):
            segmentos = linea_de_tiempo(waveform)
        st.pyplot(figura_linea_tiempo(segmentos, duracion))
        st.dataframe(
            [
                {
                    "Desde (s)": round(s.inicio, 1),
                    "Hasta (s)": round(s.fin, 1),
                    "Animal": f"{s.animal.emoji} {s.animal.nombre}" if s.animal else "—",
                    "Confianza": f"{s.score:.0%}" if s.animal else "—",
                    "Etiqueta AudioSet": s.etiqueta or "—",
                }
                for s in segmentos
            ],
            hide_index=True,
            width="stretch",
        )

# --- Pestaña 3: benchmark CPU vs GPU (no depende del audio) ---
with tab_bench:
    st.write(
        "Mide cuánto tarda AST en clasificar clips de 10 s en la CPU (fp32) y en la "
        "GPU (fp16), cuántos clips por segundo procesa y cuánta VRAM usa. "
        "Para la versión completa: `uv run python -m src.cuda_bench`."
    )
    if st.button("Ejecutar benchmark"):
        with st.spinner("Midiendo (la CPU tarda un poco)..."):
            st.session_state["benchmark"] = comparar(
                lotes=BENCH_LOTES, repeticiones=3, calentamiento=1
            )
    resultados = st.session_state.get("benchmark")
    if resultados:
        acel = aceleracion(resultados)
        if acel:
            cols = st.columns(len(acel))
            for col, (lote, x) in zip(cols, acel.items()):
                col.metric(f"GPU vs CPU, lote {lote}", f"{x:.1f}x más rápida")
        st.pyplot(figura_benchmark(resultados))
        st.dataframe(
            [
                {
                    "Dispositivo": f"{r.dispositivo} ({r.nombre})",
                    "dtype": r.dtype,
                    "Lote": r.lote,
                    "Latencia (ms)": round(r.latencia_ms, 1),
                    "Clips/s": round(r.clips_por_s, 2),
                    "VRAM modelo (MB)": round(r.vram_modelo_mb) if r.vram_modelo_mb else None,
                    "VRAM pico (MB)": round(r.vram_pico_mb) if r.vram_pico_mb else None,
                }
                for r in resultados
            ],
            hide_index=True,
            width="stretch",
        )
