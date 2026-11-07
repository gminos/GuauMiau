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
from src.model import load_model, predict, select_device
from src.plots import figura_mel, figura_onda

st.set_page_config(page_title="GuauMiau", page_icon="🐶")

# Se piden más etiquetas de las que se muestran: un animal puede estar descrito
# por varias (Dog, Bark...) y, si se corta en 5, alguna podría quedar fuera.
ETIQUETAS_A_PEDIR = 30
# Por debajo de esta probabilidad no afirmamos qué animal es.
UMBRAL = 0.10


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


st.title("GuauMiau 🐶🐱")
st.write("Sube un audio o grábalo con el micrófono y te digo qué suena.")

# --- Dispositivo: la app avisa si usa GPU (CUDA) o CPU ---
device, dtype = select_device()
if device.type == "cuda":
    st.success(f"Usando GPU: {torch.cuda.get_device_name(0)} (CUDA, fp16)")
else:
    st.warning("No se detectó GPU: se usa la CPU (fp32). Funciona, pero más lento.")

# --- Entrada de audio: archivo o micrófono ---
tab_file, tab_mic = st.tabs(["📁 Subir archivo", "🎤 Grabar"])
with tab_file:
    uploaded = st.file_uploader("Archivo de audio", type=["wav", "mp3"])
with tab_mic:
    recorded = st.audio_input("Graba un sonido")

audio_file = uploaded or recorded

if audio_file is not None:
    st.audio(audio_file)
    get_model()  # fuerza la carga (con spinner) antes de clasificar
    with st.spinner("Analizando..."):
        try:
            waveform, top = classify(audio_file)
        except (ValueError, RuntimeError, FileNotFoundError) as err:
            st.error(f"No se pudo procesar el audio: {err}")
            st.stop()

    seconds = waveform.numel() / TARGET_SR
    if seconds > 10.24:
        st.info(f"El audio dura {seconds:.1f} s; el modelo solo analiza los primeros ~10 s.")

    animales = agrupar(top)
    if not animales or animales[0]["score"] < UMBRAL:
        st.subheader("No parece un animal que conozca 🤔")
    else:
        mejor = animales[0]
        animal = mejor["animal"]
        st.subheader(f"{animal.nombre} {animal.emoji} ({mejor['score']:.0%})")
        if animal.foto:
            st.image(str(animal.foto), width=300)

        st.write("**Top-5 de animales** (cada probabilidad es independiente, no suman 100 %):")
        for r in animales[:5]:
            st.progress(
                min(r["score"], 1.0),
                text=f"{r['animal'].emoji} {r['animal'].nombre}: {r['score']:.1%}",
            )

    with st.expander("Etiquetas originales del modelo (AudioSet)"):
        for item in top[:5]:
            st.write(f"{item['label']}: {item['score']:.1%}")

    # --- Parte visual: lo que el modelo "ve" ---
    st.write("**Forma de onda**")
    st.pyplot(figura_onda(waveform.numpy(), TARGET_SR))

    # El espectrograma mel se calcula en la GPU si hay CUDA (ver src/audio.py).
    st.write("**Espectrograma mel**")
    mel_db = mel_spectrogram(waveform)
    st.pyplot(figura_mel(mel_db.numpy(), seconds))
