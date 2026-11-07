"""Carga y preprocesado de audio para GuauMiau.

Todo el audio que entra al modelo pasa por :func:`preprocess`, que lo deja en
el mismo formato sin importar de dónde venga (archivo .wav/.mp3 o micrófono):

* tensor ``torch.float32`` de una dimensión ``(num_samples,)``
* mono
* muestreado a :data:`TARGET_SR` (16 kHz)
* valores en el rango ``[-1, 1]``

Los archivos se leen con ``soundfile`` (libsndfile), que soporta WAV, FLAC, OGG
y MP3 sin necesidad de ffmpeg. Para otros formatos (p. ej. .m4a, .webm) se
intenta ``torchaudio.load``, que requiere ``torchcodec`` y ffmpeg instalados.
"""

from __future__ import annotations

from functools import lru_cache
from os import PathLike
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import torchaudio.transforms as AT

TARGET_SR = 16_000

AudioSource = str | PathLike | tuple[int, np.ndarray]


def load_audio(path: str | PathLike) -> tuple[torch.Tensor, int]:
    """Lee un archivo de audio.

    Devuelve ``(waveform, sample_rate)`` con ``waveform`` de forma
    ``(channels, num_samples)`` y tipo ``float32`` en ``[-1, 1]``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de audio: {path}")

    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (frames, ch)
        return torch.from_numpy(data.T.copy()), int(sr)
    except sf.LibsndfileError as sf_err:
        # Formato no soportado por libsndfile: probamos con torchaudio/ffmpeg.
        try:
            import torchaudio

            waveform, sr = torchaudio.load(str(path))
            return waveform.to(torch.float32), int(sr)
        except Exception as ta_err:  # noqa: BLE001
            raise RuntimeError(
                f"No se pudo leer '{path.name}'. soundfile: {sf_err}. "
                f"torchaudio: {ta_err}. Para formatos distintos de "
                "WAV/FLAC/OGG/MP3 instala ffmpeg y torchcodec."
            ) from ta_err


def to_float(samples: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convierte muestras enteras (int16/int32/uint8) o flotantes a float32 en [-1, 1]."""
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    samples = np.asarray(samples)

    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        if info.min == 0:  # sin signo (p. ej. uint8 en WAV de 8 bits)
            mid = (info.max + 1) / 2
            out = (samples.astype(np.float32) - mid) / mid
        else:
            out = samples.astype(np.float32) / max(abs(info.min), info.max)
    else:
        out = samples.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(out))


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Promedia los canales. Acepta ``(samples,)`` o ``(channels, samples)``.

    Devuelve un tensor 1D ``(samples,)``.
    """
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim != 2:
        raise ValueError(f"Se esperaba un tensor 1D o 2D, llegó {tuple(waveform.shape)}")
    return waveform.mean(dim=0)


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int = TARGET_SR) -> torch.Tensor:
    """Remuestrea ``waveform`` de ``orig_sr`` a ``target_sr`` (no-op si coinciden)."""
    if orig_sr <= 0:
        raise ValueError(f"Frecuencia de muestreo inválida: {orig_sr}")
    if orig_sr == target_sr:
        return waveform
    return AF.resample(waveform, orig_freq=orig_sr, new_freq=target_sr)


def preprocess(
    source: AudioSource | np.ndarray | torch.Tensor,
    sr: int | None = None,
    target_sr: int = TARGET_SR,
) -> torch.Tensor:
    """Carga (si hace falta), pasa a mono y remuestrea a ``target_sr``.

    ``source`` puede ser:

    * una ruta a un archivo (``str`` o ``Path``), p. ej. ``.wav`` o ``.mp3``;
    * una tupla ``(sample_rate, array)``, el formato que entrega
      ``gradio.Audio(type="numpy")`` al grabar con el micrófono;
    * un ``np.ndarray`` o ``torch.Tensor`` junto con ``sr``.

    Los arrays pueden ser ``(samples,)``, ``(samples, channels)`` o
    ``(channels, samples)``; se asume que la dimensión más pequeña son los
    canales.

    Devuelve un tensor ``float32`` 1D en ``[-1, 1]`` a ``target_sr``.
    """
    if isinstance(source, (str, PathLike)):
        waveform, sr = load_audio(source)
    else:
        if isinstance(source, tuple):
            sr, source = source
        if sr is None:
            raise ValueError("Falta 'sr' para audio en memoria")
        waveform = to_float(source)
        if waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T  # (samples, channels) -> (channels, samples)

    if waveform.numel() == 0:
        raise ValueError("El audio está vacío")

    waveform = to_mono(waveform)
    waveform = resample(waveform, int(sr), target_sr)
    return waveform.clamp_(-1.0, 1.0).contiguous()


# Parámetros del espectrograma mel. A 16 kHz, un salto de 160 muestras son 10 ms
# por columna; 128 bandas mel es lo mismo que usa el feature extractor de AST.
N_FFT = 1024
HOP_LENGTH = 160
N_MELS = 128


@lru_cache(maxsize=None)
def _mel_transforms(device: str) -> tuple[AT.MelSpectrogram, AT.AmplitudeToDB]:
    """Crea (una sola vez por dispositivo) las transformaciones y las deja en él."""
    mel = AT.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    ).to(device)
    return mel, AT.AmplitudeToDB(stype="power", top_db=80).to(device)


def mel_spectrogram(waveform: torch.Tensor, device: str | None = None) -> torch.Tensor:
    """Espectrograma mel en decibeles de un audio mono a 16 kHz.

    El cálculo (FFT + filtros mel) se hace en ``device``: por defecto la GPU si
    hay CUDA, porque son operaciones sobre matrices que se paralelizan muy bien.
    Devuelve un tensor ``(N_MELS, frames)`` ya en CPU, listo para graficar.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mel, a_db = _mel_transforms(device)
    return a_db(mel(waveform.to(device))).cpu()
