"""Modelo de clasificación de audio para GuauMiau.

Usa el Audio Spectrogram Transformer (AST) ``MIT/ast-finetuned-audioset-10-10-0.4593``
de Hugging Face, ajustado (fine-tuned) sobre AudioSet. Ver ``docs/model_card.md``
para los datos de la model card (parámetros, emisor, licencia, restricciones).

Uso típico::

    from src.audio import preprocess
    from src.model import predict

    waveform = preprocess("perro.wav")      # mono, 16 kHz, float32
    for p in predict(waveform):
        print(p["label"], p["score"])

En GPU el modelo corre en CUDA con fp16; si no hay GPU, en CPU con fp32.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16_000  # frecuencia que espera el feature extractor de AST
TOP_K = 5


@dataclass(frozen=True)
class LoadedModel:
    feature_extractor: ASTFeatureExtractor
    model: ASTForAudioClassification
    device: torch.device
    dtype: torch.dtype


def select_device() -> tuple[torch.device, torch.dtype]:
    """Devuelve ``(cuda, float16)`` si hay GPU disponible; si no, ``(cpu, float32)``."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


@lru_cache(maxsize=None)
def load_model(device: str | None = None) -> LoadedModel:
    """Descarga (la primera vez) y carga el modelo AST y su feature extractor.

    El resultado queda en caché: llamadas posteriores reutilizan el mismo modelo.
    ``device`` permite forzar ``"cpu"`` o ``"cuda"``; por defecto se autodetecta.
    """
    if device is None:
        dev, dtype = select_device()
    else:
        dev = torch.device(device)
        dtype = torch.float16 if dev.type == "cuda" else torch.float32

    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    model = ASTForAudioClassification.from_pretrained(MODEL_ID, dtype=dtype)
    model.to(dev).eval()
    return LoadedModel(feature_extractor, model, dev, dtype)


@torch.inference_mode()
def predict(
    waveform: torch.Tensor | np.ndarray,
    top_k: int = TOP_K,
    device: str | None = None,
) -> list[dict[str, float | str]]:
    """Clasifica un audio y devuelve las ``top_k`` etiquetas más probables.

    ``waveform`` debe ser mono a 16 kHz (la salida de ``src.audio.preprocess``).
    Devuelve una lista ordenada de ``{"label": str, "score": float}``.

    AudioSet es multietiqueta (un audio puede ser a la vez "Dog", "Bark" y
    "Animal"), por eso cada ``score`` es una probabilidad independiente
    (sigmoide) y no suman 1.
    """
    return predict_batch([waveform], top_k=top_k, device=device)[0]


def _as_mono_array(waveform: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(f"Se esperaba audio mono 1D, llegó forma {waveform.shape}")
    if waveform.size == 0:
        raise ValueError("El audio está vacío")
    return waveform


@torch.inference_mode()
def predict_batch(
    waveforms: list[torch.Tensor | np.ndarray],
    top_k: int = TOP_K,
    device: str | None = None,
) -> list[list[dict[str, float | str]]]:
    """Como :func:`predict`, pero para varios audios en una sola pasada.

    Todos los audios se apilan en un lote ``(N, 1024, 128)`` y el modelo los
    procesa a la vez: en la GPU esto aprovecha el paralelismo mucho mejor que
    llamar a ``predict`` N veces. Devuelve una lista de top-k por cada audio,
    en el mismo orden de entrada.
    """
    if not waveforms:
        return []
    lm = load_model(device)
    arrays = [_as_mono_array(w) for w in waveforms]

    # El feature extractor recorta/rellena cada audio a 1024 frames (~10,24 s),
    # así que todos quedan del mismo tamaño y se pueden apilar.
    inputs = lm.feature_extractor(arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    features = inputs["input_values"].to(lm.device, dtype=lm.dtype)

    probs = torch.sigmoid(lm.model(input_values=features).logits.float())  # (N, 527)

    k = min(top_k, probs.shape[-1])
    scores, indices = probs.topk(k, dim=-1)
    scores, indices = scores.cpu().tolist(), indices.cpu().tolist()
    id2label = lm.model.config.id2label
    return [
        [{"label": id2label[i], "score": s} for s, i in zip(fila_s, fila_i)]
        for fila_s, fila_i in zip(scores, indices)
    ]
