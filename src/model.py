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
    lm = load_model(device)

    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(f"Se esperaba audio mono 1D, llegó forma {waveform.shape}")
    if waveform.size == 0:
        raise ValueError("El audio está vacío")

    inputs = lm.feature_extractor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    features = inputs["input_values"].to(lm.device, dtype=lm.dtype)

    logits = lm.model(input_values=features).logits[0].float()
    probs = torch.sigmoid(logits)

    k = min(top_k, probs.numel())
    scores, indices = probs.topk(k)
    id2label = lm.model.config.id2label
    return [
        {"label": id2label[int(i)], "score": float(s)}
        for s, i in zip(scores.cpu(), indices.cpu())
    ]
