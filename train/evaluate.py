"""Precisión y matriz de confusión de AST antes y después del ajuste fino.

Compara, sobre los mismos clips de validación de ESC-50 (el fold que no se usó
para entrenar), dos modelos:

* **Antes**: ``MIT/ast-finetuned-audioset-10-10-0.4593`` tal cual, con sus 527
  etiquetas de AudioSet. Cada animal toma la probabilidad más alta de sus
  etiquetas (las de ``src/animals.py``, igual que la app) y se predice el
  animal con mayor puntaje entre los 10 de ESC-50.
* **Después**: el modelo ajustado por ``train/train.py`` (``checkpoints/``),
  con su cabeza de 10 animales.

Imprime la exactitud de cada uno y guarda en ``docs/resultados/``:

* ``matriz_confusion.png``: las dos matrices de confusión lado a lado.
* ``metricas.json``: exactitud global, exactitud por animal y las matrices.

Uso (después de entrenar)::

    uv run python train/train.py
    uv run python train/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, TensorDataset
from transformers import ASTFeatureExtractor, ASTForAudioClassification

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "train"))

from src.animals import ANIMALES  # noqa: E402
from src.model import MODEL_ID  # noqa: E402
from train import ETIQUETAS, preparar_datos  # noqa: E402

NOMBRES = {a.clave: a.nombre for a in ANIMALES}


@torch.no_grad()
def predecir(model, loader, device, amp: bool, columnas=None) -> np.ndarray:
    """Devuelve la clase predicha (índice en ``ETIQUETAS``) de cada clip.

    Sin ``columnas``, el modelo ya tiene 10 salidas y se toma el argmax.
    Con ``columnas`` (una lista de índices de AudioSet por animal), se aplica
    la sigmoide a las 527 salidas y cada animal se queda con el máximo de las
    suyas, como hace ``src.animals.agrupar`` en la app.
    """
    model.eval()
    preds = []
    for (x,) in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(input_values=x).logits.float()
        if columnas is not None:
            probs = torch.sigmoid(logits)
            logits = torch.stack([probs[:, c].amax(dim=1) for c in columnas], dim=1)
        preds.append(logits.argmax(-1).cpu())
    return torch.cat(preds).numpy()


def columnas_audioset(model) -> list[list[int]]:
    """Índices de AudioSet de cada animal de ``ETIQUETAS``, según ``src/animals.py``."""
    label2id = {v: int(k) for k, v in model.config.id2label.items()}
    por_clave = {a.clave: a for a in ANIMALES}
    columnas = []
    for clave in ETIQUETAS:
        faltan = [e for e in por_clave[clave].etiquetas if e not in label2id]
        if faltan:
            raise KeyError(f"Etiquetas de {clave} que no están en el modelo: {faltan}")
        columnas.append([label2id[e] for e in por_clave[clave].etiquetas])
    return columnas


def matriz_confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Filas = clase real, columnas = clase predicha."""
    n = len(ETIQUETAS)
    m = np.zeros((n, n), dtype=int)
    np.add.at(m, (y, pred), 1)
    return m


def figura_matrices(matrices: dict[str, np.ndarray], exactitudes: dict[str, float]) -> Figure:
    nombres = [NOMBRES[c] for c in ETIQUETAS]
    fig = Figure(figsize=(13, 5.8), layout="constrained")
    axes = fig.subplots(1, len(matrices))
    vmax = max(m.sum(axis=1).max() for m in matrices.values())
    for ax, (titulo, m) in zip(axes, matrices.items()):
        img = ax.imshow(m, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(f"{titulo}: exactitud {exactitudes[titulo]:.1%}")
        ax.set_xticks(range(len(nombres)), nombres, rotation=45, ha="right")
        ax.set_yticks(range(len(nombres)), nombres)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i, j]:
                    ax.text(j, i, m[i, j], ha="center", va="center", fontsize=9,
                            color="white" if m[i, j] > vmax / 2 else "black")
    fig.colorbar(img, ax=axes, shrink=0.8, label="Clips")
    return fig


def evaluar_modelos(args: argparse.Namespace) -> None:
    if not (args.checkpoint / "config.json").is_file():
        sys.exit(f"No hay modelo ajustado en {args.checkpoint}.\n"
                 "Primero entrena con: uv run python train/train.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    print(f"Dispositivo: {torch.cuda.get_device_name(0) if amp else 'CPU'}")

    fe = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    _, val_ds = preparar_datos(args.data_dir, fe, args.val_fold)
    y = val_ds.tensors[1].numpy()
    loader = DataLoader(TensorDataset(val_ds.tensors[0]),
                        batch_size=args.batch_size, pin_memory=amp)
    print(f"Clips de validación: {len(y)} (fold {args.val_fold})")

    preds: dict[str, np.ndarray] = {}

    model = ASTForAudioClassification.from_pretrained(MODEL_ID).to(device)
    preds["Antes (AudioSet)"] = predecir(model, loader, device, amp, columnas_audioset(model))
    del model
    if amp:
        torch.cuda.empty_cache()

    model = ASTForAudioClassification.from_pretrained(args.checkpoint).to(device)
    preds["Después (ajuste fino)"] = predecir(model, loader, device, amp)
    del model

    matrices = {k: matriz_confusion(y, p) for k, p in preds.items()}
    exactitudes = {k: float((p == y).mean()) for k, p in preds.items()}

    print(f"\n{'Animal':<10}" + "".join(f"{k:>24}" for k in preds))
    for i, clave in enumerate(ETIQUETAS):
        fila = "".join(f"{m[i, i] / m[i].sum():>24.0%}" for m in matrices.values())
        print(f"{NOMBRES[clave]:<10}{fila}")
    print(f"{'Total':<10}" + "".join(f"{a:>24.1%}" for a in exactitudes.values()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir / "matriz_confusion.png"
    figura_matrices(matrices, exactitudes).savefig(png, dpi=150)

    metricas = {
        "clips_validacion": int(len(y)),
        "fold_validacion": args.val_fold,
        "etiquetas": [NOMBRES[c] for c in ETIQUETAS],
        "modelos": {
            k: {
                "exactitud": round(exactitudes[k], 4),
                "exactitud_por_animal": {
                    NOMBRES[c]: round(float(m[i, i] / m[i].sum()), 4)
                    for i, c in enumerate(ETIQUETAS)
                },
                "matriz_confusion": m.tolist(),
            }
            for k, m in matrices.items()
        },
    }
    js = args.out_dir / "metricas.json"
    js.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado: {png} y {js}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", type=Path, default=RAIZ / "data")
    p.add_argument("--checkpoint", type=Path,
                   default=RAIZ / "checkpoints" / "ast-esc50-animales")
    p.add_argument("--out-dir", type=Path, default=RAIZ / "docs" / "resultados")
    p.add_argument("--val-fold", type=int, default=5, choices=range(1, 6),
                   help="debe ser el mismo fold de validación usado al entrenar")
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    evaluar_modelos(parse_args())
