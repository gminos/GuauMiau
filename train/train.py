"""Ajuste fino (fine-tuning) de AST con los animales de ESC-50.

Pasos:

1. Descarga ESC-50 (la primera vez) en ``data/`` y extrae solo los clips de las
   10 clases de animales.
2. Remuestrea cada clip de 44.1 kHz a 16 kHz con ``src.audio.preprocess`` y lo
   pasa por el feature extractor de AST (espectrograma log-mel 1024x128).
3. Carga ``MIT/ast-finetuned-audioset-10-10-0.4593`` y cambia su cabeza de 527
   etiquetas de AudioSet por una de 10 animales.
4. Entrena en la GPU con precisión mixta (AMP fp16 + GradScaler) y guarda el
   mejor modelo (según el fold de validación) en ``checkpoints/``.

``data/`` y ``checkpoints/`` están en ``.gitignore``: ni los datos ni los pesos
se suben al repositorio.

Uso::

    uv run python train/train.py                 # 5 épocas, fold 5 de validación
    uv run python train/train.py --epochs 1 --max-steps 20   # prueba rápida

ESC-50 (K. J. Piczak, 2015) tiene licencia CC BY-NC 3.0.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import ASTFeatureExtractor, ASTForAudioClassification

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.audio import TARGET_SR, preprocess  # noqa: E402
from src.model import MODEL_ID  # noqa: E402

ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

# Clases de animales de ESC-50 -> clave en español (la misma de src/animals.py).
CLASES: dict[str, str] = {
    "dog": "perro",
    "rooster": "gallo",
    "pig": "cerdo",
    "cow": "vaca",
    "frog": "rana",
    "cat": "gato",
    "hen": "gallina",
    "insects": "insectos",
    "sheep": "oveja",
    "crow": "cuervo",
}
ETIQUETAS = list(CLASES.values())


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def descargar_esc50(data_dir: Path) -> Path:
    """Descarga ESC-50 y extrae ``meta/esc50.csv`` y los .wav de animales.

    Devuelve la carpeta ``data_dir/ESC-50`` con ``meta/`` y ``audio/``.
    Si ya está extraído, no vuelve a descargar.
    """
    destino = data_dir / "ESC-50"
    csv_path = destino / "meta" / "esc50.csv"
    if csv_path.is_file():
        return destino

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ESC-50-master.zip"
    if not zip_path.is_file():
        print(f"Descargando ESC-50 (~600 MB) desde {ESC50_URL} ...")
        tmp = zip_path.with_suffix(".part")
        with urllib.request.urlopen(ESC50_URL) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            leido = 0
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                leido += len(chunk)
                if total:
                    print(f"\r  {leido / 2**20:6.0f} / {total / 2**20:.0f} MB", end="")
                else:
                    print(f"\r  {leido / 2**20:6.0f} MB", end="")
        print()
        tmp.rename(zip_path)

    print("Extrayendo los clips de animales ...")
    with zipfile.ZipFile(zip_path) as zf:
        meta = zf.read("ESC-50-master/meta/esc50.csv").decode("utf-8")
        filas = [r for r in csv.DictReader(io.StringIO(meta)) if r["category"] in CLASES]
        (destino / "audio").mkdir(parents=True, exist_ok=True)
        for r in filas:
            with zf.open(f"ESC-50-master/audio/{r['filename']}") as src, open(
                destino / "audio" / r["filename"], "wb"
            ) as dst:
                shutil.copyfileobj(src, dst)
        (destino / "meta").mkdir(parents=True, exist_ok=True)
        csv_path.write_text(meta, encoding="utf-8")

    zip_path.unlink()  # ya no hace falta: ocupa ~600 MB
    print(f"  {len(filas)} clips en {destino / 'audio'}")
    return destino


def leer_metadatos(esc50: Path) -> list[dict]:
    with open(esc50 / "meta" / "esc50.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["category"] in CLASES]


def calcular_features(
    filas: list[dict], esc50: Path, fe: ASTFeatureExtractor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remuestrea a 16 kHz y calcula el espectrograma de AST de cada clip."""
    feats, etiquetas = [], []
    for r in filas:
        wav = preprocess(esc50 / "audio" / r["filename"])  # mono, 44.1k -> 16k
        x = fe(wav.numpy(), sampling_rate=TARGET_SR, return_tensors="pt")["input_values"][0]
        feats.append(x)
        etiquetas.append(ETIQUETAS.index(CLASES[r["category"]]))
    return torch.stack(feats), torch.tensor(etiquetas)


def preparar_datos(
    data_dir: Path, fe: ASTFeatureExtractor, val_fold: int
) -> tuple[TensorDataset, TensorDataset]:
    """Devuelve ``(train, val)``. Los features se guardan en caché en ``data_dir``."""
    cache = data_dir / "esc50_animales_ast.pt"
    if cache.is_file():
        datos = torch.load(cache)
    else:
        esc50 = descargar_esc50(data_dir)
        filas = leer_metadatos(esc50)
        print(f"Calculando espectrogramas de {len(filas)} clips (16 kHz) ...")
        x, y = calcular_features(filas, esc50, fe)
        datos = {"x": x, "y": y, "fold": torch.tensor([int(r["fold"]) for r in filas])}
        torch.save(datos, cache)

    es_val = datos["fold"] == val_fold
    train = TensorDataset(datos["x"][~es_val], datos["y"][~es_val])
    val = TensorDataset(datos["x"][es_val], datos["y"][es_val])
    return train, val


# --------------------------------------------------------------------------- #
# Modelo y entrenamiento
# --------------------------------------------------------------------------- #
def cargar_modelo() -> ASTForAudioClassification:
    """AST de AudioSet con una cabeza nueva de 10 animales."""
    return ASTForAudioClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(ETIQUETAS),
        id2label=dict(enumerate(ETIQUETAS)),
        label2id={e: i for i, e in enumerate(ETIQUETAS)},
        ignore_mismatched_sizes=True,  # la cabeza de 527 salidas se descarta
    )


@torch.no_grad()
def evaluar(model, loader, device, amp: bool) -> tuple[float, float]:
    model.eval()
    perdida, aciertos, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(input_values=x, labels=y)
        perdida += out.loss.item() * len(y)
        aciertos += (out.logits.argmax(-1) == y).sum().item()
        n += len(y)
    return perdida / n, aciertos / n


def entrenar(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"  # AMP fp16 solo tiene sentido en la GPU
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  (AMP fp16)")
    else:
        print("Aviso: no hay CUDA, se entrena en CPU sin AMP (será muy lento).")

    fe = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    train_ds, val_ds = preparar_datos(args.data_dir, fe, args.val_fold)
    print(f"Clips: {len(train_ds)} de entrenamiento, {len(val_ds)} de validación "
          f"(fold {args.val_fold})")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          pin_memory=amp, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, pin_memory=amp)

    model = cargar_modelo().to(device)  # pesos en fp32; autocast decide qué va en fp16
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # La cabeza es nueva: aprende más rápido que el resto del transformer.
    cabeza = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    cuerpo = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    optim = torch.optim.AdamW(
        [{"params": cuerpo, "lr": args.lr}, {"params": cabeza, "lr": args.lr * 10}],
        weight_decay=0.01,
    )
    pasos_totales = args.epochs * (len(train_dl) // args.accum)
    if args.max_steps:
        pasos_totales = min(pasos_totales, args.max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=[args.lr, args.lr * 10], total_steps=max(pasos_totales, 1),
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mejor_acc, paso = -1.0, 0
    if amp:
        torch.cuda.reset_peak_memory_stats()

    for epoca in range(1, args.epochs + 1):
        model.train()
        t0, perdida_acum = time.perf_counter(), 0.0
        optim.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = model(input_values=x, labels=y).loss / args.accum
            scaler.scale(loss).backward()
            perdida_acum += loss.item()

            if i % args.accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
                paso += 1
                if paso % 10 == 0:
                    print(f"  época {epoca} paso {paso}/{pasos_totales} "
                          f"pérdida {perdida_acum / 10:.4f}")
                    perdida_acum = 0.0
                if args.max_steps and paso >= args.max_steps:
                    break

        val_loss, val_acc = evaluar(model, val_dl, device, amp)
        seg = time.perf_counter() - t0
        mem = f", VRAM pico {torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if amp else ""
        print(f"Época {epoca}: val pérdida {val_loss:.4f}, val exactitud {val_acc:.1%} "
              f"({seg:.0f} s{mem})")

        if val_acc > mejor_acc:
            mejor_acc = val_acc
            model.save_pretrained(args.out_dir)
            fe.save_pretrained(args.out_dir)
            print(f"  -> guardado en {args.out_dir}")

        if args.max_steps and paso >= args.max_steps:
            break

    print(f"Mejor exactitud de validación: {mejor_acc:.1%}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", type=Path, default=RAIZ / "data")
    p.add_argument("--out-dir", type=Path, default=RAIZ / "checkpoints" / "ast-esc50-animales")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4, help="4 cabe en una GPU de 4 GB")
    p.add_argument("--accum", type=int, default=4,
                   help="pasos de acumulación de gradiente (lote efectivo = batch * accum)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--val-fold", type=int, default=5, choices=range(1, 6))
    p.add_argument("--max-steps", type=int, default=0, help="cortar tras N pasos (pruebas)")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="menos VRAM a cambio de ~30%% más tiempo")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    entrenar(parse_args())
