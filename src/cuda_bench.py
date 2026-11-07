"""Benchmark de AST en CPU contra GPU: latencia, throughput y VRAM.

Mide lo mismo en los dos dispositivos, con la misma entrada:

* **Latencia**: milisegundos que tarda un lote en pasar por el modelo.
* **Throughput**: clips de 10 s clasificados por segundo (``lote / latencia``).
* **VRAM** (solo GPU): memoria de CUDA que ocupan los pesos del modelo y el
  pico durante la inferencia, según ``torch.cuda.memory_allocated`` y
  ``torch.cuda.max_memory_allocated``.

Cada dispositivo usa su configuración real de la app (ver ``src/model.py``):
CPU con fp32 y GPU con fp16. Antes de medir se hacen unas pasadas de
calentamiento, porque la primera llamada en CUDA incluye inicializar kernels y
no representa el tiempo normal. En GPU se llama a ``torch.cuda.synchronize()``
antes de parar el cronómetro: CUDA es asíncrono y, sin eso, se mediría solo el
tiempo de encolar el trabajo.

La entrada es ruido blanco de 10 s pasado por el feature extractor: el modelo
siempre recibe un espectrograma del mismo tamaño (1024 x 128), así que el
contenido del audio no cambia el tiempo.

Uso::

    uv run python -m src.cuda_bench            # imprime la tabla y guarda la gráfica
    uv run python -m src.cuda_bench --lotes 1 4 16 --repeticiones 10
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from src.model import MODEL_ID, SAMPLE_RATE

DURACION_S = 10.0
LOTES = (1, 8)
REPETICIONES = 5
CALENTAMIENTO = 2
MB = 1024 ** 2


@dataclass(frozen=True)
class Resultado:
    dispositivo: str  # "CPU" o "GPU"
    nombre: str  # nombre del procesador o de la GPU
    dtype: str
    lote: int
    latencia_ms: float  # mediana de las repeticiones, por lote
    latencia_std_ms: float
    clips_por_s: float
    vram_modelo_mb: float | None  # None en CPU
    vram_pico_mb: float | None


def _nombre_cpu() -> str:
    import platform

    return platform.processor() or platform.machine() or "CPU"


def entrada(lote: int, fe: ASTFeatureExtractor | None = None) -> torch.Tensor:
    """Lote de espectrogramas ``(lote, 1024, 128)`` hechos con ruido de 10 s."""
    fe = fe or ASTFeatureExtractor.from_pretrained(MODEL_ID)
    rng = np.random.default_rng(0)
    ondas = [
        rng.uniform(-0.5, 0.5, int(DURACION_S * SAMPLE_RATE)).astype(np.float32)
        for _ in range(lote)
    ]
    return fe(ondas, sampling_rate=SAMPLE_RATE, return_tensors="pt")["input_values"]


@torch.inference_mode()
def medir_dispositivo(
    device: str,
    lotes: tuple[int, ...] = LOTES,
    repeticiones: int = REPETICIONES,
    calentamiento: int = CALENTAMIENTO,
) -> list[Resultado]:
    """Carga AST en ``device`` ("cpu" o "cuda") y mide cada tamaño de lote."""
    dev = torch.device(device)
    es_gpu = dev.type == "cuda"
    if es_gpu and not torch.cuda.is_available():
        raise RuntimeError("No hay GPU con CUDA disponible")
    dtype = torch.float16 if es_gpu else torch.float32
    fe = ASTFeatureExtractor.from_pretrained(MODEL_ID)

    # La app puede tener ya su propio modelo en la GPU: se mide solo lo que
    # añade este, restando la memoria que había antes de cargarlo.
    if es_gpu:
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
    model = ASTForAudioClassification.from_pretrained(MODEL_ID, dtype=dtype).to(dev).eval()
    vram_modelo = (torch.cuda.memory_allocated() - base) / MB if es_gpu else None

    def sincronizar():
        if es_gpu:
            torch.cuda.synchronize()

    resultados = []
    try:
        for lote in lotes:
            x = entrada(lote, fe).to(dev, dtype=dtype)
            for _ in range(calentamiento):
                model(input_values=x)
            sincronizar()
            if es_gpu:
                torch.cuda.reset_peak_memory_stats()

            tiempos = []
            for _ in range(repeticiones):
                t0 = time.perf_counter()
                model(input_values=x)
                sincronizar()
                tiempos.append(time.perf_counter() - t0)

            latencia = float(np.median(tiempos))
            resultados.append(Resultado(
                dispositivo="GPU" if es_gpu else "CPU",
                nombre=torch.cuda.get_device_name() if es_gpu else _nombre_cpu(),
                dtype=str(dtype).removeprefix("torch."),
                lote=lote,
                latencia_ms=latencia * 1000,
                latencia_std_ms=float(np.std(tiempos)) * 1000,
                clips_por_s=lote / latencia,
                vram_modelo_mb=vram_modelo,
                vram_pico_mb=(torch.cuda.max_memory_allocated() - base) / MB if es_gpu else None,
            ))
            del x
    finally:
        del model
        if es_gpu:
            torch.cuda.empty_cache()
    return resultados


def comparar(
    lotes: tuple[int, ...] = LOTES,
    repeticiones: int = REPETICIONES,
    calentamiento: int = CALENTAMIENTO,
) -> list[Resultado]:
    """Mide en CPU y, si hay CUDA, también en GPU."""
    resultados = medir_dispositivo("cpu", lotes, repeticiones, calentamiento)
    if torch.cuda.is_available():
        resultados += medir_dispositivo("cuda", lotes, repeticiones, calentamiento)
    return resultados


def aceleracion(resultados: list[Resultado]) -> dict[int, float]:
    """Cuántas veces más rápida es la GPU que la CPU, por tamaño de lote."""
    cpu = {r.lote: r.latencia_ms for r in resultados if r.dispositivo == "CPU"}
    gpu = {r.lote: r.latencia_ms for r in resultados if r.dispositivo == "GPU"}
    return {lote: cpu[lote] / gpu[lote] for lote in cpu if lote in gpu}


def imprimir(resultados: list[Resultado]) -> None:
    print(f"{'Disp.':<5} {'dtype':<8} {'Lote':>4} {'Latencia (ms)':>16} "
          f"{'Clips/s':>9} {'VRAM modelo':>12} {'VRAM pico':>10}")
    for r in resultados:
        vram = (f"{r.vram_modelo_mb:>9.0f} MB {r.vram_pico_mb:>7.0f} MB"
                if r.vram_modelo_mb is not None else f"{'—':>12} {'—':>10}")
        print(f"{r.dispositivo:<5} {r.dtype:<8} {r.lote:>4} "
              f"{r.latencia_ms:>9.1f} ± {r.latencia_std_ms:<4.1f} {r.clips_por_s:>9.2f} {vram}")
    for lote, x in aceleracion(resultados).items():
        print(f"Lote {lote}: la GPU es {x:.1f}x más rápida que la CPU")


def parse_args() -> argparse.Namespace:
    raiz = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--lotes", type=int, nargs="+", default=list(LOTES))
    p.add_argument("--repeticiones", type=int, default=REPETICIONES)
    p.add_argument("--calentamiento", type=int, default=CALENTAMIENTO)
    p.add_argument("--out-dir", type=Path, default=raiz / "docs" / "resultados")
    return p.parse_args()


if __name__ == "__main__":
    from src.plots import figura_benchmark

    args = parse_args()
    resultados = comparar(tuple(args.lotes), args.repeticiones, args.calentamiento)
    imprimir(resultados)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir / "benchmark.png"
    figura_benchmark(resultados).savefig(png, dpi=150)
    js = args.out_dir / "benchmark.json"
    js.write_text(json.dumps(
        {"resultados": [asdict(r) for r in resultados],
         "aceleracion_gpu": aceleracion(resultados)},
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")
    print(f"\nGuardado: {png} y {js}")
