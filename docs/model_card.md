# Ficha del modelo: AST fine-tuned en AudioSet

Resumen de la [model card en Hugging Face](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) para la sustentación.

| Campo | Valor |
|---|---|
| Identificador | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| Arquitectura | Audio Spectrogram Transformer (AST): un ViT aplicado al espectrograma log-mel del audio |
| Parámetros | **86,6 M** (86 594 063, contado con `sum(p.numel())`) |
| Emisor | **MIT** (Massachusetts Institute of Technology). Paper de Yuan Gong, Yu-An Chung y James Glass. La model card la escribió el equipo de Hugging Face, porque los autores no publicaron una |
| Paper | *AST: Audio Spectrogram Transformer*, Interspeech 2021, [arXiv:2104.01778](https://arxiv.org/abs/2104.01778) |
| Código original | <https://github.com/YuanGongND/ast> |
| Licencia | **BSD-3-Clause** (uso comercial y académico permitido si se conserva el aviso de copyright y no se usa el nombre de MIT ni de los autores para promocionar el proyecto) |
| Datos de entrenamiento | AudioSet (clips de 10 s de YouTube etiquetados por Google) |
| Clases | **527** etiquetas de la ontología de AudioSet (multietiqueta) |
| Entrada | Audio mono a **16 kHz**. El feature extractor genera 128 bandas mel y recorta o rellena a **1024 frames (~10,24 s)** |
| Métrica | mAP ≈ **0,4593** en AudioSet eval (es el número que aparece en el nombre; `10-10` es el stride del patch en tiempo y frecuencia) |

## Restricciones y limitaciones

- **La model card no tiene una sección de limitaciones.** La única restricción formal es la licencia BSD-3-Clause.
- Solo distingue las 527 clases de AudioSet. Para animales hay etiquetas como `Dog`, `Bark`, `Cat`, `Meow`, `Bird`, `Cattle, bovinae`, etc. Para cualquier otro animal hay que mapear a la clase más cercana o a la genérica `Animal`.
- Audio de más de ~10 s se **recorta** y solo se analiza el inicio.
- Las salidas son probabilidades **independientes** (sigmoide): no suman 1, y un mismo audio puede puntuar alto en `Animal`, `Dog` y `Bark` a la vez.
- AudioSet viene de YouTube: tiene ruido en las etiquetas y sesgo hacia el contenido de esa plataforma.

## Cómo se usa en GuauMiau

`src/model.py` lo carga una sola vez (en caché). Si hay GPU, corre en **CUDA con fp16**; si no, en **CPU con fp32**. `predict()` devuelve el top-5 de etiquetas con su probabilidad.
