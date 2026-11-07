"""Mapeo de etiquetas de AudioSet a animales en español, con su foto.

AST devuelve 527 etiquetas de AudioSet y varias describen al mismo animal:
un ladrido claro puntúa alto en ``Bark`` y más bajo en ``Dog``. Para que la app
diga "Perro" y no "Bark", cada animal reúne las etiquetas que lo describen
y se queda con la probabilidad más alta de ellas.

Se usa el máximo (y no la suma) porque AudioSet es multietiqueta: cada
probabilidad es independiente, así que sumarlas podría pasar de 1.

Los nombres de las etiquetas son exactamente los de ``config.id2label`` del
modelo ``MIT/ast-finetuned-audioset-10-10-0.4593``.

Las etiquetas genéricas (``Animal``, ``Domestic animals, pets``,
``Livestock...``, ``Wild animals``, ``Fowl``) se ignoran a propósito: no dicen
qué animal es.

Fotos: ``assets/animals/<clave>.jpg``. Sus licencias y autores están en
``assets/animals/CREDITS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CARPETA_FOTOS = Path(__file__).resolve().parent.parent / "assets" / "animals"


@dataclass(frozen=True)
class Animal:
    clave: str  # también es el nombre del archivo de la foto
    nombre: str  # nombre en español
    emoji: str
    etiquetas: tuple[str, ...]  # etiquetas de AudioSet que lo describen

    @property
    def foto(self) -> Path | None:
        """Ruta de la foto, o ``None`` si el archivo no existe."""
        ruta = CARPETA_FOTOS / f"{self.clave}.jpg"
        return ruta if ruta.is_file() else None


ANIMALES: tuple[Animal, ...] = (
    Animal("perro", "Perro", "🐶", (
        "Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)",
        "Canidae, dogs, wolves",
    )),
    Animal("gato", "Gato", "🐱", ("Cat", "Purr", "Meow", "Hiss", "Caterwaul")),
    Animal("gallo", "Gallo", "🐓", ("Crowing, cock-a-doodle-doo",)),
    Animal("gallina", "Gallina", "🐔", ("Chicken, rooster", "Cluck")),
    Animal("vaca", "Vaca", "🐄", ("Cattle, bovinae", "Moo")),
    Animal("cerdo", "Cerdo", "🐷", ("Pig", "Oink")),
    Animal("oveja", "Oveja", "🐑", ("Sheep", "Bleat")),
    Animal("cabra", "Cabra", "🐐", ("Goat",)),
    Animal("caballo", "Caballo", "🐴", ("Horse", "Clip-clop", "Neigh, whinny")),
    Animal("pato", "Pato", "🦆", ("Duck", "Quack")),
    Animal("pajaro", "Pájaro", "🐦", (
        "Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet",
        "Squawk", "Pigeon, dove", "Coo", "Bird flight, flapping wings",
    )),
    Animal("cuervo", "Cuervo", "🐦‍⬛", ("Crow", "Caw")),
    Animal("buho", "Búho", "🦉", ("Owl", "Hoot")),
    Animal("rana", "Rana", "🐸", ("Frog", "Croak")),
    Animal("insectos", "Insectos", "🐝", (
        "Insect", "Cricket", "Mosquito", "Fly, housefly", "Buzz", "Bee, wasp, etc.",
    )),
    Animal("leon", "León", "🦁", ("Roaring cats (lions, tigers)", "Roar")),
    Animal("ballena", "Ballena", "🐋", ("Whale vocalization",)),
)

# Búsqueda rápida: etiqueta de AudioSet -> animal. Una etiqueta no puede
# pertenecer a dos animales, y este assert lo comprueba al importar.
_POR_ETIQUETA: dict[str, Animal] = {}
for _animal in ANIMALES:
    for _etiqueta in _animal.etiquetas:
        assert _etiqueta not in _POR_ETIQUETA, f"Etiqueta repetida: {_etiqueta}"
        _POR_ETIQUETA[_etiqueta] = _animal


def animal_de_etiqueta(etiqueta: str) -> Animal | None:
    """Devuelve el animal al que pertenece una etiqueta de AudioSet, si tiene."""
    return _POR_ETIQUETA.get(etiqueta)


def agrupar(predicciones: list[dict]) -> list[dict]:
    """Convierte etiquetas de AudioSet en animales, ordenados de mayor a menor.

    ``predicciones`` es la salida de ``src.model.predict``: una lista de
    ``{"label": str, "score": float}``. Para que un animal no quede fuera por
    cortar el top demasiado pronto, conviene pedirle a ``predict`` más
    etiquetas que las 5 que se muestran (por ejemplo ``top_k=30``).

    Devuelve ``[{"animal": Animal, "score": float, "etiqueta": str}, ...]``,
    donde ``etiqueta`` es la etiqueta original que dio la mejor probabilidad.
    Las etiquetas que no son de un animal conocido se descartan.
    """
    mejores: dict[str, dict] = {}
    for p in predicciones:
        animal = animal_de_etiqueta(p["label"])
        if animal is None:
            continue
        actual = mejores.get(animal.clave)
        if actual is None or p["score"] > actual["score"]:
            mejores[animal.clave] = {
                "animal": animal,
                "score": p["score"],
                "etiqueta": p["label"],
            }
    return sorted(mejores.values(), key=lambda r: r["score"], reverse=True)
