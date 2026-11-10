"""Generate original multilingual and code calibration text for MoE pilots."""

import argparse
from pathlib import Path
import random


def make_text(seed=803, count=2048):
    rng = random.Random(seed)
    paragraphs = []
    for i in range(count):
        a, b = rng.randint(2, 90), rng.randint(2, 90)
        paragraphs.extend([
            f"A notebook records experiment {i}. The first tray holds {a} small wooden pieces and the second holds {b}. "
            f"Their combined count is {a+b}. Moving a piece between the trays leaves that count unchanged. "
            "Describe the observation, retain the units, and check the written result before storing it.",
            f"实验记录{i}：桌子上有两个盒子，第一个盒子里有{a}个木块，第二个盒子里有{b}个木块。"
            f"一共有{a+b}个木块。把一个木块从一个盒子移到另一个盒子，总数不变。记录观察结果，检查数字和单位。",
            f"def count_items_{i}(first, second):\n    return first + second\n\n"
            f"assert count_items_{i}({a}, {b}) == {a+b}\n"
            "The test checks addition. Another test should check empty containers and repeated calls. "
            "Keep identifiers stable when sorting records or exporting structured results.",
            f"Nel registro {i} ci sono {a} oggetti rossi e {b} oggetti blu. Il totale è {a+b}. "
            "Una scatola vuota è diversa da una scatola non ancora controllata. Conservare questa distinzione "
            "permette di leggere correttamente una tabella e di confrontare le osservazioni.",
            f"Dans le carnet {i}, une boîte contient {a} objets et une autre en contient {b}. "
            f"Le total est {a+b}. Une promenade traverse un jardin calme et rejoint une petite maison. "
            "La description indique les couleurs, les positions et l'ordre des événements.",
            f"Im Protokoll {i} liegen {a} Teile links und {b} Teile rechts. Zusammen sind es {a+b}. "
            "Ein Team prüft die Liste, ordnet die Einträge nach Datum und bewahrt ihre Kennungen auf. "
            "Fehlende Angaben werden gesondert markiert und nicht als beobachtete Werte behandelt.",
            f"En la prueba {i}, una caja contiene {a} piezas y la otra contiene {b}. El total es {a+b}. "
            "Una persona abre la ventana, coloca una taza sobre la mesa y escribe una nota. "
            "La nota conserva el orden de las acciones y describe lo que se puede observar.",
            f"For row {i}, let x={a} and y={b}. Then x+y={a+b}, x*y={a*b}, and x-y={a-b}. "
            "Check an equation by substituting its values. A rectangle's area and perimeter describe "
            "different quantities. Store a length with its units and keep an explanation with the calculation.",
        ])
    return "\n\n".join(paragraphs) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=803)
    p.add_argument("--count", type=int, default=2048)
    a = p.parse_args()
    if a.count < 1:p.error("count must be positive")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(make_text(a.seed, a.count), encoding="utf-8")
