"""Generate original multilingual calibration text for text models."""

import argparse
import json
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


def make_diverse_text(seed=617029, count=6000):
    """Generate original prose, code and structured multilingual text."""
    rng = random.Random(seed)
    objects = ['ceramic bowl', 'paper lantern', 'wooden bridge', 'glass bottle', 'cotton bag', 'stone marker', 'brass bell', 'woven basket', 'garden gate', 'blue notebook', 'water tank', 'bicycle wheel', 'clay tile', 'canvas cover', 'metal shelf', 'linen curtain']
    places = ['harbor', 'library', 'orchard', 'workshop', 'courtyard', 'greenhouse', 'village square', 'riverside path', 'storage room', 'observatory', 'mountain shelter', 'community garden']
    people = ['Mira', 'Leon', 'Sofia', 'Emil', 'Nora', 'Tao', 'Amir', 'Lina', 'Iris', 'Hugo', 'Ravi', 'Alba']
    colors = ['red', 'green', 'blue', 'yellow', 'white', 'orange', 'violet', 'brown']
    notes = ['An index helps locate entries without scanning every record. It costs storage and must be updated when indexed values change. Choose the fields that match the actual search pattern. A lookup by one key and a range query over a different key need not benefit from the same ordering. Measure the complete operation, including updates, rather than judging only one convenient example.', 'A queue processes pending items in arrival order. A stack processes the most recently added item first. Both can represent unfinished work, but the removal rule changes the resulting traversal. When a task fails, retain its identifier and reason. Retrying should preserve the distinction between work that has never started and work that has already produced an external effect.', 'A coordinate describes a location relative to a chosen reference. Translation changes the origin, while scaling changes distances expressed in those coordinates. A transformation and its inverse should reconstruct the initial point within numerical precision. Check the units before combining measurements. A number written in centimeters cannot be added directly to a number written in meters without conversion.', 'A cache keeps values that may be requested again. Every entry needs a key that identifies the computation and its inputs. When an input changes, either the key must change or the old value must be invalidated. The cost of looking up an entry and storing the result is part of the total cost. Reuse is helpful when those costs are lower than recomputing.', 'A stable sort preserves the order of entries whose comparison keys are equal. This matters when one ordering is refined by another ordering. A missing value should have an explicit position in the ordering rather than silently becoming zero. Test empty input, repeated keys, negative numbers and values that contain leading or trailing spaces.', 'A configuration file records choices explicitly. Defaults make short examples convenient, but a saved artifact should also record the effective choices used to create it. A reader can then distinguish an omitted field from a field that was deliberately set. Validate required fields before starting a long operation, and identify the field when reporting a parsing error.', 'A balanced set of examples includes both common cases and cases that exercise different branches. Repeating one simple pattern many times does not necessarily introduce new information. Keep evaluation examples separate from examples used to choose parameters. When the two collections share a source, select nonoverlapping records and preserve the selection procedure.', 'A map legend explains symbols and measurement scales. Two maps can show the same region while using different projections or levels of detail. Record the coordinate system with the data. When simplifying a path, preserve the features required by the intended use. A short route and a route with fewer turns optimize different objectives.', 'An observation records what was measured, when it was measured and the instrument settings. A hypothesis proposes an explanation that can be checked against additional observations. Keep uncertain readings distinct from missing readings. A repeated reading under the same conditions helps characterize variability, while a changed condition tests a different question.', 'A directed graph has vertices connected by ordered edges. A path follows those directions. A cycle returns to an earlier vertex. Searching a graph with a visited set prevents a traversal from repeatedly processing the same vertex. If a task depends on another task, check for cycles before attempting to schedule all tasks in dependency order.']
    codes = ['def group_by_key(rows, key):\n    groups = {}\n    for row in rows:\n        groups.setdefault(row[key], []).append(row)\n    return groups\n', 'def deduplicate(items):\n    seen = set()\n    result = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result\n', 'def moving_average(values, window):\n    if window < 1:\n        raise ValueError("window must be positive")\n    return [sum(values[i:i + window]) / window\n            for i in range(len(values) - window + 1)]\n', 'def merge_counts(first, second):\n    merged = dict(first)\n    for key, count in second.items():\n        merged[key] = merged.get(key, 0) + count\n    return merged\n', 'def transpose(rows):\n    if not rows:\n        return []\n    width = len(rows[0])\n    if any(len(row) != width for row in rows):\n        raise ValueError("rows must have equal length")\n    return [list(column) for column in zip(*rows)]\n', 'def reachable(graph, start):\n    pending = [start]\n    seen = set()\n    while pending:\n        node = pending.pop()\n        if node in seen:\n            continue\n        seen.add(node)\n        pending.extend(graph.get(node, []))\n    return seen\n', 'def normalize_labels(labels):\n    return [label.strip().casefold() for label in labels if label.strip()]\n', 'def partition(values, threshold):\n    low, high = [], []\n    for value in values:\n        (low if value < threshold else high).append(value)\n    return low, high\n']
    paragraphs = []
    for i in range(count):
        (a, b, c) = (rng.randint(3, 190), rng.randint(3, 120), rng.randint(2, 45))
        (person, other) = rng.sample(people, 2)
        (obj, other_obj) = rng.sample(objects, 2)
        place = rng.choice(places)
        color = rng.choice(colors)
        kind = i % 12
        if kind == 0:
            text = f'{person} arrived at the {place} before the afternoon rain. Near the entrance stood a {color} {obj}. {other} had left a handwritten note beside a {other_obj}, asking visitors to use the side entrance while the main passage was being repaired. They moved the loose objects away from the doorway, compared the labels with the delivery list, and arranged a sheltered place for the next arrival. By evening the passage was clear. The note remained visible so that a person arriving later would understand the change.'
        elif kind == 1:
            text = f'Question {i}: A rectangular garden measures {a} meters by {b} meters. A straight fence covers its boundary. What are its area and perimeter? Answer: Multiply length by width to obtain {a * b} square meters. Add the two side lengths and multiply by two to obtain {2 * (a + b)} meters of boundary. The quantities use different units because one measures a surface and the other measures a length. If {c} equal gardens are built, their combined area is {c * a * b} square meters.'
        elif kind == 2:
            code = rng.choice(codes)
            text = f'Programming example {i}. Implement a small transformation with explicit handling of empty input.\n```python\n{code}```\nExplanation: Keep the input collection intact unless the function explicitly promises to modify it. Test repeated values and boundary conditions. Use the returned result rather than assuming that the input has changed. Document whether ordering is significant. A descriptive exception makes invalid input easier to diagnose than a partial result.'
        elif kind == 3:
            rows = [dict(id=i * 10 + j, category=rng.choice(['tools', 'supplies', 'parts', 'books']), quantity=rng.randint(0, 120), verified=bool(j % 2)) for j in range(4)]
            text = f'Structured record {i}:\n' + json.dumps(dict(location=place, owner=person, items=rows), ensure_ascii=False, indent=2) + f"\nThe quantities sum to {sum((row['quantity'] for row in rows))}. Preserve integer quantities and boolean verification fields when exporting this record. An unchecked record is not the same as a record with zero quantity. Keep identifiers stable if the list is sorted by category."
        elif kind == 4:
            text = f'Technical note {i}: ' + rng.choice(notes) + f'\nExample input lengths are {a}, {b} and {c}. Record each case separately before combining the observations. Describe both the expected result and the assumptions under which it is valid.'
        elif kind == 5:
            text = f'User: We need to organize a visit to the {place}. There are {a} visitors, {b} seats in each room and {c} available rooms. Can everyone sit down?\nAssistant: The total seating capacity is {b * c}. ' + ('There is enough capacity.' if b * c >= a else 'The capacity is insufficient for a single session.') + f' Compare capacity with attendance before assigning rooms. Ask whether visitors must remain together, reserve accessible routes, and provide a written schedule. If multiple sessions are necessary, identify the changeover time clearly.'
        elif kind == 6:
            text = f'Appunto {i}. {person} controlla il registro della biblioteca e trova {a} richieste da ordinare. Ogni richiesta conserva una data, un codice e una breve descrizione. Prima si verificano i campi obbligatori, poi si raggruppano le richieste per argomento. Le voci incomplete vengono conservate in un elenco separato. Un riepilogo deve distinguere ciò che è stato completato da ciò che richiede ancora una risposta. Non bisogna confondere una descrizione mancante con una descrizione vuota scelta intenzionalmente.'
        elif kind == 7:
            text = f'Note {i}. Dans un atelier, {person} prépare un carnet pour suivre les réparations. Chaque objet reçoit un identifiant, une description et une date de réception. On conserve les observations initiales même si une explication change ensuite. Avant de rendre un objet, une autre personne vérifie les étapes terminées. Le résumé indique les pièces remplacées et les points qui restent à vérifier. Une estimation de durée doit être accompagnée de ses hypothèses plutôt que présentée comme une garantie.'
        elif kind == 8:
            text = f'Bericht {i}. Im Gemeinschaftsgarten werden die Beete nach einem gemeinsamen Plan gepflegt. {person} notiert die verfügbaren Werkzeuge und {other} prüft den Wasserbehälter. Jede Beobachtung erhält eine Zeitangabe. Wenn eine Angabe fehlt, bleibt das Feld ausdrücklich als unbekannt markiert. Die Gruppe vergleicht zuerst die Bedingungen und anschließend die Messwerte. Eine Änderung des Plans wird mit einem Grund dokumentiert, damit spätere Entscheidungen nachvollziehbar bleiben.'
        elif kind == 9:
            text = f'Informe {i}. El equipo prepara una exposición en la biblioteca. {person} revisa las etiquetas y {other} organiza el recorrido. Cada sección necesita una explicación breve y una indicación clara de su relación con la siguiente. Antes de abrir las puertas, se comprueba que el orden de las piezas coincida con el plano. Las preguntas de los visitantes se agrupan por tema. El resumen final conserva tanto las observaciones comunes como los casos que necesitan una revisión adicional.'
        elif kind == 10:
            text = f'记录{i}：社区图书馆正在整理借阅资料。工作人员先检查每条记录的日期、编号和说明，再按主题分类。缺失的信息需要明确标记，不能直接当作零处理。如果发现两条记录的编号相同，应先核对原始来源，而不是随意删除其中一条。整理完成后，另一位工作人员抽查分类结果，并写下仍需确认的问题。这个过程把观察、解释和后续行动分开，使后来阅读记录的人能够理解每一步的依据。'
        else:
            entries = [(f'zone_{j}', rng.randint(1, 60), rng.randint(1, 90)) for j in range(5)]
            text = f'Data exercise {i}. Read this table as measurements from a fictional survey.\nzone,morning,evening\n' + '\n'.join((f'{z},{x},{y}' for (z, x, y) in entries)) + f'\nMorning total: {sum((x for (z, x, y) in entries))}. Evening total: {sum((y for (z, x, y) in entries))}. Compute a difference for each zone before discussing the overall change. A total alone does not show whether the change was evenly distributed. Keep the original measurements alongside any derived values so that the calculation can be checked.'
        paragraphs.append(text)
    rng.shuffle(paragraphs)
    return '\n\n'.join(paragraphs) + '\n'


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", choices=("basic", "diverse"), default="basic")
    p.add_argument("--seed", type=int)
    p.add_argument("--count", type=int)
    a = p.parse_args()
    if a.seed is None:
        a.seed = 617029 if a.profile == "diverse" else 803
    if a.count is None:
        a.count = 6000 if a.profile == "diverse" else 2048
    if a.count < 1:p.error("count must be positive")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    generate = make_diverse_text if a.profile == "diverse" else make_text
    a.output.write_text(generate(a.seed, a.count), encoding="utf-8")
