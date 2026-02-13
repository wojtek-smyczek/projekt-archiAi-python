import json
import math


def load_predictions(path):
    with open(path, 'r') as f:
        data = json.load(f)
    # Obsługa obu formatów: lista lub dict z kluczem "predictions"
    if isinstance(data, list):
        return data
    return data.get("predictions", [])


def get_orientation(det):
    """Zwraca 'horizontal' jeśli width > height, inaczej 'vertical'."""
    if det["width"] > det["height"]:
        return "horizontal"
    return "vertical"


def get_bbox(det):
    """Zwraca (x1, y1, x2, y2) z formatu center+size."""
    cx, cy = det["x"], det["y"]
    w, h = det["width"], det["height"]
    return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)


def overlap_ratio(a_min, a_max, b_min, b_max):
    """Ile procent zakresu A pokrywa się z zakresem B (0.0 - 1.0)."""
    overlap = max(0, min(a_max, b_max) - max(a_min, b_min))
    length_a = a_max - a_min
    if length_a == 0:
        return 0.0
    return overlap / length_a


def assign_dimensions_to_walls(predictions):
    """
    Przypisuje dimension_value bezpośrednio do wall na podstawie:
    1. Zgodności orientacji (wymiar obok ściany wzdłuż jej osi)
    2. Pokrycia zakresu (czy wartość leży w obrębie długości ściany)
    3. Odległości prostopadłej (im bliżej ściany, tym lepiej)

    Zwraca listę dict: {"wall": ..., "dimension_value": ..., "value": ..., "score": ...}
    """
    walls = [d for d in predictions if d["class"] == "wall"]
    dim_values = [d for d in predictions if d["class"] == "dimension_value"]

    if not walls or not dim_values:
        return []

    # Oblicz score dla każdej pary (dimension_value, wall)
    scores = []
    for dv in dim_values:
        dv_bbox = get_bbox(dv)
        dv_cx, dv_cy = dv["x"], dv["y"]

        for wall in walls:
            wall_bbox = get_bbox(wall)
            wall_cx, wall_cy = wall["x"], wall["y"]
            wall_orient = get_orientation(wall)

            if wall_orient == "horizontal":
                # Ściana pozioma: wymiar powinien być nad/pod nią
                # i pokrywać się w osi X
                axis_overlap = overlap_ratio(dv_bbox[0], dv_bbox[2],
                                             wall_bbox[0], wall_bbox[2])
                perp_dist = abs(dv_cy - wall_cy)
            else:
                # Ściana pionowa: wymiar powinien być po lewej/prawej
                # i pokrywać się w osi Y
                axis_overlap = overlap_ratio(dv_bbox[1], dv_bbox[3],
                                             wall_bbox[1], wall_bbox[3])
                perp_dist = abs(dv_cx - wall_cx)

            # Brak pokrycia wzdłuż osi ściany = słabe dopasowanie,
            # ale nie zerowe (wymiar może być lekko poza zakresem ściany)
            if axis_overlap < 0.01:
                # Sprawdź czy centrum wartości jest w rozszerzonym zakresie ściany (margines 20%)
                if wall_orient == "horizontal":
                    margin = wall["width"] * 0.2
                    in_range = (wall_bbox[0] - margin) <= dv_cx <= (wall_bbox[2] + margin)
                else:
                    margin = wall["height"] * 0.2
                    in_range = (wall_bbox[1] - margin) <= dv_cy <= (wall_bbox[3] + margin)

                if not in_range:
                    continue
                axis_overlap = 0.1  # minimalny bonus za bycie w marginesie

            # Score: wysoki overlap + niska odległość prostopadła
            # Normalizujemy dystans - im bliżej tym lepiej
            score = axis_overlap / (1 + perp_dist / 100)

            scores.append({
                "wall": wall,
                "dim_value": dv,
                "score": score,
                "perp_dist": perp_dist,
                "axis_overlap": axis_overlap,
            })

    # Sortuj malejąco po score
    scores.sort(key=lambda s: s["score"], reverse=True)

    # Greedy assignment: każdy dimension_value przypisany do max 1 ściany
    assigned_values = set()
    results = []

    for s in scores:
        dv_id = s["dim_value"]["detection_id"]
        if dv_id in assigned_values:
            continue

        assigned_values.add(dv_id)
        results.append({
            "wall_id": s["wall"]["detection_id"],
            "wall_center": (s["wall"]["x"], s["wall"]["y"]),
            "wall_size": (s["wall"]["width"], s["wall"]["height"]),
            "wall_orientation": get_orientation(s["wall"]),
            "value": s["dim_value"].get("value"),
            "value_center": (s["dim_value"]["x"], s["dim_value"]["y"]),
            "score": round(s["score"], 4),
        })

    return results


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "predykcje_raw.json"
    predictions = load_predictions(path)

    print(f"Wczytano {len(predictions)} detekcji")
    print(f"  - wall: {sum(1 for d in predictions if d['class'] == 'wall')}")
    print(f"  - dimension_value: {sum(1 for d in predictions if d['class'] == 'dimension_value')}")
    print(f"  - dimension_line: {sum(1 for d in predictions if d['class'] == 'dimension_line')} (pomijane)")
    print()

    results = assign_dimensions_to_walls(predictions)

    for r in results:
        orient = "↔" if r["wall_orientation"] == "horizontal" else "↕"
        print(f"  {orient} Ściana w {r['wall_center']} ({r['wall_size'][0]:.0f}x{r['wall_size'][1]:.0f})"
              f"  ←  wymiar {r['value']} mm"
              f"  (score: {r['score']})")

    # Sprawdź nieprzypisane ściany
    assigned_wall_ids = {r["wall_id"] for r in results}
    walls = [d for d in predictions if d["class"] == "wall"]
    unassigned = [w for w in walls if w["detection_id"] not in assigned_wall_ids]
    if unassigned:
        print(f"\nŚciany bez wymiaru ({len(unassigned)}):")
        for w in unassigned:
            orient = "↔" if get_orientation(w) == "horizontal" else "↕"
            print(f"  {orient} {w['detection_id'][:8]}... w ({w['x']}, {w['y']})")
