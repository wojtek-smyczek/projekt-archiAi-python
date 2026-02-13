import cv2
import torch
import torch.nn as nn
import json
import os
import numpy as np
import ezdxf
from roboflow import Roboflow
from torchvision import models

# 1. KONFIGURACJA MODELU I PLIKU WAG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model_wojtka.pth"

def get_trained_model():
    # 1. Tworzymy bazę ResNet18
    model = models.resnet18(weights=None)
    
    # 2. Dopasowujemy warstwę wejściową (1 kanał)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # 3. Definiujemy warstwę wyjściową IDENTYCZNIE jak w trainer.py
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),           # To jest kluczowe!
        nn.Linear(num_ftrs, 10)    # To są brakujące "fc.1.weight" i "fc.1.bias"
    )
    
    model.to(device)

    # 4. Ładowanie wag
    if os.path.exists(MODEL_PATH):
        print(f"--- Ładowanie modelu Twojego pisma (z Dropout): {MODEL_PATH} ---")
        # Załadowanie wag do nowej struktury Sequential
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"BŁĄD: Nie znaleziono pliku {MODEL_PATH}!")
        exit()
    
    model.eval() # PAMIĘTAJ: To wyłącza Dropout na potrzeby rozpoznawania
    return model
model_ai = get_trained_model()

# 2. KONFIGURACJA ROBOFLOW
rf = Roboflow(api_key='ITGmB9QXokUotWH0cwtf')
project = rf.workspace("python-hegfw").project("my-first-project-gpcqz")
model_yolo = project.version(5).model

# 3. NMS DLA DETEKCJI DIMENSION_VALUE
def nms_dimension_values(predictions, overlap_threshold=0.3):
    """Usuwa nakładające się detekcje dimension_value (Non-Maximum Suppression).
    Używa intersection/min_area zamiast IoU — lepiej łapie częściowe nakładanie."""
    dim_vals = [p for p in predictions if p['class'] == 'dimension_value']
    others = [p for p in predictions if p['class'] != 'dimension_value']

    if len(dim_vals) == 0:
        return predictions

    # Sortuj po confidence malejąco
    dim_vals.sort(key=lambda p: p['confidence'], reverse=True)

    kept = []
    for candidate in dim_vals:
        cx, cy, cw, ch = candidate['x'], candidate['y'], candidate['width'], candidate['height']
        cx1, cy1 = cx - cw/2, cy - ch/2
        cx2, cy2 = cx + cw/2, cy + ch/2

        is_duplicate = False
        for selected in kept:
            sx, sy, sw, sh = selected['x'], selected['y'], selected['width'], selected['height']
            sx1, sy1 = sx - sw/2, sy - sh/2
            sx2, sy2 = sx + sw/2, sy + sh/2

            inter_x1 = max(cx1, sx1)
            inter_y1 = max(cy1, sy1)
            inter_x2 = min(cx2, sx2)
            inter_y2 = min(cy2, sy2)

            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                candidate_area = cw * ch
                selected_area = sw * sh
                min_area = min(candidate_area, selected_area)
                # intersection / mniejszy box — łapie gdy mały box jest "w środku" większego
                overlap_ratio = inter_area / min_area

                if overlap_ratio > overlap_threshold:
                    is_duplicate = True
                    # Rozszerz wybrany box do unii obu boxów
                    selected['x'] = (min(cx1, sx1) + max(cx2, sx2)) / 2
                    selected['y'] = (min(cy1, sy1) + max(cy2, sy2)) / 2
                    selected['width'] = max(cx2, sx2) - min(cx1, sx1)
                    selected['height'] = max(cy2, sy2) - min(cy1, sy1)
                    break

        if not is_duplicate:
            kept.append(candidate)

    return kept + others

def merge_adjacent_dimension_values(predictions):
    """Scala sąsiednie detekcje dimension_value na tej samej linii.
    Np. '1' + '100' obok siebie → jedno złączone bbox do ponownego OCR."""
    dim_vals = [p for p in predictions if p['class'] == 'dimension_value']
    others = [p for p in predictions if p['class'] != 'dimension_value']

    if len(dim_vals) < 2:
        return predictions

    merged = []
    used = set()

    for i, a in enumerate(dim_vals):
        if i in used:
            continue
        ax1, ay1 = a['x'] - a['width']/2, a['y'] - a['height']/2
        ax2, ay2 = a['x'] + a['width']/2, a['y'] + a['height']/2

        group = [a]
        used.add(i)

        for j, b in enumerate(dim_vals):
            if j in used:
                continue
            bx1, by1 = b['x'] - b['width']/2, b['y'] - b['height']/2
            bx2, by2 = b['x'] + b['width']/2, b['y'] + b['height']/2

            # Na tej samej linii Y (overlap pionowy) i blisko w X
            y_overlap = min(ay2, by2) - max(ay1, by1)
            min_h = min(a['height'], b['height'])
            x_gap = max(bx1 - ax2, ax1 - bx2, 0)
            max_w = max(a['width'], b['width'])

            if y_overlap > min_h * 0.5 and x_gap < max_w * 0.5:
                group.append(b)
                used.add(j)
                # Rozszerz bbox A o B
                ax1 = min(ax1, bx1)
                ay1 = min(ay1, by1)
                ax2 = max(ax2, bx2)
                ay2 = max(ay2, by2)

        if len(group) > 1:
            # Scal w jeden bbox — wartość zostanie przeliczona przez OCR
            merged_entry = dict(group[0])
            merged_entry['x'] = (ax1 + ax2) / 2
            merged_entry['y'] = (ay1 + ay2) / 2
            merged_entry['width'] = ax2 - ax1
            merged_entry['height'] = ay2 - ay1
            # Usuń starą wartość — OCR przeliczy na scalonym bbox
            if 'value' in merged_entry:
                del merged_entry['value']
            merged.append(merged_entry)
            print(f"  Scalono {len(group)} sąsiednich dim_value w jeden bbox")
        else:
            merged.append(a)

    return merged + others

# 4. FUNKCJA SEGMENTUJĄCA (Ujednolicona z dataset_creator)
def _pad_and_resize(digit_crop):
    """Pad do kwadratu i resize do 64x64."""
    h, w = digit_crop.shape
    side = max(h, w) + 40
    square = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = digit_crop
    return cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)

def segment_digits(thresh_roi):
    column_sums = cv2.reduce(thresh_roi, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)[0]
    roi_h = thresh_roi.shape[0]

    # Krok 1: Znajdź surowe segmenty (ciągłe zakresy z pikseli > 0)
    raw_segments = []
    in_digit = False
    start_col = 0
    for x, val in enumerate(column_sums):
        if val > 0 and not in_digit:
            in_digit = True
            start_col = x
        elif val == 0 and in_digit:
            in_digit = False
            if x - start_col >= 3:
                raw_segments.append((start_col, x))
    if in_digit and thresh_roi.shape[1] - start_col >= 3:
        raw_segments.append((start_col, thresh_roi.shape[1]))

    # Krok 2: Dla każdego segmentu sprawdź czy trzeba podzielić
    # (szukaj lokalnych minimów kolumn jako potencjalnych granic cyfr)
    final_segments = []
    for seg_start, seg_end in raw_segments:
        seg_w = seg_end - seg_start
        if seg_w < roi_h * 0.15:
            continue  # za wąski — szum

        seg_sums = column_sums[seg_start:seg_end]
        max_sum = max(seg_sums)

        # Jeśli segment jest szeroki, spróbuj znaleźć wewnętrzne minimum do podziału
        # Typowa cyfra ma szerokość ~0.4-0.7x wysokości ROI
        if seg_w > roi_h * 0.9:
            # Szukaj najgłębszego lokalnego minimum w środkowej części segmentu
            margin = int(seg_w * 0.2)  # nie dziel zbyt blisko krawędzi
            search_zone = seg_sums[margin:seg_w - margin]
            if len(search_zone) > 0:
                min_idx = int(np.argmin(search_zone)) + margin
                min_val = seg_sums[min_idx]
                # Podziel jeśli minimum jest <50% maksimum
                if min_val < max_sum * 0.5:
                    split_x = seg_start + min_idx
                    final_segments.append((seg_start, split_x))
                    final_segments.append((split_x, seg_end))
                    continue

        final_segments.append((seg_start, seg_end))

    # Krok 3: Konwertuj segmenty na obrazki
    digits_images = []
    for seg_start, seg_end in final_segments:
        digit_crop = thresh_roi[:, seg_start:seg_end]
        if digit_crop.shape[1] > 2:
            digits_images.append(_pad_and_resize(digit_crop))
    return digits_images

# 4. GŁÓWNY PROCES PRZETWARZANIA
image_path = 'test.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Błąd: Nie znaleziono pliku {image_path}")
    exit()

prediction = model_yolo.predict(image_path, confidence=25).json()

# Aplikuj NMS aby usunąć nakładające się detekcje dimension_value
raw_predictions = prediction.get('predictions', [])
filtered_predictions = nms_dimension_values(raw_predictions)
# Scala sąsiednie detekcje dimension_value (np. "1" + "100" → "1100")
filtered_predictions = merge_adjacent_dimension_values(filtered_predictions)
print(f"Detekcje: {len(raw_predictions)} surowych -> {len(filtered_predictions)} po NMS+merge")

final_results = []
debug_dir = 'debug_digits'
os.makedirs(debug_dir, exist_ok=True)

print(f"Przetwarzanie obrazu: {image_path}...")

for i, obj in enumerate(filtered_predictions):
    klasa = obj['class']
    x, y, w, h = obj['x'], obj['y'], obj['width'], obj['height']
    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)

    if klasa == 'dimension_value':
        # Dodaj padding do bounding boxa żeby nie obcinać cyfr na krawędziach
        pad = 10
        img_h, img_w = image.shape[:2]
        roi = image[max(0, y1-pad):min(img_h, y2+pad), max(0, x1-pad):min(img_w, x2+pad)]
        
        if roi.size > 0:
            red_channel = roi[:, :, 2]
            # Próg 50 (zgodny z Twoimi testami)
            _, thresh = cv2.threshold(red_channel, 50, 255, cv2.THRESH_BINARY)
            
            digit_imgs = segment_digits(thresh)
            full_number_str = ""
            
            print(f"\nObiekt {i} (wymiar):")
            
            for d_idx, d_img in enumerate(digit_imgs):
                cv2.imwrite(f"{debug_dir}/roi_{i}_digit_{d_idx}.png", d_img)
                
                img_tensor = torch.from_numpy(d_img).float().unsqueeze(0).unsqueeze(0) / 255.0
                img_tensor = img_tensor.to(device)
                
                with torch.no_grad():
                    output = model_ai(img_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, pred = torch.max(probabilities, dim=1)

                    pred_val = pred.item()
                    conf_val = confidence.item() * 100

                    if conf_val < 70.0:
                        print(f"  Cyfra {d_idx}: {pred_val} (Pewność: {conf_val:.1f}%) ⚠ NIEPEWNA")
                    else:
                        print(f"  Cyfra {d_idx}: {pred_val} (Pewność: {conf_val:.1f}%)")
                    full_number_str += str(pred_val)
            
            if full_number_str:
                obj['value'] = int(full_number_str)
                print(f"  WYNIK KOŃCOWY: {full_number_str}")

    final_results.append(obj)

# 5. ZAPIS DO PLIKU
with open('dane_projektu.json', 'w') as f:
    json.dump(final_results, f, indent=4)

print("\nGotowe! Wszystkie wyniki znajdziesz w dane_projektu.json")

# 6. EKSPORT ŚCIAN DO DXF
print("\n--- Eksport DXF ---")

img_height = image.shape[0]

# Klasyfikacja orientacji: H (poziomy) jeśli width/height > 1.5, inaczej V (pionowy)
def get_orientation(obj):
    return 'H' if obj['width'] / obj['height'] > 1.5 else 'V'

# Dystans euklidesowy między centrami bboxów
def center_dist(a, b):
    return ((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2) ** 0.5

# Podział obiektów na kategorie
walls = [o for o in final_results if o['class'] == 'wall']
dim_values = [o for o in final_results if o['class'] == 'dimension_value' and 'value' in o]

print(f"Znaleziono: {len(walls)} ścian, {len(dim_values)} wartości wymiarów")

# Krok 2: Budowanie list ścian i grafu połączeń

doc = ezdxf.new('R2010')
msp = doc.modelspace()

SNAP_DIST = max(200, int(max(image.shape[:2]) * 0.15))  # adaptacyjny próg tolerancji
print(f"SNAP_DIST = {SNAP_DIST}px (obraz: {image.shape[1]}x{image.shape[0]})")

# Posortowane listy ścian (bez przypisanego wymiaru — wymiary idą do luk)
v_walls_list = []
h_walls_list = []
for w_idx, w in enumerate(walls):
    orient = get_orientation(w)
    entry = {
        'w_idx': w_idx, 'px_x': w['x'], 'px_y': w['y'],
        'half_w': w['width'] / 2, 'half_h': w['height'] / 2,
    }
    if orient == 'V':
        v_walls_list.append(entry)
    else:
        h_walls_list.append(entry)

v_walls_list.sort(key=lambda w: w['px_x'])
h_walls_list.sort(key=lambda w: w['px_y'])
n_v = len(v_walls_list)
n_h = len(h_walls_list)

v_labels = [f"x={v['px_x']:.0f}" for v in v_walls_list]
h_labels = [f"y={h['px_y']:.0f}" for h in h_walls_list]
print(f"\nŚciany V (lewa→prawa): {v_labels}")
print(f"Ściany H (góra→dół):   {h_labels}")

# --- Znajdź ścianę V/H najbliższą endpointowi ---

def find_v_at_h_endpoint(hw, endpoint):
    """Znajdź indeks V wall najbliższej endpointowi H wall."""
    target_x = hw['px_x'] - hw['half_w'] if endpoint == 'left' else hw['px_x'] + hw['half_w']
    best_vi, best_dist = None, SNAP_DIST
    for vi, vw in enumerate(v_walls_list):
        if not (vw['px_y'] - vw['half_h'] - SNAP_DIST <= hw['px_y'] <= vw['px_y'] + vw['half_h'] + SNAP_DIST):
            continue
        d = abs(vw['px_x'] - target_x)
        if d < best_dist:
            best_dist = d
            best_vi = vi
    return best_vi

def find_h_at_v_endpoint(vw, endpoint):
    """Znajdź indeks H wall najbliższej endpointowi V wall."""
    target_y = vw['px_y'] - vw['half_h'] if endpoint == 'top' else vw['px_y'] + vw['half_h']
    best_hi, best_dist = None, SNAP_DIST
    for hi, hw in enumerate(h_walls_list):
        if not (hw['px_x'] - hw['half_w'] - SNAP_DIST <= vw['px_x'] <= hw['px_x'] + hw['half_w'] + SNAP_DIST):
            continue
        d = abs(hw['px_y'] - target_y)
        if d < best_dist:
            best_dist = d
            best_hi = hi
    return best_hi

# --- Buduj graf połączeń ---
h_conn = {}  # hi -> (left_vi, right_vi)
v_conn = {}  # vi -> (top_hi, bottom_hi)

for hi, hw in enumerate(h_walls_list):
    h_conn[hi] = (find_v_at_h_endpoint(hw, 'left'), find_v_at_h_endpoint(hw, 'right'))
    lv, rv = h_conn[hi]
    print(f"  H[{hi}] y={hw['px_y']:.0f}: lewy→V[{lv}] prawy→V[{rv}]")

for vi, vw in enumerate(v_walls_list):
    v_conn[vi] = (find_h_at_v_endpoint(vw, 'top'), find_h_at_v_endpoint(vw, 'bottom'))
    th, bh = v_conn[vi]
    print(f"  V[{vi}] x={vw['px_x']:.0f}: góra→H[{th}] dół→H[{bh}]")

# --- Krok 3: Identyfikacja luk (gaps) między kolejnymi równoległymi ścianami ---
print("\n--- Identyfikacja luk (gaps) ---")

h_gaps = []  # luki poziome: odległości między kolejnymi V walls
for i in range(n_v - 1):
    gap = {
        'left_vi': i, 'right_vi': i + 1,
        'px_left': v_walls_list[i]['px_x'],
        'px_right': v_walls_list[i + 1]['px_x'],
        'px_center': (v_walls_list[i]['px_x'] + v_walls_list[i + 1]['px_x']) / 2,
        'px_span': v_walls_list[i + 1]['px_x'] - v_walls_list[i]['px_x'],
    }
    h_gaps.append(gap)
    print(f"  H gap {i}: V[{i}]↔V[{i+1}] px=[{gap['px_left']:.0f}, {gap['px_right']:.0f}] span={gap['px_span']:.0f}")

v_gaps = []  # luki pionowe: odległości między kolejnymi H walls
for j in range(n_h - 1):
    gap = {
        'top_hi': j, 'bottom_hi': j + 1,
        'px_top': h_walls_list[j]['px_y'],
        'px_bottom': h_walls_list[j + 1]['px_y'],
        'px_center': (h_walls_list[j]['px_y'] + h_walls_list[j + 1]['px_y']) / 2,
        'px_span': h_walls_list[j + 1]['px_y'] - h_walls_list[j]['px_y'],
    }
    v_gaps.append(gap)
    print(f"  V gap {j}: H[{j}]↔H[{j+1}] px=[{gap['px_top']:.0f}, {gap['px_bottom']:.0f}] span={gap['px_span']:.0f}")

# --- Krok 4: Dopasowanie dim_value → gap (scoring + greedy) ---
print("\n--- Dopasowanie wymiarów do luk ---")

MARGIN = max(image.shape[:2]) * 0.05  # margines na zakres pozycji

# Zbierz kandydatów: (score, gap_type, gap_idx, dv_idx)
candidates = []
for dv_idx, dv in enumerate(dim_values):
    dv_x, dv_y = dv['x'], dv['y']

    # Dopasuj do H gaps (dim_value.x w zakresie V[i].px_x .. V[i+1].px_x)
    for g_idx, gap in enumerate(h_gaps):
        if gap['px_left'] - MARGIN <= dv_x <= gap['px_right'] + MARGIN:
            dist = abs(dv_x - gap['px_center'])
            score = dist / max(gap['px_span'], 1)
            candidates.append((score, 'H', g_idx, dv_idx))

    # Dopasuj do V gaps (dim_value.y w zakresie H[j].px_y .. H[j+1].px_y)
    for g_idx, gap in enumerate(v_gaps):
        if gap['px_top'] - MARGIN <= dv_y <= gap['px_bottom'] + MARGIN:
            dist = abs(dv_y - gap['px_center'])
            score = dist / max(gap['px_span'], 1)
            candidates.append((score, 'V', g_idx, dv_idx))

# Greedy: najlepszy score wygrywa, bez duplikatów (1 dim_value → 1 gap)
candidates.sort(key=lambda c: c[0])
h_gap_values = [None] * len(h_gaps)
v_gap_values = [None] * len(v_gaps)
used_dvs = set()
used_h_gaps = set()
used_v_gaps = set()

for score, gap_type, g_idx, dv_idx in candidates:
    if dv_idx in used_dvs:
        continue
    if gap_type == 'H' and g_idx in used_h_gaps:
        continue
    if gap_type == 'V' and g_idx in used_v_gaps:
        continue

    val = dim_values[dv_idx]['value']
    if gap_type == 'H':
        h_gap_values[g_idx] = val
        used_h_gaps.add(g_idx)
        print(f"  dim_value={val} → H gap {g_idx} (V[{h_gaps[g_idx]['left_vi']}]↔V[{h_gaps[g_idx]['right_vi']}]) score={score:.3f}")
    else:
        v_gap_values[g_idx] = val
        used_v_gaps.add(g_idx)
        print(f"  dim_value={val} → V gap {g_idx} (H[{v_gaps[g_idx]['top_hi']}]↔H[{v_gaps[g_idx]['bottom_hi']}]) score={score:.3f}")
    used_dvs.add(dv_idx)

for dv_idx, dv in enumerate(dim_values):
    if dv_idx not in used_dvs:
        print(f"  Nieprzypisany dim_value={dv['value']} @ ({dv['x']:.0f}, {dv['y']:.0f})")

print(f"H gap values: {h_gap_values}")
print(f"V gap values: {v_gap_values}")

# --- Pixel fallback: proporcja z known gaps dla luk bez wartości ---
known_h_scales = []
for i, gap in enumerate(h_gaps):
    if h_gap_values[i] is not None and gap['px_span'] > 0:
        known_h_scales.append(h_gap_values[i] / gap['px_span'])
if known_h_scales:
    avg_h_scale = sum(known_h_scales) / len(known_h_scales)
    for i in range(len(h_gaps)):
        if h_gap_values[i] is None:
            h_gap_values[i] = h_gaps[i]['px_span'] * avg_h_scale
            print(f"  Pixel fallback: H gap {i} = {h_gap_values[i]:.1f} (scale={avg_h_scale:.4f})")

known_v_scales = []
for j, gap in enumerate(v_gaps):
    if v_gap_values[j] is not None and gap['px_span'] > 0:
        known_v_scales.append(v_gap_values[j] / gap['px_span'])
if known_v_scales:
    avg_v_scale = sum(known_v_scales) / len(known_v_scales)
    for j in range(len(v_gaps)):
        if v_gap_values[j] is None:
            v_gap_values[j] = v_gaps[j]['px_span'] * avg_v_scale
            print(f"  Pixel fallback: V gap {j} = {v_gap_values[j]:.1f} (scale={avg_v_scale:.4f})")

# --- Krok 5: Sekwencyjne wyznaczanie pozycji ---
print("\n--- Pozycje DXF (sekwencyjne) ---")

# V walls: pozycje X (lewa→prawa, V[0]=0)
v_x = [None] * n_v
v_x[0] = 0.0
for i in range(len(h_gaps)):
    if v_x[i] is not None and h_gap_values[i] is not None:
        v_x[i + 1] = v_x[i] + h_gap_values[i]

# H walls: pozycje Y (dół→góra, H[last]=0)
h_y = [None] * n_h
if n_h > 0:
    h_y[n_h - 1] = 0.0
for j in range(len(v_gaps) - 1, -1, -1):
    if h_y[j + 1] is not None and v_gap_values[j] is not None:
        h_y[j] = h_y[j + 1] + v_gap_values[j]

print(f"v_x = {v_x}")
print(f"h_y = {h_y}")

# --- Krok 6: Rysowanie ścian ---
print("\n--- Rysowanie ścian ---")
drawn = 0

for hi, hw in enumerate(h_walls_list):
    y = h_y[hi]
    if y is None:
        print(f"  POMINIĘTO H[{hi}] (y nieznane)")
        continue
    lv, rv = h_conn[hi]

    # Wyznacz x1 (lewy koniec) — z pozycji lewej V wall
    if lv is not None and v_x[lv] is not None:
        x1 = v_x[lv]
    else:
        print(f"  POMINIĘTO H[{hi}] (brak lewej V wall z pozycją)")
        continue

    # Wyznacz x2 (prawy koniec) — z pozycji prawej V wall
    if rv is not None and v_x[rv] is not None:
        x2 = v_x[rv]
    else:
        print(f"  POMINIĘTO H[{hi}] (brak prawej V wall z pozycją)")
        continue

    msp.add_line((x1, y), (x2, y))
    print(f"  DXF H[{hi}]: ({x1:.1f}, {y:.1f}) → ({x2:.1f}, {y:.1f})  [długość={abs(x2-x1):.1f}]")
    drawn += 1

for vi, vw in enumerate(v_walls_list):
    x = v_x[vi]
    if x is None:
        print(f"  POMINIĘTO V[{vi}] (x nieznane)")
        continue
    th, bh = v_conn[vi]

    # Wyznacz y1 (dolny koniec) — z pozycji dolnej H wall
    if bh is not None and h_y[bh] is not None:
        y1 = h_y[bh]
    else:
        print(f"  POMINIĘTO V[{vi}] (brak dolnej H wall z pozycją)")
        continue

    # Wyznacz y2 (górny koniec) — z pozycji górnej H wall
    if th is not None and h_y[th] is not None:
        y2 = h_y[th]
    else:
        print(f"  POMINIĘTO V[{vi}] (brak górnej H wall z pozycją)")
        continue

    msp.add_line((x, y1), (x, y2))
    print(f"  DXF V[{vi}]: ({x:.1f}, {y1:.1f}) → ({x:.1f}, {y2:.1f})  [długość={abs(y2-y1):.1f}]")
    drawn += 1

# --- Krok 7: Domknięcie — otwarte końce ścian → łącz brakującą ścianą ---
unassigned_dvs = [dv for dv_idx, dv in enumerate(dim_values) if dv_idx not in used_dvs]

if unassigned_dvs:
    print(f"\n--- Domknięcie: {len(unassigned_dvs)} nieprzypisanych wymiarów ---")

    open_endpoints = []

    # V walls z otwartym końcem: znajdź najbliższy nieprzypisany wymiar jako długość
    for vi, vw in enumerate(v_walls_list):
        x = v_x[vi]
        if x is None:
            continue
        th, bh = v_conn[vi]
        has_top = (th is not None and h_y[th] is not None)
        has_bottom = (bh is not None and h_y[bh] is not None)

        if has_top == has_bottom:  # obie strony znane lub obie nieznane — pomiń
            continue

        # Znajdź najbliższy nieprzypisany wymiar
        best_dv, best_dist = None, float('inf')
        for dv in unassigned_dvs:
            d = ((vw['px_x'] - dv['x'])**2 + (vw['px_y'] - dv['y'])**2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_dv = dv
        if best_dv is None:
            continue
        val = best_dv['value']

        if has_bottom and not has_top:
            top_y = h_y[bh] + val
            open_endpoints.append({'x': x, 'y': top_y, 'type': 'V', 'idx': vi, 'side': 'top'})
            print(f"  V[{vi}] otwarta u góry: ({x:.1f}, {top_y:.1f}) [użyto dim={val}]")
        elif has_top and not has_bottom:
            bot_y = h_y[th] - val
            open_endpoints.append({'x': x, 'y': bot_y, 'type': 'V', 'idx': vi, 'side': 'bottom'})
            print(f"  V[{vi}] otwarta u dołu: ({x:.1f}, {bot_y:.1f}) [użyto dim={val}]")

    # H walls z otwartym końcem
    for hi, hw in enumerate(h_walls_list):
        y = h_y[hi]
        if y is None:
            continue
        lv, rv = h_conn[hi]
        has_left = (lv is not None and v_x[lv] is not None)
        has_right = (rv is not None and v_x[rv] is not None)

        if has_left == has_right:
            continue

        best_dv, best_dist = None, float('inf')
        for dv in unassigned_dvs:
            d = ((hw['px_x'] - dv['x'])**2 + (hw['px_y'] - dv['y'])**2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_dv = dv
        if best_dv is None:
            continue
        val = best_dv['value']

        if has_left and not has_right:
            right_x = v_x[lv] + val
            open_endpoints.append({'x': right_x, 'y': y, 'type': 'H', 'idx': hi, 'side': 'right'})
            print(f"  H[{hi}] otwarta z prawej: ({right_x:.1f}, {y:.1f}) [użyto dim={val}]")
        elif has_right and not has_left:
            left_x = v_x[rv] - val
            open_endpoints.append({'x': left_x, 'y': y, 'type': 'H', 'idx': hi, 'side': 'left'})
            print(f"  H[{hi}] otwarta z lewej: ({left_x:.1f}, {y:.1f}) [użyto dim={val}]")

    Y_TOL = 50  # tolerancja dopasowania pozycji Y/X otwartych końców

    # Paruj otwarte końce V walls na tej samej wysokości → łącz ścianą H
    v_open = [ep for ep in open_endpoints if ep['type'] == 'V']
    for i in range(len(v_open)):
        for j in range(i + 1, len(v_open)):
            ep1, ep2 = v_open[i], v_open[j]
            if abs(ep1['y'] - ep2['y']) < Y_TOL:
                y_conn = (ep1['y'] + ep2['y']) / 2
                x1_conn = min(ep1['x'], ep2['x'])
                x2_conn = max(ep1['x'], ep2['x'])
                length = abs(x2_conn - x1_conn)

                msp.add_line((x1_conn, y_conn), (x2_conn, y_conn))
                drawn += 1
                match = [dv for dv in unassigned_dvs if abs(dv['value'] - length) < length * 0.15]
                label = f", wymiar={match[0]['value']}" if match else ""
                print(f"  DOMKNIĘCIE H: ({x1_conn:.1f}, {y_conn:.1f}) → ({x2_conn:.1f}, {y_conn:.1f})  [długość={length:.1f}{label}]")

    # Paruj otwarte końce H walls na tej samej pozycji X → łącz ścianą V
    h_open = [ep for ep in open_endpoints if ep['type'] == 'H']
    for i in range(len(h_open)):
        for j in range(i + 1, len(h_open)):
            ep1, ep2 = h_open[i], h_open[j]
            if abs(ep1['x'] - ep2['x']) < Y_TOL:
                x_conn = (ep1['x'] + ep2['x']) / 2
                y1_conn = min(ep1['y'], ep2['y'])
                y2_conn = max(ep1['y'], ep2['y'])
                length = abs(y2_conn - y1_conn)

                msp.add_line((x_conn, y1_conn), (x_conn, y2_conn))
                drawn += 1
                match = [dv for dv in unassigned_dvs if abs(dv['value'] - length) < length * 0.15]
                label = f", wymiar={match[0]['value']}" if match else ""
                print(f"  DOMKNIĘCIE V: ({x_conn:.1f}, {y1_conn:.1f}) → ({x_conn:.1f}, {y2_conn:.1f})  [długość={length:.1f}{label}]")

dxf_path = 'rzut.dxf'
doc.saveas(dxf_path)
print(f"\nZapisano {drawn} ścian do {dxf_path} (pominięto {n_v + n_h - drawn} odłączonych)")