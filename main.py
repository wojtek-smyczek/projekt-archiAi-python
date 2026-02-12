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

# 4. FUNKCJA SEGMENTUJĄCA (Ujednolicona z dataset_creator)
def segment_digits(thresh_roi):
    column_sums = cv2.reduce(thresh_roi, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)[0]
    digits_images = []
    in_digit = False
    start_col = 0
    roi_h = thresh_roi.shape[0]

    for x, val in enumerate(column_sums):
        if val > 0 and not in_digit:
            in_digit = True
            start_col = x
        elif val == 0 and in_digit:
            in_digit = False
            digit_crop = thresh_roi[:, start_col:x]
            digit_w = digit_crop.shape[1]

            # Filtr: odrzuć fragmenty zbyt wąskie względem wysokości ROI (szum/artefakty)
            if digit_w < roi_h * 0.15:
                continue

            if digit_w > 2:
                # Padding identyczny jak w Twoim skrypcie do tworzenia datasetu
                h, w = digit_crop.shape
                side = max(h, w) + 40 # Zwiększony margines dla stabilności
                square = np.zeros((side, side), dtype=np.uint8)

                y_off = (side - h) // 2
                x_off = (side - w) // 2
                square[y_off:y_off+h, x_off:x_off+w] = digit_crop

                # Używamy INTER_AREA dla lepszej jakości przy skalowaniu
                square = cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)
                digits_images.append(square)
    return digits_images

# 4. GŁÓWNY PROCES PRZETWARZANIA
image_path = 'test.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Błąd: Nie znaleziono pliku {image_path}")
    exit()

prediction = model_yolo.predict(image_path, confidence=40).json()

# Aplikuj NMS aby usunąć nakładające się detekcje dimension_value
raw_predictions = prediction.get('predictions', [])
filtered_predictions = nms_dimension_values(raw_predictions)
print(f"Detekcje: {len(raw_predictions)} surowych -> {len(filtered_predictions)} po NMS")

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
                    # Obliczanie prawdopodobieństwa
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, pred = torch.max(probabilities, dim=1)
                    
                    pred_val = pred.item()
                    conf_val = confidence.item() * 100
                    
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
dim_lines = [o for o in final_results if o['class'] == 'dimension_line']
dim_values = [o for o in final_results if o['class'] == 'dimension_value' and 'value' in o]

print(f"Znaleziono: {len(walls)} ścian, {len(dim_lines)} linii wymiarowych, {len(dim_values)} wartości wymiarów")

# Krok 2: dimension_value → najbliższa dimension_line
line_to_value = {}
for dv in dim_values:
    if not dim_lines:
        break
    best_line = min(dim_lines, key=lambda dl: center_dist(dv, dl))
    best_idx = dim_lines.index(best_line)
    # Jeśli ta linia nie ma jeszcze wartości lub ta jest bliżej — przypisz
    if best_idx not in line_to_value or center_dist(dv, best_line) < center_dist(line_to_value[best_idx][1], best_line):
        line_to_value[best_idx] = (dv['value'], dv)

# Krok 3: dimension_line → najbliższa równoległa wall
wall_to_value = {}
unmatched_pairs = []  # pary (dimension_line, value) bez pasującej ściany
matched_wall_idxs = set()

for line_idx, (value, dv) in line_to_value.items():
    dl = dim_lines[line_idx]
    dl_orient = get_orientation(dl)
    parallel_walls = [(i, w) for i, w in enumerate(walls) if get_orientation(w) == dl_orient]
    if not parallel_walls:
        # Brak ściany o tej orientacji — zapamiętaj do syntetycznych ścian
        unmatched_pairs.append((dl, value))
        continue
    best_wall_idx, _ = min(parallel_walls, key=lambda iw: center_dist(dl, iw[1]))
    wall_to_value[best_wall_idx] = value
    matched_wall_idxs.add(best_wall_idx)

# Fallback: dla par dimension_line+value bez ściany, twórz syntetyczne ściany
# Pozycja = centrum linii wymiarowej, długość = z dimension_line bbox
for dl, value in unmatched_pairs:
    synthetic_wall = {
        'x': dl['x'],
        'y': dl['y'],
        'width': dl['width'],
        'height': dl['height'],
        'class': 'wall',
        'synthetic': True,
    }
    new_idx = len(walls)
    walls.append(synthetic_wall)
    wall_to_value[new_idx] = value
    print(f"  Syntetyczna ściana z dim_line ({get_orientation(dl)}), wartość={value}")

print(f"Sparowano {len(wall_to_value)} ścian z wymiarami ({len(unmatched_pairs)} syntetycznych)")

# Krok 4: Zapis DXF — pozycje ścian na podstawie dimension_value (nie pikseli)
doc = ezdxf.new('R2010')
msp = doc.modelspace()

SNAP_DIST = 100  # max pikseli tolerancji dla wykrywania połączeń

# Posortowane listy ścian z ich dimension_value
v_walls_list = []
h_walls_list = []
for w_idx, w in enumerate(walls):
    orient = get_orientation(w)
    entry = {
        'w_idx': w_idx, 'px_x': w['x'], 'px_y': w['y'],
        'half_w': w['width'] / 2, 'half_h': w['height'] / 2,
        'value': wall_to_value.get(w_idx),
    }
    if orient == 'V':
        v_walls_list.append(entry)
    else:
        h_walls_list.append(entry)

v_walls_list.sort(key=lambda w: w['px_x'])  # od lewej do prawej
h_walls_list.sort(key=lambda w: w['px_y'])  # od góry do dołu (obraz)

print(f"\nŚciany V (lewa→prawa): {[f'x={v['px_x']:.0f} val={v['value']}' for v in v_walls_list]}")
print(f"Ściany H (góra→dół):   {[f'y={h['px_y']:.0f} val={h['value']}' for h in h_walls_list]}")

# --- Funkcje do wykrywania połączeń między ścianami ---

def find_h_connecting(v_left, v_right):
    """Znajdź ścianę H łączącą dwie ściany V (endpointy blisko obu V)."""
    best, best_score = None, float('inf')
    for hw in h_walls_list:
        left_end = hw['px_x'] - hw['half_w']
        right_end = hw['px_x'] + hw['half_w']
        d_left = abs(left_end - v_left['px_x'])
        d_right = abs(right_end - v_right['px_x'])
        if d_left > SNAP_DIST or d_right > SNAP_DIST:
            continue
        # Sprawdź overlap Y z obiema V
        ok = True
        for vw in [v_left, v_right]:
            if not (vw['px_y'] - vw['half_h'] - SNAP_DIST <= hw['px_y'] <= vw['px_y'] + vw['half_h'] + SNAP_DIST):
                ok = False
                break
        if ok:
            score = d_left + d_right
            if score < best_score:
                best_score = score
                best = hw
    return best

def find_v_connecting(h_top, h_bottom):
    """Znajdź ścianę V łączącą dwie ściany H (endpointy blisko obu H)."""
    best, best_score = None, float('inf')
    for vw in v_walls_list:
        top_end = vw['px_y'] - vw['half_h']
        bottom_end = vw['px_y'] + vw['half_h']
        d_top = abs(top_end - h_top['px_y'])
        d_bottom = abs(bottom_end - h_bottom['px_y'])
        if d_top > SNAP_DIST or d_bottom > SNAP_DIST:
            continue
        ok = True
        for hw in [h_top, h_bottom]:
            if not (hw['px_x'] - hw['half_w'] - SNAP_DIST <= vw['px_x'] <= hw['px_x'] + hw['half_w'] + SNAP_DIST):
                ok = False
                break
        if ok:
            score = d_top + d_bottom
            if score < best_score:
                best_score = score
                best = vw
    return best

def find_v_at_h_endpoint(hw, endpoint):
    """Znajdź indeks ściany V najbliższej endpointowi ściany H."""
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
    """Znajdź indeks ściany H najbliższej endpointowi ściany V."""
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

# --- Pozycje X ścian V (z dimension_value ścian H) ---
n_v = len(v_walls_list)
v_x = [None] * n_v
v_x[0] = 0.0  # lewa ściana V na x=0

# Pass 1: kolejne pary V (użyj H wall łączącej je)
for i in range(n_v - 1):
    h = find_h_connecting(v_walls_list[i], v_walls_list[i + 1])
    if h and h['value'] and h['value'] > 0:
        v_x[i + 1] = v_x[i] + h['value']
        print(f"  X: V[{i}]→V[{i+1}] przez H(val={h['value']}) → x={v_x[i+1]}")

# Pass 2: uzupełnij luki z niekolejnych H walls
for i in range(n_v):
    if v_x[i] is not None:
        continue
    for j in range(n_v):
        if v_x[j] is None:
            continue
        h = find_h_connecting(v_walls_list[min(i, j)], v_walls_list[max(i, j)])
        if h and h['value'] and h['value'] > 0:
            v_x[i] = v_x[j] + h['value'] if i > j else v_x[j] - h['value']
            print(f"  X: V[{j}]→V[{i}] (uzupełnienie) H(val={h['value']}) → x={v_x[i]}")
            break

# --- Pozycje Y ścian H (z dimension_value ścian V) ---
n_h = len(h_walls_list)
h_y = [None] * n_h
h_y[-1] = 0.0  # dolna ściana H na y=0

# Pass 1: kolejne pary H (od dołu do góry, użyj V wall łączącej je)
for i in range(n_h - 1, 0, -1):
    v = find_v_connecting(h_walls_list[i - 1], h_walls_list[i])
    if v and v['value'] and v['value'] > 0:
        h_y[i - 1] = h_y[i] + v['value']
        print(f"  Y: H[{i}]→H[{i-1}] przez V(val={v['value']}) → y={h_y[i-1]}")

# Pass 2: uzupełnij luki z niekolejnych V walls
for i in range(n_h):
    if h_y[i] is not None:
        continue
    for j in range(n_h):
        if h_y[j] is None:
            continue
        v = find_v_connecting(h_walls_list[min(i, j)], h_walls_list[max(i, j)])
        if v and v['value'] and v['value'] > 0:
            h_y[i] = h_y[j] + v['value'] if i < j else h_y[j] - v['value']
            print(f"  Y: H[{j}]→H[{i}] (uzupełnienie) V(val={v['value']}) → y={h_y[i]}")
            break

# Fallback: jeśli nadal brak pozycji
for i in range(n_v):
    if v_x[i] is None:
        v_x[i] = v_x[i - 1] + 100 if i > 0 and v_x[i - 1] is not None else 0
        print(f"  UWAGA: V[{i}] brak dimension_value, fallback x={v_x[i]}")
for i in range(n_h):
    if h_y[i] is None:
        h_y[i] = h_y[i + 1] + 100 if i < n_h - 1 and h_y[i + 1] is not None else 0
        print(f"  UWAGA: H[{i}] brak dimension_value, fallback y={h_y[i]}")

print(f"\nPozycje DXF — V walls x: {[f'{x:.1f}' for x in v_x]}")
print(f"Pozycje DXF — H walls y: {[f'{y:.1f}' for y in h_y]}")

# --- Rysowanie ścian ---
for hi, hw in enumerate(h_walls_list):
    left_vi = find_v_at_h_endpoint(hw, 'left')
    right_vi = find_v_at_h_endpoint(hw, 'right')
    x1 = v_x[left_vi] if left_vi is not None else 0
    x2 = v_x[right_vi] if right_vi is not None else v_x[-1]
    y = h_y[hi]
    msp.add_line((x1, y), (x2, y))
    print(f"  DXF H: ({x1:.1f}, {y:.1f}) → ({x2:.1f}, {y:.1f})  [długość={abs(x2-x1):.1f}]")

for vi, vw in enumerate(v_walls_list):
    top_hi = find_h_at_v_endpoint(vw, 'top')
    bottom_hi = find_h_at_v_endpoint(vw, 'bottom')
    x = v_x[vi]
    y1 = h_y[bottom_hi] if bottom_hi is not None else 0
    y2 = h_y[top_hi] if top_hi is not None else h_y[0]
    msp.add_line((x, y1), (x, y2))
    print(f"  DXF V: ({x:.1f}, {y1:.1f}) → ({x:.1f}, {y2:.1f})  [długość={abs(y2-y1):.1f}]")

dxf_path = 'rzut.dxf'
doc.saveas(dxf_path)
print(f"\nZapisano {n_v + n_h} ścian do {dxf_path}")