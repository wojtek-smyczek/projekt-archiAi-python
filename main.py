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

# Krok 4: Obliczenie globalnej skali
scales = []
for w_idx, value in wall_to_value.items():
    w = walls[w_idx]
    orient = get_orientation(w)
    pixel_length = w['width'] if orient == 'H' else w['height']
    if pixel_length > 0:
        scales.append(value / pixel_length)

if scales:
    global_scale = sum(scales) / len(scales)
    print(f"Globalna skala: {global_scale:.4f} cm/px (z {len(scales)} par)")
else:
    # Fallback: 1 piksel = 1 jednostka
    global_scale = 1.0
    print("UWAGA: Brak sparowanych wymiarów, używam skali 1:1")

# Krok 5: Zapis DXF — ściany stykają się w narożnikach
doc = ezdxf.new('R2010')
msp = doc.modelspace()

# Oblicz centra ścian w DXF i podziel na H/V
h_walls = []  # (cy_dxf,)  — linie poziome
v_walls = []  # (cx_dxf,)  — linie pionowe

for w_idx, w in enumerate(walls):
    orient = get_orientation(w)
    cx_dxf = w['x'] * global_scale
    cy_dxf = (img_height - w['y']) * global_scale
    if orient == 'H':
        h_walls.append(cy_dxf)
    else:
        v_walls.append(cx_dxf)

# Wyznacz granice prostokąta z pozycji ścian
x_min = min(v_walls) if v_walls else 0
x_max = max(v_walls) if v_walls else 0
y_min = min(h_walls) if h_walls else 0
y_max = max(h_walls) if h_walls else 0

print(f"Narożniki DXF: ({x_min:.1f}, {y_min:.1f}) - ({x_max:.1f}, {y_max:.1f})")

# Rysuj ściany H (od lewej V do prawej V)
for cy in h_walls:
    msp.add_line((x_min, cy), (x_max, cy))

# Rysuj ściany V (od dolnej H do górnej H)
for cx in v_walls:
    msp.add_line((cx, y_min), (cx, y_max))

dxf_path = 'rzut.dxf'
doc.saveas(dxf_path)
print(f"Zapisano {len(h_walls) + len(v_walls)} ścian do {dxf_path} ({len(unmatched_pairs)} syntetycznych)")