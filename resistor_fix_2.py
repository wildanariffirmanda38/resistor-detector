import cv2
import requests
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# ===== Pra-pemrosesan Gambar =====
def enhance_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {img_path}")
    enhanced = cv2.convertScaleAbs(img, alpha=1.5, beta=20)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return blurred

def save_enhanced(enhanced, filename):
    cv2.imwrite(filename, enhanced)

# ===== Fungsi Deteksi Roboflow =====
def roboflow_detect(img_path, api_key, model_id, version_num=1, conf=0.2):
    url = f"https://detect.roboflow.com/{model_id}/{version_num}?api_key={api_key}&confidence={conf}"
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    response = requests.post(
        url,
        files={"file": img_bytes},
        data={"format": "json"}
    )
    return response.json()

# ===== Non-max suppression: Satu area satu label (no duplikat)
def suppress_duplicate_predictions(predictions, iou_threshold=0.5):
    final_preds = []
    used = [False] * len(predictions)
    for i, pred_i in enumerate(predictions):
        if used[i]:
            continue
        # Compare to others
        best = pred_i
        for j, pred_j in enumerate(predictions):
            if i == j or used[j]:
                continue
            xi1, yi1 = int(pred_i['x'] - pred_i['width']//2), int(pred_i['y'] - pred_i['height']//2)
            xi2, yi2 = int(pred_i['x'] + pred_i['width']//2), int(pred_i['y'] + pred_i['height']//2)
            xj1, yj1 = int(pred_j['x'] - pred_j['width']//2), int(pred_j['y'] - pred_j['height']//2)
            xj2, yj2 = int(pred_j['x'] + pred_j['width']//2), int(pred_j['y'] + pred_j['height']//2)
            inter_x1 = max(xi1, xj1)
            inter_y1 = max(yi1, yj1)
            inter_x2 = min(xi2, xj2)
            inter_y2 = min(yi2, yj2)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            area_i = (xi2 - xi1) * (yi2 - yi1)
            area_j = (xj2 - xj1) * (yj2 - yj1)
            union_area = area_i + area_j - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            # JIka overlapped, ambil yang confidence lebih tinggi
            if iou > iou_threshold:
                if pred_j['confidence'] > best['confidence']:
                    best = pred_j
                used[j] = True
        final_preds.append(best)
        used[i] = True
    return final_preds

# ===== Gambar bounding box =====
def draw_detections(image, predictions):
    for pred in predictions:
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        box_color = (0, 255, 0)
        x1 = x - w // 2
        y1 = y - h // 2
        x2 = x + w // 2
        y2 = y + h // 2
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
    return image

# ===== Helper: Simbol resistor selalu diakhir =====
def move_resistor_symbol_to_end(band_list):
    bands = [b for b in band_list if b.lower() != "resistor symbol"]
    symbols = [b for b in band_list if b.lower() == "resistor symbol"]
    return bands + symbols

# ===== Grouping dan pengurutan warna (reverse jika gold di kiri) =====
def group_and_order_bands(predictions, reverse_if_gold_left=True):
    predictions.sort(key=lambda p: (p['y'], p['x']))
    grouped = []
    current_group = []
    prev_y = None
    for pred in predictions:
        if prev_y is None or abs(pred['y'] - prev_y) < 30:
            current_group.append(pred)
        else:
            grouped.append(current_group)
            current_group = [pred]
        prev_y = pred['y']
    if current_group:
        grouped.append(current_group)
    ordered = []
    for bands in grouped:
        bands_sorted = sorted(bands, key=lambda b: b['x'])
        if reverse_if_gold_left and bands_sorted and bands_sorted[0]['class'].lower() == 'gold':
            bands_sorted = list(reversed(bands_sorted))
        band_names = [b['class'] for b in bands_sorted]
        band_names = move_resistor_symbol_to_end(band_names)
        ordered.append(band_names)
    return ordered

# ===== Konversi Warna ke Nilai Resistor =====
def color_to_digit(color):
    color = color.lower()
    mapping = {
        'black': 0, 'brown': 1, 'red': 2, 'orange':3, 'yellow':4,
        'green':5, 'blue':6, 'violet':7, 'grey':8, 'white':9
    }
    return mapping.get(color, -1)

def color_to_multiplier(color):
    color = color.lower()
    mapping = {
        'pink': 0.001, 'silver': 0.01, 'gold':0.1,
        'black': 1, 'brown':10, 'red':100, 'orange':1000, 'yellow':10000,
        'green':100000, 'blue':1000000, 'violet':10000000,
        'grey':100000000, 'white':1000000000
    }
    return mapping.get(color, None)

def color_to_tolerance(color):
    color = color.lower()
    mapping = {
        'brown': 1, 'red': 2, 'green': 0.5, 'blue': 0.25, 'violet': 0.1,
        'grey': 0.05, 'gold': 5, 'silver': 10
    }
    return mapping.get(color, None)

def calculate_resistance(bands):
    bands = [b.lower() for b in bands if b.lower() != 'resistor symbol']
    if len(bands) < 3:
        return "Tidak cukup band"
    digit1 = color_to_digit(bands[0])
    digit2 = color_to_digit(bands[1])
    multiplier = color_to_multiplier(bands[2])
    tolerance = color_to_tolerance(bands[3]) if len(bands) >= 4 else None
    if digit1 == -1 or digit2 == -1 or multiplier is None:
        return "Warna tidak valid"
    resistance = (digit1 * 10 + digit2) * multiplier
    if resistance >= 1e6:
        resistance_str = f"{resistance / 1e6:.2f} MΩ"
    elif resistance >= 1e3:
        resistance_str = f"{resistance / 1e3:.2f} kΩ"
    else:
        resistance_str = f"{resistance:.2f} Ω"
    if tolerance:
        resistance_str += f" ±{tolerance}%"
    return resistance_str

# ====== FUNGSI KALIBRASI WARNA OTOMATIS ======
def get_band_mean_color(image, pred, mode='hsv'):
    x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
    x1 = max(x - w // 2, 0)
    y1 = max(y - h // 2, 0)
    x2 = x + w // 2
    y2 = y + h // 2
    band_crop = image[y1:y2, x1:x2]
    if band_crop.size == 0:
        return None
    if mode=='hsv':
        band_crop = cv2.cvtColor(band_crop, cv2.COLOR_BGR2HSV)
    mean_val = np.mean(band_crop.reshape(-1, 3), axis=0)
    return mean_val

def calibrate_color(image_path, predictions, save_to="calib_result.txt"):
    calib_dict = {}
    image = cv2.imread(image_path)
    for pred in predictions:
        class_name = pred['class'].lower()
        mean_hsv = get_band_mean_color(image, pred, 'hsv')
        if mean_hsv is not None:
            mean_h, mean_s, mean_v = mean_hsv
            if class_name not in calib_dict:
                calib_dict[class_name] = []
            calib_dict[class_name].append((mean_h, mean_s, mean_v))
    with open(save_to, "w") as f:
        for k, v in calib_dict.items():
            for hsv in v:
                f.write(f"{k}: H={hsv[0]:.1f}, S={hsv[1]:.1f}, V={hsv[2]:.1f}\n")
    print(f"Hasil kalibrasi warna tersimpan di {save_to}")
    return calib_dict

# ===== Panel label pojok kiri atas + confidence =====
def put_labels_topleft(image, predictions_grouped, predictions_raw):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    green = (0, 255, 0)
    red = (0, 0, 255)
    margin_side = 40
    margin_top = 30
    box_width = 1050
    row_height = 36
    total_rows = max(len(predictions_grouped), 5)
    box_height = total_rows * row_height + margin_top
    x0 = margin_side
    y0 = margin_top
    cv2.rectangle(image, (x0, y0), (x0 + box_width, y0 + box_height), (255, 255, 255), -1)
    cv2.putText(
        image,
        f"Deteksi: {len(predictions_grouped)} resistor",
        (x0 + 9, y0 + 20),
        font,
        scale,
        red,
        thickness
    )
    for i, colors in enumerate(predictions_grouped):
        ohm_value = calculate_resistance(colors)
        label_parts = []
        for color in colors:
            band_pred = next((p for p in predictions_raw if p['class'].lower() == color.lower()), None)
            conf_val = band_pred.get('confidence', 0.0) if band_pred else 0.0
            label_parts.append(f"{color}({conf_val:.2f})")
        label_str = ", ".join(label_parts)
        label = f"Resistor {i+1}: " + label_str + f" | Nilai: {ohm_value}"
        y_pos = y0 + 40 + i * row_height
        x_pos = x0 + 9
        cv2.putText(image, label, (x_pos, y_pos), font, scale, green, thickness)
    return image

# ===== Tampilkan dengan SCROLL (Tkinter Canvas) =====
def show_with_scroll(image_path):
    img = Image.open(image_path)
    window = tk.Tk()
    window.title("Deteksi Gelang Resistor")
    frame = tk.Frame(window, bd=2, relief=tk.SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    xscroll = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky="ew")
    yscroll = tk.Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky="ns")
    canvas = tk.Canvas(frame, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set, width=900, height=700)
    canvas.grid(row=0, column=0, sticky="news")
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=tk.BOTH,expand=True)
    photo = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor='nw', image=photo)
    canvas.config(scrollregion=canvas.bbox(tk.ALL))
    window.mainloop()

# ===== Main Program =====
if __name__ == "__main__":
    input_img = "resistor_6.jpg"  # Ganti sesuai nama file gambar Anda
    api_key = "DQZKIFQ43T9QqF4tyG0c"
    model_id = "rres-evbwv-tnfkq"

    confidence_threshold = 0.2
    enhanced_img = enhance_image(input_img)
    enhanced_path = "resistor_enhanced.jpg"
    save_enhanced(enhanced_img, enhanced_path)

    result = roboflow_detect(enhanced_path, api_key, model_id, conf=confidence_threshold)
    predictions_raw = result.get("predictions", [])
    predictions = [p for p in predictions_raw if p.get("confidence", 1.0) >= confidence_threshold]
    predictions = suppress_duplicate_predictions(predictions, iou_threshold=0.5)  # FILTER DUPLIKAT
    ordered_groups = group_and_order_bands(predictions)

    _ = calibrate_color(enhanced_path, predictions)

    print(f"Jumlah resistor terdeteksi: {len(ordered_groups)}")
    img_with_boxes = cv2.imread(enhanced_path)
    img_with_boxes = draw_detections(img_with_boxes, predictions)
    img_with_labels = put_labels_topleft(img_with_boxes, ordered_groups, predictions)

    cv2.imwrite("display_result.jpg", img_with_labels)
    show_with_scroll("display_result.jpg")
