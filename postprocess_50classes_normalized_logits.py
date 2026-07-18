import numpy as np
import cv2

from config_50classes_normalized_logits import CLASS_NAMES, INPUT_WIDTH, INPUT_HEIGHT


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _squeeze_batch(arr):
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _find_output(outputs, h, w, c):
    """
    Hailo output sözlüğünden istenen shape'e sahip output'u bulur.
    Beklenen format: (H,W,C) veya (1,H,W,C)
    """
    for name, value in outputs.items():
        arr = _squeeze_batch(value)
        if arr.shape == (h, w, c):
            return name, arr
    raise RuntimeError(f"Beklenen output bulunamadı: {(h, w, c)}")


def decode_yolo11_outputs(outputs, conf_threshold=0.60):
    """
    YOLO11/YOLOv8 tarzı DFL bbox + class output decode.

    Model çıkışları:
    - 80x80x64 bbox dağılımı
    - 80x80x18 sınıf skorları
    - 40x40x64 bbox dağılımı
    - 40x40x18 sınıf skorları
    - 20x20x64 bbox dağılımı
    - 20x20x18 sınıf skorları

    Dönüş:
    boxes: [[x1,y1,x2,y2], ...] 640x640 input koordinatlarında
    scores: [confidence, ...]
    class_ids: [class_id, ...]
    """

    scales = [
        (80, 80, 8),
        (40, 40, 16),
        (20, 20, 32),
    ]

    reg_max = 16
    proj = np.arange(reg_max, dtype=np.float32)

    all_boxes = []
    all_scores = []
    all_class_ids = []

    for h, w, stride in scales:
        _, bbox_raw = _find_output(outputs, h, w, 64)
        _, cls_raw = _find_output(outputs, h, w, len(CLASS_NAMES))

        bbox_raw = bbox_raw.astype(np.float32)
        cls_raw = cls_raw.astype(np.float32)

        # Sınıf skorları bazı HEF'lerde sigmoid sonrası gelebilir.
        # Eğer değerler 0-1 dışındaysa sigmoid uygula.
        if cls_raw.min() < 0.0 or cls_raw.max() > 1.0:
            cls_score = sigmoid(cls_raw)
        else:
            cls_score = cls_raw

        # bbox_raw: H,W,64 -> H,W,4,16
        bbox_dist = bbox_raw.reshape(h, w, 4, reg_max)
        bbox_prob = softmax(bbox_dist, axis=-1)
        distances = np.sum(bbox_prob * proj, axis=-1)  # H,W,4

        # Grid merkezleri
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        cx = (grid_x + 0.5) * stride
        cy = (grid_y + 0.5) * stride

        left = distances[:, :, 0] * stride
        top = distances[:, :, 1] * stride
        right = distances[:, :, 2] * stride
        bottom = distances[:, :, 3] * stride

        x1 = cx - left
        y1 = cy - top
        x2 = cx + right
        y2 = cy + bottom

        best_class = np.argmax(cls_score, axis=-1)
        best_score = np.max(cls_score, axis=-1)


        # Sınıfa özel güven eşikleri:
        # Trafik ışıkları ve yaya geçidi düşük eşikte korunur.
        # Diğer trafik levhalarında yanlış atamaları azaltmak için
        # daha yüksek eşik kullanılır.
        special_keywords = (
            "crosswalk",
            "yaya",
            "gecit",
            "green_light",
            "red_light",
            "yellow_light",
            "green light",
            "red light",
            "yellow light",
            "yesil",
            "kirmizi",
            "sari",
        )

        class_thresholds = np.full(
            best_score.shape,
            0.35,
            dtype=np.float32,
        )

        for class_index, class_name in enumerate(CLASS_NAMES):
            normalized_name = (
                class_name
                .strip()
                .lower()
                .replace("ı", "i")
                .replace("ş", "s")
                .replace("ğ", "g")
                .replace("ü", "u")
                .replace("ö", "o")
                .replace("ç", "c")
            )

            if any(
                keyword in normalized_name
                for keyword in special_keywords
            ):
                class_thresholds[best_class == class_index] = conf_threshold

        mask = best_score >= class_thresholds
        
        print(
        "Scale:", h, "x", w,
        "| max:", best_score.max(),
        "| mean:", best_score.mean(),
        "| gecen:", int(np.sum(mask))
        )


        if not np.any(mask):
            continue

        boxes = np.stack([x1, y1, x2, y2], axis=-1)[mask]
        scores = best_score[mask]
        class_ids = best_class[mask]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_class_ids.append(class_ids)

    if not all_boxes:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    return (
        np.concatenate(all_boxes).astype(np.float32),
        np.concatenate(all_scores).astype(np.float32),
        np.concatenate(all_class_ids).astype(np.int32),
    )


def apply_nms(boxes, scores, class_ids, iou_threshold=0.45):
    """
    OpenCV NMSBoxes kullanır.
    """
    if len(boxes) == 0:
        return boxes, scores, class_ids

    # OpenCV NMSBoxes formatı: x,y,w,h
    xywh = []
    for x1, y1, x2, y2 in boxes:
        xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

    indices = cv2.dnn.NMSBoxes(
        xywh,
        scores.astype(float).tolist(),
        score_threshold=0.0,
        nms_threshold=float(iou_threshold),
    )

    if len(indices) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    indices = np.array(indices).reshape(-1)

    return boxes[indices], scores[indices], class_ids[indices]


def scale_boxes_to_original(boxes, original_width, original_height):
    """
    Şu an preprocess sadece resize yaptığı için basit ölçekleme yapıyoruz.
    Letterbox eklenirse burası güncellenecek.
    """
    if len(boxes) == 0:
        return boxes

    scale_x = original_width / INPUT_WIDTH
    scale_y = original_height / INPUT_HEIGHT

    scaled = boxes.copy()
    scaled[:, [0, 2]] *= scale_x
    scaled[:, [1, 3]] *= scale_y

    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, original_width - 1)
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, original_height - 1)

    return scaled
