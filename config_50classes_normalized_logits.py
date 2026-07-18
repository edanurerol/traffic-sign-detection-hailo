from pathlib import Path

PROJECT_DIR = Path.home() / "hailo_traffic" / "traffic_project"

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "eski"
    / "traffic_yolo11n_50classes_6heads_normalized_logits.hef"
)

CLASS_NAMES_PATH = (
    PROJECT_DIR
    / "models"
    / "eski"
    / "class_names.txt"
)

VIDEO_PATH = (
    PROJECT_DIR
    / "videos"
    / "trafik_video.mp4"
)

VIDEO2_PATH = (
    PROJECT_DIR
    / "videos"
    / "trafik_video2.mp4"
)

OUTPUT_DIR = PROJECT_DIR / "outputs"
LOG_DIR = PROJECT_DIR / "logs"

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"HEF dosyası bulunamadı: {MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Sınıf isimleri dosyası bulunamadı: {CLASS_NAMES_PATH}"
    )

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8",
) as file:
    CLASS_NAMES = [
        line.strip()
        for line in file
        if line.strip()
    ]

if len(CLASS_NAMES) != 50:
    raise RuntimeError(
        f"50 sınıf bekleniyordu, bulunan: {len(CLASS_NAMES)}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
