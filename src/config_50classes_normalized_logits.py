from pathlib import Path

# Repo ana klasörü:
# /home/pi/hailo_traffic/final_backup
PROJECT_DIR = Path(__file__).resolve().parent.parent

# HEF modeli repo ana klasöründe
MODEL_PATH = (
    PROJECT_DIR
    / "traffic_yolo11n_50classes_6heads_normalized_logits.hef"
)

# Sınıf isimleri models klasöründe
CLASS_NAMES_PATH = (
    PROJECT_DIR
    / "models"
    / "class_names.txt"
)

# İsteğe bağlı yerel video dosyaları
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

# Çıktı ve log klasörleri
OUTPUT_DIR = PROJECT_DIR / "outputs"
LOG_DIR = PROJECT_DIR / "logs"

# Model giriş çözünürlüğü
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# Gerekli dosyaları kontrol et
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"HEF dosyası bulunamadı: {MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Sınıf isimleri dosyası bulunamadı: {CLASS_NAMES_PATH}"
    )

# Sınıf isimlerini oku
with CLASS_NAMES_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    CLASS_NAMES = [
        line.strip()
        for line in file
        if line.strip()
    ]

# Model 50 sınıflı olmalı
if len(CLASS_NAMES) != 50:
    raise RuntimeError(
        f"50 sınıf bekleniyordu, bulunan: {len(CLASS_NAMES)}"
    )

# Çıktı klasörlerini otomatik oluştur
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
