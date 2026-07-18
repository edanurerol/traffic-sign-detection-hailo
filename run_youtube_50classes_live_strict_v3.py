#!/usr/bin/env python3

import argparse
import subprocess
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from config_50classes_normalized_logits import (
    MODEL_PATH,
    CLASS_NAMES,
    INPUT_WIDTH,
    INPUT_HEIGHT,
    OUTPUT_DIR,
)

from hailo_inference import HailoYoloInference

from postprocess_50classes_normalized_logits import (
    decode_yolo11_outputs,
    apply_nms,
    scale_boxes_to_original,
)


def resolve_youtube_url(youtube_url):
    """
    OpenCV için tek bağlantılı MP4/H.264 YouTube akışı seçer.
    """
    print("YouTube akış adresi alınıyor...")

    command = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "-f",
        "b[ext=mp4][vcodec^=avc1]/b[ext=mp4]/b",
        "-g",
        youtube_url,
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "YouTube akış adresi alınırken zaman aşımı oluştu."
        ) from exc

    except subprocess.CalledProcessError as exc:
        error_text = exc.stderr.strip()

        raise RuntimeError(
            "yt-dlp YouTube akış adresini alamadı.\n"
            + error_text
        ) from exc

    stream_urls = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith(
            ("http://", "https://")
        )
    ]

    if not stream_urls:
        raise RuntimeError(
            "yt-dlp geçerli bir video akış adresi döndürmedi."
        )

    stream_url = stream_urls[0]

    print("Akış adresi başarıyla alındı.")
    print("Akış sayısı:", len(stream_urls))
    print("OpenCV'ye tek video akışı gönderiliyor.")

    return stream_url



def preprocess(frame_bgr):
    resized = cv2.resize(
        frame_bgr,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB,
    )

    return np.ascontiguousarray(
        rgb.astype(np.uint8)
    )



def _normalize_class_name(class_name):
    return (
        str(class_name)
        .strip()
        .lower()
        .replace("ı", "i")
        .replace("ş", "s")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ö", "o")
        .replace("ç", "c")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _is_special_class(class_name):
    """
    Trafik ışıkları ve yaya geçidi sınıfları.
    Bunlar düşük eşikte ve bekletilmeden gösterilir.
    """
    normalized = _normalize_class_name(class_name)

    special_keywords = (
        "crosswalk",
        "yaya",
        "gecit",
        "green_light",
        "red_light",
        "yellow_light",
        "yesil",
        "kirmizi",
        "sari",
    )

    return any(
        keyword in normalized
        for keyword in special_keywords
    )


def _is_class_30(class_name):
    normalized = (
        _normalize_class_name(class_name)
        .replace("_", "")
    )

    return normalized in {
        "30",
        "30kmh",
        "speed30",
        "hiz30",
        "speedlimit30",
    }


def _is_gecit_yok(class_name):
    normalized = (
        _normalize_class_name(class_name)
        .replace("_", "")
    )

    return normalized in {
        "gecityok",
        "gecisyok",
        "noentry",
        "noaccess",
        "entryprohibited",
    }


def _is_saga_donus_yok(class_name):
    normalized = (
        _normalize_class_name(class_name)
        .replace("_", "")
    )

    return normalized in {
        "sagadonusyok",
        "sagdonusyok",
        "noturnright",
        "norightturn",
        "rightturnprohibited",
    }


def _single_box_iou(box_a, box_b):
    """İki xyxy kutusunun IoU değerini hesaplar."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection = intersection_width * intersection_height

    area_a = (
        max(0.0, float(box_a[2]) - float(box_a[0]))
        * max(0.0, float(box_a[3]) - float(box_a[1]))
    )

    area_b = (
        max(0.0, float(box_b[2]) - float(box_b[0]))
        * max(0.0, float(box_b[3]) - float(box_b[1]))
    )

    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def filter_false_positives(
    boxes,
    scores,
    class_ids,
    frame_width,
    frame_height,
    temporal_tracks,
    frame_number,
    required_frames=2,
    match_iou=0.25,
    minimum_area_ratio=0.0002,
):
    """
    Yanlış pozitifleri azaltır.

    - Işık/yaya geçidi: 0.20 ve anında gösterilir.
    - 30 levhası: 0.45 ve iki kare doğrulaması.
    - Diğer levhalar: 0.40 ve iki kare doğrulaması.
    - Aşırı küçük kutular elenir.
    """
    if len(boxes) == 0:
        temporal_tracks[:] = [
            track
            for track in temporal_tracks
            if frame_number - track["last_seen"] <= 1
        ]

        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    frame_area = float(frame_width * frame_height)
    minimum_box_area = frame_area * minimum_area_ratio

    kept_boxes = []
    kept_scores = []
    kept_class_ids = []

    updated_tracks = []

    for box, score, class_id in zip(
        boxes,
        scores,
        class_ids,
    ):
        class_id = int(class_id)
        score = float(score)

        if class_id < 0 or class_id >= len(CLASS_NAMES):
            continue

        class_name = CLASS_NAMES[class_id]

        x1, y1, x2, y2 = [
            float(value)
            for value in box
        ]

        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        box_area = box_width * box_height

        special_class = _is_special_class(class_name)
        class_30 = _is_class_30(class_name)
        class_gecit_yok = _is_gecit_yok(class_name)
        class_saga_donus_yok = _is_saga_donus_yok(class_name)

        # Kutunun görüntü içindeki büyüklüğü.
        box_area_ratio = (
            box_area / frame_area
            if frame_area > 0
            else 0.0
        )

        # 30 sınıfı biraz daha geniş toleransla kabul edilir.
        if class_30:
            required_confidence = 0.70
            detection_required_frames = 3

            if box_area_ratio < 0.0015:
                continue

            if box_width < 24 or box_height < 24:
                continue

        # Geçit yok ve sağa dönüş yok sınıfları çok fazla
        # yanlış pozitif ürettiği için sert doğrulanır.
        elif class_gecit_yok or class_saga_donus_yok:
            required_confidence = 0.80
            detection_required_frames = 4

            if box_area_ratio < 0.0025:
                continue

            if box_width < 30 or box_height < 30:
                continue

        # Işıklar ve yaya geçidi mevcut başarılı ayarlarda kalır.
        elif special_class:
            required_confidence = 0.20
            detection_required_frames = 1

        elif box_area_ratio < 0.0006:
            required_confidence = 0.50
            detection_required_frames = 3

        elif box_area_ratio < 0.0020:
            required_confidence = 0.44
            detection_required_frames = 2

        else:
            required_confidence = 0.40
            detection_required_frames = 2

        if score < required_confidence:
            continue

        # Çok küçük kutuları ele.
        # Trafik ışıkları uzakta küçük olabileceği için özel sınıflara
        # daha toleranslı davranılır.
        if special_class:
            special_minimum_area = minimum_box_area * 0.35

            if (
                box_area < special_minimum_area
                or box_width < 5
                or box_height < 5
            ):
                continue
        else:
            if (
                box_area < minimum_box_area
                or box_width < 8
                or box_height < 8
            ):
                continue

        # Trafik ışıkları ve yaya geçidi hemen gösterilir.
        if special_class:
            kept_boxes.append(box)
            kept_scores.append(score)
            kept_class_ids.append(class_id)
            continue

        matched_track = None

        for track in temporal_tracks:
            if track["class_id"] != class_id:
                continue

            # Yalnızca bir önceki karede görülen kayıt devam ettirilir.
            if frame_number - track["last_seen"] > 1:
                continue

            if _single_box_iou(track["box"], box) >= match_iou:
                matched_track = track
                break

        if matched_track is None:
            new_track = {
                "box": np.asarray(box, dtype=np.float32),
                "class_id": class_id,
                "count": 1,
                "last_seen": frame_number,
            }
        else:
            new_track = {
                "box": np.asarray(box, dtype=np.float32),
                "class_id": class_id,
                "count": matched_track["count"] + 1,
                "last_seen": frame_number,
            }

        updated_tracks.append(new_track)

        if new_track["count"] >= detection_required_frames:
            kept_boxes.append(box)
            kept_scores.append(score)
            kept_class_ids.append(class_id)

    # Önceki kareden kalıp bu karede eşleşmeyen kayıtlar silinir.
    temporal_tracks[:] = updated_tracks

    if not kept_boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    return (
        np.asarray(kept_boxes, dtype=np.float32),
        np.asarray(kept_scores, dtype=np.float32),
        np.asarray(kept_class_ids, dtype=np.int32),
    )


def draw_detections(
    frame,
    boxes,
    scores,
    class_ids,
):
    output = frame.copy()

    for box, score, class_id in zip(
        boxes,
        scores,
        class_ids,
    ):
        class_id = int(class_id)

        if class_id < 0 or class_id >= len(CLASS_NAMES):
            continue

        x1, y1, x2, y2 = [
            int(round(value))
            for value in box
        ]

        class_name = CLASS_NAMES[class_id]
        label = f"{class_name} {float(score):.2f}"

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )

        text_width, text_height = text_size

        text_top = max(
            0,
            y1 - text_height - baseline - 8,
        )

        cv2.rectangle(
            output,
            (x1, text_top),
            (x1 + text_width + 8, y1),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            output,
            label,
            (x1 + 4, max(text_height, y1 - baseline - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def open_youtube_stream(stream_url):
    cap = cv2.VideoCapture(
        stream_url,
        cv2.CAP_FFMPEG,
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    if not cap.isOpened():
        cap.release()

        raise RuntimeError(
            "YouTube akışı OpenCV ile açılamadı."
        )

    return cap


def main():
    parser = argparse.ArgumentParser(
        description=(
            "YouTube üzerinden Hailo-8L "
            "50 sınıflı trafik algılama"
        )
    )

    parser.add_argument(
        "--url",
        required=True,
        help="YouTube bağlantısı",
    )

    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--display",
        action="store_true",
    )

    parser.add_argument(
        "--save",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        default=str(
            OUTPUT_DIR / "youtube_50classes_result.mp4"
        ),
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0 sınırsızdır",
    )

    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Her inference sonrası atlanacak kare sayısı",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(
            f"HEF bulunamadı: {model_path}"
        )

    print("=" * 80)
    print("YOUTUBE + HAILO-8L 50 SINIFLI TEST")
    print("=" * 80)
    print("Model:", model_path)
    print("Sınıf sayısı:", len(CLASS_NAMES))
    print("Confidence:", args.conf)
    print("IoU:", args.iou)
    print("YouTube:", args.url)

    stream_url = resolve_youtube_url(args.url)

    print("Akış adresi başarıyla alındı.")

    cap = open_youtube_stream(stream_url)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if width <= 0 or height <= 0:
        width = 1280
        height = 720

    if source_fps <= 0 or source_fps > 120:
        source_fps = 25.0

    print("Akış çözünürlüğü:", width, "x", height)
    print("Akış FPS:", source_fps)

    writer = None

    if args.save:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (width, height),
        )

        if not writer.isOpened():
            cap.release()

            raise RuntimeError(
                f"Çıktı videosu açılamadı: {output_path}"
            )

        print("Kayıt yolu:", output_path)

    processed_frames = 0
    total_detections = 0
    failed_reads = 0

    # Levhaların ardışık kare doğrulaması için kullanılır.
    temporal_tracks = []

    # Son yaklaşık 20 saniyelik işlenmiş görüntüyü bellekte tutar.
    buffer_seconds = 20
    buffer_size = max(
        30,
        int(round(source_fps * buffer_seconds)),
    )

    review_buffer = deque(maxlen=buffer_size)
    should_exit = False

    screenshots_dir = Path("review_screenshots")
    screenshots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"İnceleme tamponu: yaklaşık {buffer_seconds} saniye "
        f"({buffer_size} kare)"
    )
    print("Kontroller:")
    print("  SPACE : Duraklat / devam et")
    print("  A     : 5 kare geri")
    print("  D     : 5 kare ileri")
    print("  Sol   : Yaklaşık 1 saniye geri")
    print("  Sağ   : Yaklaşık 1 saniye ileri")
    print("  S     : Görüntüyü kaydet")
    print("  Q/ESC : Çık")

    start_time = time.perf_counter()

    try:
        with HailoYoloInference(model_path) as model:

            while True:
                ok, frame = cap.read()

                if not ok:
                    failed_reads += 1

                    if failed_reads >= 20:
                        print(
                            "Akıştan art arda 20 kare alınamadı."
                        )
                        break

                    time.sleep(0.1)
                    continue

                failed_reads = 0

                input_image = preprocess(frame)

                outputs = model.infer(
                    input_image
                )

                boxes, scores, class_ids = (
                    decode_yolo11_outputs(
                        outputs,
                        conf_threshold=args.conf,
                    )
                )

                boxes, scores, class_ids = apply_nms(
                    boxes,
                    scores,
                    class_ids,
                    iou_threshold=args.iou,
                )

                boxes = scale_boxes_to_original(
                    boxes,
                    frame.shape[1],
                    frame.shape[0],
                )

                # Yanlış pozitifleri azaltan sınıf, boyut ve
                # ardışık kare filtreleri.
                boxes, scores, class_ids = filter_false_positives(
                    boxes,
                    scores,
                    class_ids,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                    temporal_tracks=temporal_tracks,
                    frame_number=processed_frames,
                    required_frames=2,
                    match_iou=0.25,
                    minimum_area_ratio=0.0002,
                )

                annotated = draw_detections(
                    frame,
                    boxes,
                    scores,
                    class_ids,
                )

                processed_frames += 1
                total_detections += len(boxes)

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                average_fps = (
                    processed_frames / elapsed
                    if elapsed > 0
                    else 0.0
                )

                status_text = (
                    f"Hailo FPS: {average_fps:.1f} "
                    f"Detections: {len(boxes)}"
                )

                cv2.putText(
                    annotated,
                    status_text,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if writer is not None:
                    if (
                        annotated.shape[1] != width
                        or annotated.shape[0] != height
                    ):
                        frame_to_save = cv2.resize(
                            annotated,
                            (width, height),
                        )
                    else:
                        frame_to_save = annotated

                    writer.write(
                        frame_to_save
                    )

                if args.display:
                    # İşlenmiş kareyi inceleme tamponuna ekle.
                    review_buffer.append(
                        {
                            "frame": annotated.copy(),
                            "number": processed_frames,
                        }
                    )

                    cv2.imshow(
                        "YouTube Hailo 50 Classes",
                        annotated,
                    )

                    key = cv2.waitKeyEx(1)

                    if key in (ord("q"), ord("Q"), 27):
                        print(
                            "Kullanıcı testi sonlandırdı."
                        )
                        should_exit = True
                        break

                    if key in (ord("s"), ord("S")):
                        screenshot_path = (
                            screenshots_dir
                            / f"frame_{processed_frames:06d}.png"
                        )

                        cv2.imwrite(
                            str(screenshot_path),
                            annotated,
                        )

                        print(
                            "Kare kaydedildi:",
                            screenshot_path,
                        )

                    if key == 32:
                        # SPACE ile inceleme moduna gir.
                        review_index = len(review_buffer) - 1
                        paused = True

                        print(
                            "Video duraklatıldı. "
                            "SPACE ile devam edebilirsin."
                        )

                        while paused and review_buffer:
                            selected = review_buffer[review_index]
                            review_frame = selected["frame"].copy()
                            review_number = selected["number"]

                            info_text = (
                                f"DURAKLATILDI | Kare: {review_number} "
                                f"| Tampon: {review_index + 1}/"
                                f"{len(review_buffer)}"
                            )

                            cv2.rectangle(
                                review_frame,
                                (0, review_frame.shape[0] - 45),
                                (review_frame.shape[1], review_frame.shape[0]),
                                (0, 0, 0),
                                -1,
                            )

                            cv2.putText(
                                review_frame,
                                info_text,
                                (15, review_frame.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                (0, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )

                            cv2.imshow(
                                "YouTube Hailo 50 Classes",
                                review_frame,
                            )

                            review_key = cv2.waitKeyEx(30)

                            # SPACE: canlı akışa geri dön
                            if review_key == 32:
                                paused = False
                                print(
                                    "Canlı akışa devam ediliyor."
                                )

                            # Q veya ESC: programdan çık
                            elif review_key in (
                                ord("q"),
                                ord("Q"),
                                27,
                            ):
                                paused = False
                                should_exit = True

                            # A: 5 kare geri
                            elif review_key in (
                                ord("a"),
                                ord("A"),
                            ):
                                review_index = max(
                                    0,
                                    review_index - 5,
                                )

                            # D: 5 kare ileri
                            elif review_key in (
                                ord("d"),
                                ord("D"),
                            ):
                                review_index = min(
                                    len(review_buffer) - 1,
                                    review_index + 5,
                                )

                            # Sol ok: yaklaşık 1 saniye geri
                            elif review_key in (
                                2424832,
                                65361,
                                81,
                            ):
                                review_index = max(
                                    0,
                                    review_index
                                    - int(round(source_fps)),
                                )

                            # Sağ ok: yaklaşık 1 saniye ileri
                            elif review_key in (
                                2555904,
                                65363,
                                83,
                            ):
                                review_index = min(
                                    len(review_buffer) - 1,
                                    review_index
                                    + int(round(source_fps)),
                                )

                            # S: seçili kareyi kaydet
                            elif review_key in (
                                ord("s"),
                                ord("S"),
                            ):
                                screenshot_path = (
                                    screenshots_dir
                                    / (
                                        f"frame_{review_number:06d}"
                                        "_review.png"
                                    )
                                )

                                cv2.imwrite(
                                    str(screenshot_path),
                                    selected["frame"],
                                )

                                print(
                                    "İncelenen kare kaydedildi:",
                                    screenshot_path,
                                )

                        if should_exit:
                            print(
                                "Kullanıcı testi sonlandırdı."
                            )
                            break

                for _ in range(args.skip):
                    cap.grab()

                if processed_frames % 50 == 0:
                    print(
                        f"İşlenen kare: {processed_frames}"
                        f" | Son kare tahmin: {len(boxes)}"
                        f" | Ortalama FPS: {average_fps:.2f}"
                    )

                if (
                    args.max_frames > 0
                    and processed_frames >= args.max_frames
                ):
                    print(
                        "Maksimum kare sayısına ulaşıldı:",
                        args.max_frames,
                    )
                    break

    finally:
        cap.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    average_fps = (
        processed_frames / elapsed
        if elapsed > 0
        else 0.0
    )

    print("=" * 80)
    print("YOUTUBE TESTİ TAMAMLANDI")
    print("İşlenen kare:", processed_frames)
    print("Toplam tahmin:", total_detections)
    print(f"Ortalama FPS: {average_fps:.2f}")

    if writer is not None:
        print("Kaydedilen video:", output_path)

    print("=" * 80)


if __name__ == "__main__":
    main()
