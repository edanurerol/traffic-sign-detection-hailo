# Turkish Traffic Detection with YOLO and Hailo

Real-time traffic sign, traffic light, and road-object detection system developed for Raspberry Pi using YOLO and a Hailo AI accelerator.

## Project Overview

This project performs real-time object detection on YouTube videos and video streams. The trained YOLO model was converted to the Hailo HEF format and optimized for inference on a Raspberry Pi with a Hailo AI accelerator.

The system is designed to detect Turkish traffic signs, traffic lights, pedestrian crossings, and other traffic-related objects.

## Features

* Real-time inference with Hailo acceleration
* YouTube video stream support
* 50 traffic-related object classes
* Traffic sign detection
* Red, yellow, and green traffic light detection
* Pedestrian crossing detection
* Confidence threshold filtering
* Non-Maximum Suppression
* Detection screenshots for manual review
* Configurable post-processing

## Hardware

* Raspberry Pi
* Hailo AI accelerator
* HailoRT
* Linux operating system

## Technologies

* Python
* YOLO
* HailoRT
* OpenCV
* NumPy
* FFmpeg
* yt-dlp

## Project Files

```text
.
├── README.md
├── class_names.txt
├── config_50classes_normalized_logits.py
├── hailo_inference.py
├── postprocess_50classes_normalized_logits.py
├── run_youtube_50classes_live_strict_v3.py
└── review_screenshots/
```

### Main Files

* `run_youtube_50classes_live_strict_v3.py`
  Main application used for YouTube stream processing and real-time detection.

* `hailo_inference.py`
  Handles communication with the Hailo device and performs model inference.

* `postprocess_50classes_normalized_logits.py`
  Decodes YOLO model outputs, applies confidence filtering, and performs Non-Maximum Suppression.

* `config_50classes_normalized_logits.py`
  Contains model input, output, and post-processing configuration.

* `class_names.txt`
  Contains the names of the 50 detectable classes.

## Model File

The compiled Hailo model is not included in this repository because model files are excluded from Git tracking.

Required model filename:

```text
traffic_yolo11n_50classes_6heads_normalized_logits.hef
```

Place the HEF file in the project directory before running the application.

## Running the Project

Enter the project directory:

```bash
cd /home/pi/hailo_traffic/final_backup
```

Activate the Python environment if required:

```bash
source ~/hailo_env/bin/activate
```

Run the main application:

```bash
python3 run_youtube_50classes_live_strict_v3.py
```

Enter a YouTube video or stream URL when requested.

## Current Results

The system successfully performs real-time inference using the Hailo accelerator.

Traffic lights are generally detected reliably. Traffic sign detection performance varies depending on:

* Object distance
* Image resolution
* Motion blur
* Sign size
* Lighting conditions
* Similarity between traffic sign classes
* Dataset class imbalance

Small, distant, or blurred traffic signs remain the main challenge.

## Dataset and Training

The project was developed using a custom traffic dataset containing Turkish traffic signs, traffic lights, pedestrian crossings, and road-related objects.

The dataset was cleaned, merged, and split into training, validation, and test sets. The model was trained at an input resolution of 640 × 640 pixels.

## Future Improvements

* Improve traffic sign dataset quality
* Add more real Turkish road images
* Balance classes with limited training samples
* Reduce false-positive detections
* Improve small-object detection
* Add tracking between video frames
* Add live camera support
* Create a graphical user interface
* Add performance and accuracy benchmarks

## 📸 Demo

### Traffic Sign Detection
![](docs/screenshots/image1.jpeg)

### Traffic Light Detection
![](docs/screenshots/image2.jpeg)

### YouTube Live Detection
![](docs/screenshots/image3.jpeg)

### Crosswalk Detection
![](docs/screenshots/image4.jpeg)

## Disclaimer

This project is under development and should not be used as the sole decision-making system in a real vehicle or safety-critical environment.
