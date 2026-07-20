<h1 align="center">🚦 Traffic Sign Detection with Raspberry Pi 5 + Hailo-8L</h1>

<p align="center">
  Real-Time Traffic Sign, Traffic Light and Crosswalk Detection using YOLO and Hailo-8L AI Accelerator
</p>

<p align="center">
  <img src="assets/demo.gif" alt="Traffic Sign Detection Demo" width="900">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/YOLO-v8%20%7C%20v11-red">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/Raspberry%20Pi-5-C51A4A?logo=raspberrypi&logoColor=white">
  <img src="https://img.shields.io/badge/Hailo-8L-orange">
  <img src="https://img.shields.io/badge/Linux-Ubuntu-E95420?logo=ubuntu&logoColor=white">
</p>

---


<p align="center">
  <img src="assets/demogif.gif">
</p>

# 📸 Demo

The GIF at the top of this page shows the model running in real time on Raspberry Pi 5 with the Hailo-8L accelerator.

## Traffic Sign Detection

![](docs/screenshots/image1.jpeg)

![](docs/screenshots/image4.jpeg)

## Traffic Light Detection

![](docs/screenshots/image2.jpeg)

## Pedestrian Crossing Detection

![](docs/screenshots/image3.jpeg)

# 📌 Overview

This project implements a **real-time traffic perception system** on **Raspberry Pi 5** using the **Hailo-8L AI Accelerator**. A YOLO-based object detection model is trained, optimized, compiled into **HEF** format, and deployed for high-performance edge inference.

The system detects:

- 🚦 Traffic Signs
- 🚥 Traffic Lights
- 🚶 Crosswalks

from video streams or live camera input with low latency.

---

# ✨ Features

- Real-time object detection
- Raspberry Pi 5 deployment
- Hailo-8L hardware acceleration
- YOLO-based detection
- HEF optimized model
- Video inference
- Live camera inference
- OpenCV visualization

---

# 🛠️ Hardware

- Raspberry Pi 5
- Hailo-8L AI Accelerator
- USB Camera / Video Input

---

# 💻 Software

- Python 3.10
- Ultralytics YOLO
- OpenCV
- HailoRT
- Hailo Dataflow Compiler
- Ubuntu

---

# 📂 Project Structure

```text
traffic-sign-detection-hailo
│
├── assets/
│   └── demo.gif
│
├── models/
│
├── src/
│
├── videos/
│
├── README.md
│
└── requirements.txt
```

---

# 🚀 Pipeline

```
Dataset
    │
    ▼
YOLO Training
    │
    ▼
Export ONNX
    │
    ▼
Hailo Parse
    │
    ▼
Optimize
    │
    ▼
Compile (.hef)
    │
    ▼
Raspberry Pi 5
    │
    ▼
Real-Time Detection
```

---



# Author

**Edanur Erol**
--- 
**Rumeysa Leyla Demir** 
---
Computer Engineering Student
----

# License
This project is licensed under the **MIT License**.

The included model and dataset should be used according to their respective licenses.

**AI • Computer Vision • Embedded AI • Hailo**

