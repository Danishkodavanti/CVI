# CVI
# 🎭 FaceFilter Pro: Real-Time AI Augmentation

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/AI-MediaPipe-green.svg?style=for-the-badge)](https://google.github.io/mediapipe/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer--Vision-red.svg?style=for-the-badge&logo=opencv)](https://opencv.org/)

A high-performance real-time facial landmark detection and filter augmentation system. This project leverages Google's MediaPipe framework and OpenCV to provide low-latency, precise facial overlays.

---

## ⚡ Core Features
* **High-Fidelity Tracking:** Utilizes the BlazeFace short-range model for 468-point facial landmark detection.
* **Real-time Performance:** Optimized for 30+ FPS on standard CPU hardware.
* **Modular Design:** Clear separation between detection logic (`core`), filter algorithms (`filters`), and image processing (`utils`).
* **Cross-Platform:** Compatible with Windows, macOS, and Linux environments.

---

## 🛠️ Installation & Setup

Ensure you have Python 3.9+ installed on your system.

### 1. Local Setup & Installation
```bash
git clone [https://github.com/Danishkodavanti/CVI.git](https://github.com/Danishkodavanti/CVI.git)
cd CVI

# Create environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Mac/Linux)
# source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
```
### 2. Execution

```
python main.py
```

## 📁 Project Structure

```
FACE_FILTER/
├── core/            # Facial detection and landmark logic
├── filters/         # Specialized filter implementations
├── utils/           # Image manipulation and helper functions
├── main.py          # Application entry point
├── .gitignore       # Version control exclusion rules
└── requirements.txt # Dependency manifest
