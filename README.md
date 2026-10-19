# 👁️ Real-Time Face Detection & Feature Tracking

An advanced computer vision application built with **OpenCV** and **Python**, supporting real-time FPS benchmarking, dual detection engines (classical Haar Cascades & modern Deep Learning SSD), hierarchical eye/smile tracking, and instant snapshot capturing.

---

## 🌟 Key Features

1. **Dual Detection Engines (Haar Cascades & OpenCV DNN SSD)**:
   - **Modern Deep Learning (SSD ResNet-10)**: High accuracy, rotation-tolerant, returns calibrated confidence scores. Compatible with both modern TensorFlow SSD (`.pb` / `.pbtxt` in OpenCV 5+) and Caffe SSD (`.caffemodel` / `.prototxt` in OpenCV 4.x).
   - **Classical Haar Cascades**: Ultra-lightweight detection using Viola-Jones framework.
   - **Hot-swappable**: Press `d` anytime during live execution to switch between detectors on the fly.

2. **Hierarchical ROI Feature Detection (Eyes & Smiles)**:
   - Scans **only** within the detected face Region of Interest (ROI).
   - **Smart Smile Detection**: Scans the lower 50% of the face ROI to eliminate false positives from eyes and forehead.
   - Separate color-coded visual indicators for faces, eyes, and smiles.

3. **Real-Time FPS Counter & Modern HUD**:
   - Smoothed moving-average FPS calculation (`time.perf_counter()`).
   - Translucent head-up display (HUD) showing current FPS, active detector, face count, and toggle states.

4. **Interactive Snapshot Capture (`s` Key)**:
   - Press `s` at any time to save high-resolution detection frames into the `captures/` folder with timestamped filenames (`snapshot_YYYYMMDD_HHMMSS.jpg`).
   - Displays an on-screen toast alert and terminal confirmation.

5. **Professional CLI (`argparse`)**:
   - Built-in validation, descriptive help menus (`--help`), and intuitive subcommands for `webcam`, `image`, and `video`.

---

## 🚀 Quickstart

### Option 1: One-Click Launch (Easiest)
On Windows, simply **double-click `run.bat`**! It will automatically install dependencies and show an interactive menu to launch live detection, test on a sample image, or run tests.

---

### Option 2: Run via Terminal

#### 1. Requirements & Setup
Install the dependencies:
```bash
pip install -r requirements.txt
```

Run the automated test suite:
```bash
python test_face_detection.py
```

### 2. Live Webcam Detection
Run live detection with the default DNN detector:
```bash
python face_detection.py webcam
```

Enable hierarchical eye and smile detection:
```bash
python face_detection.py webcam --detector dnn --eyes --smiles
```

### 3. Still Image Detection
Detect on an image and save the annotated result:
```bash
python face_detection.py image path/to/your_image.jpg --detector dnn --eyes --smiles --output output.jpg
```
Bring your own image for this example. Face photos and captured frames are not included in this public repository.

### 4. Video File Processing
Analyze a pre-recorded video:
```bash
python face_detection.py video path/to/video.mp4 --detector dnn --eyes --output processed_video.mp4
```

---

## ⌨️ Interactive Keyboard Controls

When running `webcam` or `video` mode, use these hotkeys:

| Key | Action |
| :--- | :--- |
| **`s`** | **Capture snapshot** (saves annotated frame to `captures/`) |
| **`d`** | **Switch detector** dynamically between Haar and DNN |
| **`e`** | **Toggle Eye detection** ON / OFF |
| **`m`** | **Toggle Smile detection** ON / OFF |
| **`q`** or **`ESC`** | **Quit** the application |

---

## 🛠️ Command-Line Arguments Reference

```text
usage: face_detection [-h] {webcam,image,video} ...

Options:
  --detector {haar,dnn}    Detection algorithm (default: dnn)
  --confidence CONFIDENCE  Confidence threshold for DNN (0.0 to 1.0, default: 0.5)
  --eyes                   Enable hierarchical eye detection
  --smiles                 Enable hierarchical smile detection
  --output, -o OUTPUT      Save processed image or video
  --no-show                Run headlessly without opening GUI window
  --no-fps                 Hide FPS counter in HUD
  --camera CAMERA          Webcam index (default: 0)
```

---

## 📁 Project Structure

```text
face-detection-app/
├── captures/                                    # Timestamped snapshot exports (created locally at runtime)
├── deploy.prototxt                              # Caffe SSD model architecture definition
├── res10_300x300_ssd_iter_140000_fp16.caffemodel# Caffe SSD deep learning weights (OpenCV 4.x)
├── opencv_face_detector.pbtxt                   # TensorFlow SSD model text configuration (OpenCV 5+)
├── opencv_face_detector_uint8.pb                # TensorFlow SSD deep learning weights (OpenCV 5+)
├── haarcascade_frontalface_default.xml          # Classical Viola-Jones Haar face cascade
├── haarcascade_eye.xml                          # Haar eye detection cascade
├── haarcascade_smile.xml                        # Haar smile detection cascade
├── face_detection.py                            # Core application: dual detectors, ROI pipeline, HUD & CLI
├── test_face_detection.py                       # Automated test suite (11 test cases across models & features)
├── your_image.jpg                               # Supply your own image for still-image mode
├── requirements.txt                             # Python dependencies (opencv-python, numpy)
├── run.bat                                      # Interactive launcher menu for Windows
├── run.sh                                       # Interactive launcher script for Linux/macOS
└── README.md                                    # Documentation and user guide
```

---

## 📊 Technical Comparison: Haar vs. DNN SSD

| Metric | Haar Cascades | OpenCV DNN SSD (ResNet-10) |
| :--- | :--- | :--- |
| **Framework** | Classical Viola-Jones (Handcrafted Haar features) | Deep Learning Single Shot MultiBox Detector |
| **Confidence Scores** | ❌ None (Discrete stage thresholds) | ✅ Yes (Calibrated probabilities 0–100%) |
| **Pose Tolerance** | Low (struggles with tilted or rotated faces) | High (handles angles, occlusions, and lighting changes) |
| **False Positives** | Moderate (especially in noisy backgrounds) | Very Low |
| **ROI Chaining** | Used for Eyes & Smile detection inside face box | Robust face bounding box provides stable ROI for cascades |
