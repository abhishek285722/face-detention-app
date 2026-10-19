"""Advanced Face Detection Application with OpenCV.

Features:
1. Real-Time FPS counter with moving average smoothing.
2. Hierarchical detection (Eyes & Smiles in Face ROI).
3. Interactive snapshot trigger ('s' key) saving to `captures/` folder.
4. Professional CLI powered by argparse with auto-generated help menus.
5. Dual detection engines: Classical Haar Cascades & Modern Deep Learning (SSD DNN).
"""

import argparse
from datetime import datetime
import os
import sys
import time

import cv2
import numpy as np

# Base directory for models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
HAAR_FACE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
HAAR_EYE_PATH = os.path.join(BASE_DIR, "haarcascade_eye.xml")
HAAR_SMILE_PATH = os.path.join(BASE_DIR, "haarcascade_smile.xml")

# DNN Model paths (TensorFlow and Caffe variants)
TF_PB_PATH = os.path.join(BASE_DIR, "opencv_face_detector_uint8.pb")
TF_PBTXT_PATH = os.path.join(BASE_DIR, "opencv_face_detector.pbtxt")
CAFFE_MODEL_PATH = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
CAFFE_PROTO_PATH = os.path.join(BASE_DIR, "deploy.prototxt")

CAPTURES_DIR = os.path.join(BASE_DIR, "captures")


class ModelManager:
    """Manages loading and inference for Haar Cascades and DNN detectors."""

    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.smile_cascade = None
        self.dnn_net = None
        self.dnn_type = None

    def load_haar_face(self):
        if self.face_cascade is None:
            if not os.path.exists(HAAR_FACE_PATH):
                raise FileNotFoundError(f"Missing Haar face model: {HAAR_FACE_PATH}")
            self.face_cascade = cv2.CascadeClassifier(HAAR_FACE_PATH)
            if self.face_cascade.empty():
                raise RuntimeError(f"Failed to load Haar cascade from {HAAR_FACE_PATH}")
        return self.face_cascade

    def load_haar_eyes(self):
        if self.eye_cascade is None:
            if not os.path.exists(HAAR_EYE_PATH):
                print(f"[Warning] Eye cascade not found at {HAAR_EYE_PATH}")
                return None
            self.eye_cascade = cv2.CascadeClassifier(HAAR_EYE_PATH)
            if self.eye_cascade.empty():
                print(f"[Warning] Failed to load eye cascade from {HAAR_EYE_PATH}")
                self.eye_cascade = None
        return self.eye_cascade

    def load_haar_smiles(self):
        if self.smile_cascade is None:
            if not os.path.exists(HAAR_SMILE_PATH):
                print(f"[Warning] Smile cascade not found at {HAAR_SMILE_PATH}")
                return None
            self.smile_cascade = cv2.CascadeClassifier(HAAR_SMILE_PATH)
            if self.smile_cascade.empty():
                print(f"[Warning] Failed to load smile cascade from {HAAR_SMILE_PATH}")
                self.smile_cascade = None
        return self.smile_cascade

    def load_dnn(self):
        """Loads DNN face detector. Supports TensorFlow SSD (OpenCV 5+) and Caffe SSD (OpenCV 4.x)."""
        if self.dnn_net is not None:
            return self.dnn_net

        # 1. Prefer TensorFlow SSD model (works seamlessly across OpenCV 4.x and 5.x)
        if os.path.exists(TF_PB_PATH) and os.path.exists(TF_PBTXT_PATH):
            try:
                self.dnn_net = cv2.dnn.readNetFromTensorflow(TF_PB_PATH, TF_PBTXT_PATH)
                self.dnn_type = "TensorFlow SSD"
                return self.dnn_net
            except Exception as e:
                print(f"[Notice] Failed to load TensorFlow model: {e}")

        # 2. Fallback to Caffe SSD model if supported by this OpenCV build
        if os.path.exists(CAFFE_MODEL_PATH) and os.path.exists(CAFFE_PROTO_PATH):
            if hasattr(cv2.dnn, "readNetFromCaffe"):
                try:
                    self.dnn_net = cv2.dnn.readNetFromCaffe(CAFFE_PROTO_PATH, CAFFE_MODEL_PATH)
                    self.dnn_type = "Caffe SSD"
                    return self.dnn_net
                except Exception as e:
                    print(f"[Notice] Failed to load Caffe model: {e}")

        raise RuntimeError(
            "Could not load DNN face detector model. "
            f"Ensure {os.path.basename(TF_PB_PATH)} and {os.path.basename(TF_PBTXT_PATH)} are present."
        )


def detect_faces_haar(face_cascade, gray_frame, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """Detect faces using Haar Cascades. Returns list of (x, y, w, h, confidence=None)."""
    rects = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    if len(rects) == 0:
        return []
    return [(int(x), int(y), int(w), int(h), None) for (x, y, w, h) in rects]


def detect_faces_dnn(dnn_net, frame, conf_threshold=0.5):
    """Detect faces using OpenCV DNN SSD detector. Returns list of (x, y, w, h, confidence)."""
    h, w = frame.shape[:2]
    # SSD ResNet-10 expects 300x300 BGR input with mean subtraction (104.0, 177.0, 123.0)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    dnn_net.setInput(blob)
    detections = dnn_net.forward()

    faces = []
    # detections shape: [1, 1, num_detections, 7]
    # Each row: [image_id, label, confidence, startX, startY, endX, endY]
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_threshold:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        startX, startY, endX, endY = box.astype(int)

        # Clamp bounding box coordinates within frame dimensions
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w, endX)
        endY = min(h, endY)

        box_w = endX - startX
        box_h = endY - startY

        if box_w > 10 and box_h > 10:
            faces.append((startX, startY, box_w, box_h, confidence))

    return faces


def detect_hierarchical_features(frame, gray_frame, faces, eye_cascade=None, smile_cascade=None):
    """Scan only within the face Region of Interest (ROI) to detect eyes and smiles."""
    annotated_faces = []

    for (x, y, w, h, conf) in faces:
        face_info = {
            "box": (x, y, w, h),
            "conf": conf,
            "eyes": [],
            "smiles": [],
        }

        # Face ROI
        roi_gray = gray_frame[y : y + h, x : x + w]

        # 1. Eye detection inside face ROI
        if eye_cascade is not None and roi_gray.size > 0:
            raw_eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(15, 15),
            )
            # Limit to at most 2 most prominent eyes
            if len(raw_eyes) > 2:
                raw_eyes = sorted(raw_eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
            for (ex, ey, ew, eh) in raw_eyes:
                face_info["eyes"].append((x + ex, y + ey, ew, eh))

        # 2. Smile detection inside lower-half of face ROI (reduces false positives from eyes/forehead)
        if smile_cascade is not None and roi_gray.size > 0:
            lower_half_y = int(h * 0.5)
            roi_gray_lower = roi_gray[lower_half_y:h, 0:w]
            if roi_gray_lower.size > 0:
                raw_smiles = smile_cascade.detectMultiScale(
                    roi_gray_lower,
                    scaleFactor=1.6,
                    minNeighbors=20,
                    minSize=(25, 15),
                )
                if len(raw_smiles) > 0:
                    # Pick largest detected smile
                    sx, sy, sw, sh = max(raw_smiles, key=lambda s: s[2] * s[3])
                    face_info["smiles"].append((x + sx, y + lower_half_y + sy, sw, sh))

        annotated_faces.append(face_info)

    return annotated_faces


def render_detections(frame, annotated_faces, detector_mode="dnn"):
    """Draw stylish bounding boxes, confidence badges, eye markers, and smile indicators."""
    # OpenCV uses BGR channel order (Blue, Green, Red)
    primary_color = (255, 191, 0) if detector_mode == "dnn" else (50, 205, 50)  # BGR: Deep Sky Blue (DNN) vs Lime Green (Haar)
    eye_color = (255, 255, 0)      # BGR: Cyan (Blue + Green)
    smile_color = (0, 215, 255)    # BGR: Amber / Gold (Green + Red)

    for face in annotated_faces:
        x, y, w, h = face["box"]
        conf = face["conf"]

        # Corner-accented stylish bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), primary_color, 2)
        corner_len = max(8, int(min(w, h) * 0.15))
        t = 3
        # Top-left corner
        cv2.line(frame, (x, y), (x + corner_len, y), primary_color, t)
        cv2.line(frame, (x, y), (x, y + corner_len), primary_color, t)
        # Top-right corner
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), primary_color, t)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), primary_color, t)
        # Bottom-left corner
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), primary_color, t)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), primary_color, t)
        # Bottom-right corner
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), primary_color, t)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), primary_color, t)

        # Label badge above face
        label = f"Face: {conf * 100:.1f}%" if conf is not None else "Face"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        badge_y1 = max(0, y - lh - 8)
        badge_y2 = y
        cv2.rectangle(frame, (x, badge_y1), (x + lw + 10, badge_y2), primary_color, -1)
        cv2.putText(
            frame,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # Render eyes
        for (ex, ey, ew, eh) in face["eyes"]:
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), eye_color, 1)
            cv2.putText(
                frame,
                "Eye",
                (ex, max(0, ey - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                eye_color,
                1,
                cv2.LINE_AA,
            )

        # Render smiles
        for (sx, sy, sw, sh) in face["smiles"]:
            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), smile_color, 1)
            cv2.putText(
                frame,
                "Smile",
                (sx, sy + sh + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                smile_color,
                1,
                cv2.LINE_AA,
            )

    return frame


def render_hud(frame, fps=None, detector="dnn", face_count=0, eyes_on=False, smiles_on=False, toast_msg=""):
    """Render modern translucent HUD header and interactive status."""
    h, w = frame.shape[:2]

    # Translucent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (20, 20, 20), -1)
    # Bottom controls bar
    cv2.rectangle(overlay, (0, h - 30), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Top Bar Text
    info_parts = []
    if fps is not None:
        info_parts.append(f"FPS: {fps:.1f}")
    info_parts.append(f"Detector: {detector.upper()}")
    info_parts.append(f"Faces: {face_count}")
    info_parts.append(f"Eyes: {'ON' if eyes_on else 'OFF'}")
    info_parts.append(f"Smiles: {'ON' if smiles_on else 'OFF'}")
    top_text = "  |  ".join(info_parts)

    cv2.putText(
        frame,
        top_text,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Bottom Bar Shortcut Hints
    controls_text = "[s] Snapshot   [d] Switch Detector   [e] Toggle Eyes   [m] Toggle Smiles   [q/ESC] Quit"
    cv2.putText(
        frame,
        controls_text,
        (12, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # Temporary Toast Notification (e.g., snapshot saved)
    if toast_msg:
        (tw, th), _ = cv2.getTextSize(toast_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx = (w - tw) // 2
        ty = 85
        # Draw toast box
        cv2.rectangle(frame, (tx - 15, ty - th - 10), (tx + tw + 15, ty + 10), (0, 140, 0), -1)
        cv2.rectangle(frame, (tx - 15, ty - th - 10), (tx + tw + 15, ty + 10), (255, 255, 255), 1)
        cv2.putText(
            frame,
            toast_msg,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def save_snapshot(frame, directory=CAPTURES_DIR):
    """Save the current frame to the captures directory with timestamp."""
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    filename = f"snapshot_{timestamp}.jpg"
    filepath = os.path.join(directory, filename)
    cv2.imwrite(filepath, frame)
    print(f"[Snapshot] Saved to: {filepath}")
    return filepath


def process_frame_pipeline(frame, model_mgr, detector="dnn", conf_threshold=0.5, detect_eyes=False, detect_smiles=False):
    """Core detection pipeline per frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if detector == "dnn":
        try:
            dnn_net = model_mgr.load_dnn()
            faces = detect_faces_dnn(dnn_net, frame, conf_threshold=conf_threshold)
        except Exception as e:
            print(f"[DNN Error] Falling back to Haar: {e}")
            face_cascade = model_mgr.load_haar_face()
            faces = detect_faces_haar(face_cascade, gray)
            detector = "haar"
    else:
        face_cascade = model_mgr.load_haar_face()
        faces = detect_faces_haar(face_cascade, gray)

    eye_cascade = model_mgr.load_haar_eyes() if detect_eyes else None
    smile_cascade = model_mgr.load_haar_smiles() if detect_smiles else None

    annotated_faces = detect_hierarchical_features(frame, gray, faces, eye_cascade, smile_cascade)
    render_detections(frame, annotated_faces, detector_mode=detector)

    return frame, len(faces), detector


def run_webcam(args, model_mgr):
    """Run real-time webcam face detection with live controls and FPS counter."""
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {args.camera}")

    # Set preferred resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "=" * 55)
    print("  Face Detection Running (Webcam)")
    print("  Controls:")
    print("    [s] Save snapshot to captures/")
    print("    [d] Toggle between Haar and DNN detector")
    print("    [e] Toggle eye detection")
    print("    [m] Toggle smile detection")
    print("    [q] or [ESC] Quit")
    print("=" * 55 + "\n")

    current_detector = args.detector
    detect_eyes = args.eyes
    detect_smiles = args.smiles

    fps = 0.0
    prev_time = time.perf_counter()
    toast_message = ""
    toast_expiry = 0.0

    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        ret, sample_frame = cap.read()
        if ret:
            sh, sw = sample_frame.shape[:2]
            out_writer = cv2.VideoWriter(args.output, fourcc, 20.0, (sw, sh))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Error] Failed to read frame from webcam.")
                break

            current_time = time.perf_counter()
            delta = current_time - prev_time
            prev_time = current_time
            if delta > 0:
                current_fps = 1.0 / delta
                fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps

            # Process frame
            processed_frame, face_count, current_detector = process_frame_pipeline(
                frame,
                model_mgr,
                detector=current_detector,
                conf_threshold=args.confidence,
                detect_eyes=detect_eyes,
                detect_smiles=detect_smiles,
            )

            # Check toast expiry
            if time.time() > toast_expiry:
                toast_message = ""

            # Render HUD
            render_hud(
                processed_frame,
                fps=fps if not args.no_fps else None,
                detector=current_detector,
                face_count=face_count,
                eyes_on=detect_eyes,
                smiles_on=detect_smiles,
                toast_msg=toast_message,
            )

            if out_writer:
                out_writer.write(processed_frame)

            cv2.imshow("Face Detection - Webcam", processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key == ord("s"):
                filepath = save_snapshot(processed_frame)
                toast_message = f"Snapshot Saved: {os.path.basename(filepath)}"
                toast_expiry = time.time() + 2.0
            elif key == ord("d"):
                current_detector = "haar" if current_detector == "dnn" else "dnn"
                print(f"[Detector switched to: {current_detector.upper()}]")
            elif key == ord("e"):
                detect_eyes = not detect_eyes
                print(f"[Eye Detection: {'ON' if detect_eyes else 'OFF'}]")
            elif key == ord("m"):
                detect_smiles = not detect_smiles
                print(f"[Smile Detection: {'ON' if detect_smiles else 'OFF'}]")

    finally:
        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()


def run_video(args, model_mgr):
    """Process video file with real-time HUD and controls."""
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Video file not found: {args.source}")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {args.source}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    delay_ms = max(1, int(1000 / (video_fps if video_fps > 0 else 30)))

    current_detector = args.detector
    detect_eyes = args.eyes
    detect_smiles = args.smiles

    fps = 0.0
    prev_time = time.perf_counter()
    toast_message = ""
    toast_expiry = 0.0

    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        ret, sample = cap.read()
        if ret:
            vh, vw = sample.shape[:2]
            out_writer = cv2.VideoWriter(args.output, fourcc, video_fps or 25.0, (vw, vh))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Info] End of video.")
                break

            current_time = time.perf_counter()
            delta = current_time - prev_time
            prev_time = current_time
            if delta > 0:
                current_fps = 1.0 / delta
                fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps

            processed_frame, face_count, current_detector = process_frame_pipeline(
                frame,
                model_mgr,
                detector=current_detector,
                conf_threshold=args.confidence,
                detect_eyes=detect_eyes,
                detect_smiles=detect_smiles,
            )

            if time.time() > toast_expiry:
                toast_message = ""

            render_hud(
                processed_frame,
                fps=fps if not args.no_fps else None,
                detector=current_detector,
                face_count=face_count,
                eyes_on=detect_eyes,
                smiles_on=detect_smiles,
                toast_msg=toast_message,
            )

            if out_writer:
                out_writer.write(processed_frame)

            cv2.imshow("Face Detection - Video", processed_frame)

            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                filepath = save_snapshot(processed_frame)
                toast_message = f"Snapshot Saved: {os.path.basename(filepath)}"
                toast_expiry = time.time() + 2.0
            elif key == ord("d"):
                current_detector = "haar" if current_detector == "dnn" else "dnn"
            elif key == ord("e"):
                detect_eyes = not detect_eyes
            elif key == ord("m"):
                detect_smiles = not detect_smiles

    finally:
        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()


def run_image(args, model_mgr):
    """Detect faces on a still image and display/save the result."""
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Image not found: {args.source}")

    image = cv2.imread(args.source)
    if image is None:
        raise RuntimeError(f"Could not decode image at: {args.source}")

    processed_frame, face_count, detector = process_frame_pipeline(
        image,
        model_mgr,
        detector=args.detector,
        conf_threshold=args.confidence,
        detect_eyes=args.eyes,
        detect_smiles=args.smiles,
    )

    render_hud(
        processed_frame,
        fps=None,
        detector=detector,
        face_count=face_count,
        eyes_on=args.eyes,
        smiles_on=args.smiles,
    )

    print(f"\n[Detection Complete] Detected {face_count} face(s) using {detector.upper()}.")
    print("  Press 's' to save snapshot, or any other key / 'q' to exit.")

    if args.output:
        cv2.imwrite(args.output, processed_frame)
        print(f"[Saved] Output written to {args.output}")

    if not args.no_show:
        cv2.imshow("Face Detection - Image", processed_frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("s"):
            save_snapshot(processed_frame)
        cv2.destroyAllWindows()


def build_parser():
    """Build command line argument parser with subcommands and flags."""
    parser = argparse.ArgumentParser(
        prog="face_detection",
        description="Advanced Face Detection System with OpenCV (Haar Cascades & ResNet-10 SSD DNN).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python face_detection.py webcam
  python face_detection.py webcam --detector dnn --eyes --smiles
  python face_detection.py image photo.jpg --detector dnn --eyes
  python face_detection.py video clip.mp4 --detector haar --output output.mp4

Interactive Hotkeys (Webcam / Video):
  [s]  Capture snapshot to captures/ folder
  [d]  Toggle between Haar and DNN detector
  [e]  Toggle Eye detection
  [m]  Toggle Smile detection
  [q]  Quit
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Input source mode")

    # Common parent parser for shared arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--detector",
        choices=["haar", "dnn"],
        default="dnn",
        help="Detection algorithm: 'dnn' (ResNet-10 SSD, accurate) or 'haar' (classical Haar Cascades). Default: dnn",
    )
    common.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for DNN detection (0.0 to 1.0). Default: 0.5",
    )
    common.add_argument(
        "--eyes",
        action="store_true",
        help="Enable hierarchical eye detection within face ROI.",
    )
    common.add_argument(
        "--smiles",
        action="store_true",
        help="Enable hierarchical smile detection within lower face ROI.",
    )
    common.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to save processed image or video.",
    )
    common.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open GUI display window (ideal for headless testing or batch processing).",
    )

    # Webcam mode
    parser_webcam = subparsers.add_parser("webcam", parents=[common], help="Live detection using webcam")
    parser_webcam.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index. Default: 0",
    )
    parser_webcam.add_argument(
        "--no-fps",
        action="store_true",
        help="Hide FPS counter in HUD overlay.",
    )

    # Image mode
    parser_image = subparsers.add_parser("image", parents=[common], help="Still image detection")
    parser_image.add_argument(
        "source",
        type=str,
        help="Path to input image file (JPG, PNG, etc.)",
    )

    # Video mode
    parser_video = subparsers.add_parser("video", parents=[common], help="Video file detection")
    parser_video.add_argument(
        "source",
        type=str,
        help="Path to input video file (MP4, AVI, MOV, etc.)",
    )
    parser_video.add_argument(
        "--no-fps",
        action="store_true",
        help="Hide FPS counter in HUD overlay.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    model_mgr = ModelManager()

    if args.mode == "webcam":
        run_webcam(args, model_mgr)
    elif args.mode == "image":
        run_image(args, model_mgr)
    elif args.mode == "video":
        run_video(args, model_mgr)


if __name__ == "__main__":
    main()