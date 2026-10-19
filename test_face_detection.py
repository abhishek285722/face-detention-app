"""Unit test suite for Real-Time Face Detection & Feature Tracking."""
import os
import tempfile
import unittest

import cv2
import numpy as np

from face_detection import (
    BASE_DIR,
    HAAR_EYE_PATH,
    HAAR_FACE_PATH,
    HAAR_SMILE_PATH,
    TF_PB_PATH,
    TF_PBTXT_PATH,
    ModelManager,
    detect_faces_dnn,
    detect_faces_haar,
    detect_hierarchical_features,
    process_frame_pipeline,
    render_detections,
    render_hud,
    save_snapshot,
)


class TestFaceDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_mgr = ModelManager()
        cls.sample_image_path = os.path.join(BASE_DIR, "sample_face.jpg")
        cls.sample_image = cv2.imread(cls.sample_image_path)
        cls.gray_image = cv2.cvtColor(cls.sample_image, cv2.COLOR_BGR2GRAY)

    def test_01_model_files_exist(self):
        """Verify all essential Haar and DNN model files exist in the project."""
        self.assertTrue(os.path.exists(HAAR_FACE_PATH), f"Missing: {HAAR_FACE_PATH}")
        self.assertTrue(os.path.exists(HAAR_EYE_PATH), f"Missing: {HAAR_EYE_PATH}")
        self.assertTrue(os.path.exists(HAAR_SMILE_PATH), f"Missing: {HAAR_SMILE_PATH}")
        self.assertTrue(
            (os.path.exists(TF_PB_PATH) and os.path.exists(TF_PBTXT_PATH)),
            "Missing TensorFlow SSD DNN model weights/config.",
        )

    def test_02_model_manager_loading(self):
        """Verify ModelManager initializes and loads Haar and DNN engines without errors."""
        haar_face = self.model_mgr.load_haar_face()
        self.assertIsNotNone(haar_face)
        self.assertFalse(haar_face.empty())

        haar_eyes = self.model_mgr.load_haar_eyes()
        self.assertIsNotNone(haar_eyes)
        self.assertFalse(haar_eyes.empty())

        haar_smiles = self.model_mgr.load_haar_smiles()
        self.assertIsNotNone(haar_smiles)
        self.assertFalse(haar_smiles.empty())

        dnn_net = self.model_mgr.load_dnn()
        self.assertIsNotNone(dnn_net)

    def test_03_sample_image_available(self):
        """Verify test image exists and is readable."""
        self.assertTrue(os.path.exists(self.sample_image_path), "sample_face.jpg must exist.")
        self.assertIsNotNone(self.sample_image)
        self.assertGreater(self.sample_image.shape[0], 0)
        self.assertGreater(self.sample_image.shape[1], 0)

    def test_04_dnn_face_detection(self):
        """Verify DNN SSD detector detects face in sample image with high confidence."""
        dnn_net = self.model_mgr.load_dnn()
        faces = detect_faces_dnn(dnn_net, self.sample_image, conf_threshold=0.5)
        self.assertGreaterEqual(len(faces), 1, "DNN should detect at least 1 face.")

        x, y, w, h, conf = faces[0]
        self.assertGreater(w, 50)
        self.assertGreater(h, 50)
        self.assertGreater(conf, 0.70, "Confidence should exceed 70% for sample face.")

    def test_05_haar_face_detection(self):
        """Verify Haar cascade detects face in sample image."""
        face_cascade = self.model_mgr.load_haar_face()
        faces = detect_faces_haar(face_cascade, self.gray_image)
        self.assertGreaterEqual(len(faces), 1, "Haar should detect at least 1 face.")

        x, y, w, h, conf = faces[0]
        self.assertGreater(w, 50)
        self.assertGreater(h, 50)
        self.assertIsNone(conf)

    def test_06_hierarchical_features(self):
        """Verify eye and smile detection within detected face ROI."""
        dnn_net = self.model_mgr.load_dnn()
        eye_cascade = self.model_mgr.load_haar_eyes()
        smile_cascade = self.model_mgr.load_haar_smiles()

        faces = detect_faces_dnn(dnn_net, self.sample_image, conf_threshold=0.5)
        annotated = detect_hierarchical_features(
            self.sample_image,
            self.gray_image,
            faces,
            eye_cascade=eye_cascade,
            smile_cascade=smile_cascade,
        )

        self.assertEqual(len(annotated), len(faces))
        first_face = annotated[0]
        self.assertIn("box", first_face)
        self.assertIn("eyes", first_face)
        self.assertIn("smiles", first_face)

    def test_07_blank_image_handling(self):
        """Verify robustness when processing an all-black image."""
        blank = np.zeros((300, 300, 3), dtype=np.uint8)
        blank_gray = np.zeros((300, 300), dtype=np.uint8)

        dnn_net = self.model_mgr.load_dnn()
        faces_dnn = detect_faces_dnn(dnn_net, blank, conf_threshold=0.5)
        self.assertEqual(len(faces_dnn), 0)

        face_cascade = self.model_mgr.load_haar_face()
        faces_haar = detect_faces_haar(face_cascade, blank_gray)
        self.assertEqual(len(faces_haar), 0)

    def test_08_render_detections(self):
        """Verify rendering function generates annotated image without shape alteration."""
        dnn_net = self.model_mgr.load_dnn()
        faces = detect_faces_dnn(dnn_net, self.sample_image, conf_threshold=0.5)
        annotated = detect_hierarchical_features(self.sample_image, self.gray_image, faces)

        canvas = self.sample_image.copy()
        render_detections(canvas, annotated, detector_mode="dnn")
        self.assertEqual(canvas.shape, self.sample_image.shape)

    def test_09_render_hud(self):
        """Verify HUD overlay renders cleanly with all status fields and toasts."""
        canvas = self.sample_image.copy()
        result = render_hud(
            canvas,
            fps=32.5,
            detector="dnn",
            face_count=1,
            eyes_on=True,
            smiles_on=True,
            toast_msg="Test Toast Alert",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.sample_image.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_10_save_snapshot(self):
        """Verify snapshot file creation, timestamped naming, and image integrity."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = save_snapshot(self.sample_image, directory=tmp_dir)
            self.assertTrue(os.path.exists(filepath), f"Snapshot file not created at {filepath}")
            self.assertTrue(os.path.basename(filepath).startswith("snapshot_"))
            self.assertTrue(filepath.endswith(".jpg"))

            saved_img = cv2.imread(filepath)
            self.assertIsNotNone(saved_img)
            self.assertEqual(saved_img.shape, self.sample_image.shape)

    def test_11_process_frame_pipeline(self):
        """Verify full per-frame processing pipeline across DNN and Haar modes."""
        # Test DNN pipeline
        canvas_dnn = self.sample_image.copy()
        res_dnn, count_dnn, det_dnn = process_frame_pipeline(
            canvas_dnn,
            self.model_mgr,
            detector="dnn",
            conf_threshold=0.5,
            detect_eyes=True,
            detect_smiles=True,
        )
        self.assertEqual(res_dnn.shape, self.sample_image.shape)
        self.assertGreaterEqual(count_dnn, 1)
        self.assertEqual(det_dnn, "dnn")

        # Test Haar pipeline
        canvas_haar = self.sample_image.copy()
        res_haar, count_haar, det_haar = process_frame_pipeline(
            canvas_haar,
            self.model_mgr,
            detector="haar",
            detect_eyes=True,
            detect_smiles=False,
        )
        self.assertEqual(res_haar.shape, self.sample_image.shape)
        self.assertGreaterEqual(count_haar, 1)
        self.assertEqual(det_haar, "haar")


if __name__ == "__main__":
    unittest.main()
