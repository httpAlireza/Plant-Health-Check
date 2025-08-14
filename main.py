import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QSize

from disease_classifier import DiseaseClassifier
from leaf_detector import LeafDetector


def _cvimg_to_qpixmap(cv_img, target_size: QSize):
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _generate_distinct_color(index, total):
    hue = int(180 * (index / max(1, total)))
    color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in color)


class DiseaseDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plant Disease Detection")
        self.cv_img = None
        self.annotated_img = None

        # GUI elements
        self.image_label = QLabel("No image loaded.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Arial", 12))

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.open_image)
        self.detect_button = QPushButton("Detect Disease")
        self.detect_button.clicked.connect(self.detect_disease)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_app)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.detect_button)
        button_layout.addWidget(self.reset_button)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(button_layout)
        layout.addWidget(self.results_text)
        self.setLayout(layout)
        self.resize(800, 600)

        # Models
        self.leaf_detector = LeafDetector()
        self.disease_classifier = DiseaseClassifier()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display_image()

    def update_display_image(self):
        if self.cv_img is not None:
            img_to_show = self.annotated_img if self.annotated_img is not None else self.cv_img
            pixmap = _cvimg_to_qpixmap(img_to_show, self.image_label.size())
            self.image_label.setPixmap(pixmap)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if file_path:
            self.cv_img = cv2.imread(file_path)
            self.annotated_img = None
            self.results_text.clear()
            self.update_display_image()

    def detect_disease(self):
        if self.cv_img is None:
            return

        self.results_text.clear()
        self.annotated_img = self.cv_img.copy()

        start_time = time.time()
        boxes = self.leaf_detector.detect(self.cv_img)
        leaf_detection_time = time.time() - start_time

        if not boxes:
            self.results_text.setText("No leaves detected.")
            return

        results = self.process_leaves(boxes)

        total_leaves = len(boxes)
        self.results_text.setText(
            f"{total_leaves} {'Leaf' if total_leaves == 1 else 'Leaves'} detected in {leaf_detection_time:.3f} sec.\n" +
            "\n".join(results)
        )

        self.update_display_image()

    def process_leaves(self, boxes):
        leaf_imgs = []
        leaf_confs = []

        for coords, leaf_conf in boxes:
            x1, y1, x2, y2 = coords
            leaf_img = self.cv_img[y1:y2, x1:x2]
            leaf_imgs.append(leaf_img)
            leaf_confs.append(leaf_conf)

        if not leaf_imgs:
            return []

        results = []

        start_time = time.time()
        classifications = self.disease_classifier.classify(leaf_imgs)
        disease_classification_time = time.time() - start_time

        results.append(f"Disease classification took {disease_classification_time:.3f} sec.\n")
        total_leaves = len(leaf_imgs)
        for i, ((pred_class, disease_conf), leaf_conf) in enumerate(zip(classifications, leaf_confs)):
            x1, y1, x2, y2 = boxes[i][0]
            self.draw_leaf_annotation(i, total_leaves, x1, y1, x2, y2, pred_class)
            results.append(
                f"Leaf {i + 1}: {pred_class} {disease_conf * 100:.2f}% - "
                f"(Leaf confidence: {leaf_conf * 100:.2f}%)"
            )

        return results

    def draw_leaf_annotation(self, leaf_index, total_leaves, x1, y1, x2, y2, pred_class):
        color = _generate_distinct_color(leaf_index, total_leaves)
        thickness = max(1, min(self.cv_img.shape[:2]) // 200)
        font_scale = max(0.7, min(self.cv_img.shape[:2]) / 800)

        text = f'{leaf_index + 1}- {pred_class}'
        cv2.rectangle(self.annotated_img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(self.annotated_img, text, (x1, y1 + int(30 * font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    def reset_app(self):
        self.cv_img = None
        self.annotated_img = None
        self.image_label.setText("No image loaded.")
        self.results_text.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DiseaseDetectionApp()
    window.show()
    sys.exit(app.exec())
