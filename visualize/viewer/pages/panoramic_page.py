from pathlib import Path

import cv2
from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget

from panoramic.predict import predict_panoramic
from post_processing.panoramic import (
    draw_panoramic,
    find_panoramic_image,
    load_panoramic,
)
from utils.image import imread
from visualize.ui.images import AutoFitImage, bgr_to_pixmap
from visualize.ui.layouts import HorizontalLayout

PREDICTION_FILENAME = 'panoramic_predict.txt'


class PanoramicControls(HorizontalLayout):
    def _init_widgets(self):
        return [
            QPushButton('Predict'),
            None,
            QPushButton('Next'),
        ]

class PanoramicPage(QWidget):
    next_requested = Signal(str)

    def __init__(self, panoramic_models, working_dir=Path('working')):
        super().__init__()
        self.panoramic_models = panoramic_models
        self.working_dir = working_dir

        self.controls = PanoramicControls()
        self.image_view = AutoFitImage()

        layout = QVBoxLayout(self)
        layout.addLayout(self.controls)
        layout.addWidget(self.image_view)

        self.predict_button, self.next_button = self.controls.widgets

        self.predict_button.clicked.connect(self._on_predict)
        self.next_button.clicked.connect(self._on_next)

        self.patient = None
        self.image_path = None
        self.prediction_path = None

    @property
    def is_ready(self):
        return self.image_path is not None

    def sizeHint(self):
        size = super().sizeHint()
        screen_width = QApplication.primaryScreen().availableGeometry().width()

        return QSize(screen_width // 2, size.height())

    def set_patient(self, patient):
        self.patient = patient
        self.image_path = None
        self.prediction_path = None

        if not patient:
            return

        image_path = find_panoramic_image(self.working_dir, patient)
        if image_path is None:
            return

        self.image_path = image_path
        self.prediction_path = self.working_dir / patient / PREDICTION_FILENAME

        if self.prediction_path.is_file():
            self._display()
        else:
            image = imread(self.image_path)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.image_view.update_image(bgr_to_pixmap(image))

    def _on_predict(self):
        if self.image_path is None:
            return

        self.prediction_path.parent.mkdir(parents=True, exist_ok=True)
        predict_panoramic(self.image_path, self.panoramic_models, self.prediction_path)

        self._display()

    def _display(self):
        image, tooth_data = load_panoramic(self.image_path, self.prediction_path)
        image = draw_panoramic(image, tooth_data, draw_keypoints=True)
        self.image_view.update_image(bgr_to_pixmap(image))

    def _on_next(self):
        if self.prediction_path is None or not self.prediction_path.is_file():
            return

        self.next_requested.emit(self.patient)
