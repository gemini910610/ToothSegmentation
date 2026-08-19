from pathlib import Path

import numpy
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from constants.mode import Mode
from segmentation.predict import predict_patient
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.views import VolumeViewer


class PredictControls(HorizontalLayout):
    def _init_widgets(self):
        return [
            QPushButton('Predict'),
            QPushButton('Flip'),
            None,
            QPushButton('Save and Next'),
        ]

class PredictPage(QWidget):
    next_requested = Signal(str)

    def __init__(self, model, config, working_dir=Path('working')):
        super().__init__()
        self.model = model
        self.config = config
        self.working_dir = working_dir

        self.controls = PredictControls()
        self.viewer = VolumeViewer(1)

        layout = QVBoxLayout(self)
        layout.addLayout(self.controls)
        layout.addWidget(self.viewer)

        self.predict_button, self.flip_button, self.save_button = self.controls.widgets

        self.predict_button.clicked.connect(self._on_predict)
        self.flip_button.clicked.connect(self._on_flip)
        self.save_button.clicked.connect(self._on_save_and_next)

        self.patient = None
        self.image_volume = None
        self.predict_volume = None
        self.dirty = False

    @property
    def is_ready(self):
        return True

    def closeEvent(self, event):
        super().closeEvent(event)
        self.viewer.close()

    def set_patient(self, patient):
        self.patient = patient
        self.image_volume = None
        self.predict_volume = None
        self.dirty = False
        self.viewer.views[0].clear_actors()

        if patient and self._load_existing_predict():
            self._update_view()

    def _on_predict(self):
        if self.patient is None:
            return

        if not self._load_existing_predict():
            image_dir = self.working_dir / self.patient / 'image'
            self.image_volume, self.predict_volume = predict_patient(self.model, image_dir, self.config)
            self.dirty = True

        self._update_view()

    def _load_existing_predict(self):
        patient_dir = self.working_dir / self.patient
        image_path = patient_dir / f'{Mode.IMAGE}.npy'
        predict_path = patient_dir / f'{Mode.PREDICT}.npy'

        if not (image_path.exists() and predict_path.exists()):
            return False

        self.image_volume = numpy.load(image_path)
        self.predict_volume = numpy.load(predict_path)
        self.dirty = False

        return True

    def _on_flip(self):
        if self.predict_volume is None:
            return

        self.image_volume = numpy.flip(self.image_volume, axis=2).copy()
        self.predict_volume = numpy.flip(self.predict_volume, axis=2).copy()
        self.dirty = True
        self._update_view()

    def _on_save_and_next(self):
        if self.predict_volume is None:
            return

        if self.dirty:
            patient_dir = self.working_dir / self.patient
            numpy.save(patient_dir / f'{Mode.IMAGE}.npy', self.image_volume)
            numpy.save(patient_dir / f'{Mode.PREDICT}.npy', self.predict_volume)
            self.dirty = False

        self.next_requested.emit(self.patient)

    def _update_view(self):
        self.viewer.views[0].update_volume(self.predict_volume)
