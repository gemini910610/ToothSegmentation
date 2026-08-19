from pathlib import Path

import numpy
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from constants.mode import Mode
from post_processing.connected_component import filter_connected_component
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.views import VolumeViewer


class ConnectedComponentControls(HorizontalLayout):
    def __init__(self, threshold_label, default_threshold):
        self._threshold_label = threshold_label
        self._default_threshold = default_threshold
        super().__init__()

    def _init_widgets(self):
        return [
            QLabel(self._threshold_label),
            QSpinBox(minimum=0, maximum=1000000, value=self._default_threshold),
            QPushButton('Filter'),
            None,
            QPushButton('Save and Next'),
        ]

class ConnectedComponentPage(QWidget):
    next_requested = Signal(str)

    label = None
    is_bone = None
    threshold_label = None
    default_threshold = None
    output_mode = None

    def __init__(self, working_dir=Path('working')):
        super().__init__()
        self.working_dir = working_dir

        self.controls = ConnectedComponentControls(self.threshold_label, self.default_threshold)
        self.viewer = VolumeViewer()
        self.viewer.set_titles('Original', 'Filtered')

        layout = QVBoxLayout(self)
        layout.addLayout(self.controls)
        layout.addWidget(self.viewer)

        _, self.threshold_input, self.filter_button, self.next_button = self.controls.widgets

        self.filter_button.clicked.connect(self._on_filter)
        self.next_button.clicked.connect(self._on_next)

        self.patient = None
        self.predict_volume = None
        self.filtered_volume = None

    @property
    def is_ready(self):
        return self.predict_volume is not None

    def closeEvent(self, event):
        super().closeEvent(event)
        self.viewer.close()

    def set_patient(self, patient):
        self.patient = patient
        self.predict_volume = None
        self.filtered_volume = None
        self.viewer.views[0].clear_actors()
        self.viewer.views[1].clear_actors()

        if not patient:
            return

        predict_path = self.working_dir / patient / f'{Mode.PREDICT}.npy'
        if not predict_path.exists():
            return

        self.predict_volume = numpy.load(predict_path)

        output_path = self.working_dir / patient / f'{self.output_mode}.npy'
        if output_path.exists():
            self.filtered_volume = numpy.load(output_path)
            original = (self.predict_volume == self.label).astype(numpy.uint8) * self.label
            self.viewer.views[0].update_volume(original)
            self.viewer.views[1].update_volume(self.filtered_volume, include_labels=True)
        else:
            self._compute()

    def _compute(self):
        threshold = self.threshold_input.value()

        self.filtered_volume = filter_connected_component(self.predict_volume == self.label, is_bone=self.is_bone, voxel_threshold=threshold)

        original = (self.predict_volume == self.label).astype(numpy.uint8) * self.label
        self.viewer.views[0].update_volume(original)
        self.viewer.views[1].update_volume(self.filtered_volume, include_labels=True)

    def _on_filter(self):
        if self.predict_volume is None:
            return

        self._compute()

    def _on_next(self):
        if self.filtered_volume is None:
            return

        patient_dir = self.working_dir / self.patient
        numpy.save(patient_dir / f'{self.output_mode}.npy', self.filtered_volume)

        self._after_save(patient_dir)

        self.next_requested.emit(self.patient)

    def _after_save(self, patient_dir):
        pass
