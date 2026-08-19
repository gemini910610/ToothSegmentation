from pathlib import Path

import numpy
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from constants.mode import Mode
from post_processing.refine_split import refine_split
from post_processing.utils import ROIVolume
from post_processing.watershed import split_k_component
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.selectors import IconLabelSelector
from visualize.ui.views import VolumeViewer


class SplitControls(HorizontalLayout):
    def _init_widgets(self):
        return [
            IconLabelSelector(),
            QSpinBox(suffix=' Components', minimum=2),
            QPushButton('Execute'),
            QPushButton('Save'),
            QPushButton('Clear'),
            QLabel('Saved: (none)'),
            None,
            QPushButton('Save and Next'),
        ]

class SplitPage(QWidget):
    next_requested = Signal(str)

    def __init__(self, working_dir=Path('working')):
        super().__init__()
        self.working_dir = working_dir

        self.controls = SplitControls()
        self.viewer = VolumeViewer(3, (512, 512))
        self.viewer.set_titles(Mode.get_title(Mode.TOOTH_CONNECTED_COMPONENT), Mode.get_title(Mode.WATERSHED), 'Instance')

        layout = QVBoxLayout(self)
        layout.addLayout(self.controls)
        layout.addWidget(self.viewer)

        self.label_selector, self.count_input, self.execute_button, self.save_button, self.clear_button, self.saved_label, self.next_button = self.controls.widgets

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.execute_button.clicked.connect(self._on_execute)
        self.save_button.clicked.connect(self._on_save)
        self.clear_button.clicked.connect(self._on_clear)
        self.next_button.clicked.connect(self._on_next)
        self.viewer.views[0].picked.connect(self._on_picked)

        self.patient = None
        self.tooth_volume = None
        self.watershed_volume = None
        self.label_volume = None
        self.split_components = {}

    @property
    def is_ready(self):
        return self.tooth_volume is not None

    def closeEvent(self, event):
        super().closeEvent(event)
        self.viewer.close()

    def set_patient(self, patient):
        self.patient = patient
        self.tooth_volume = None
        self.watershed_volume = None
        self.label_volume = None
        self.split_components = {}
        self._update_saved_label()
        for view in self.viewer.views:
            view.clear_actors()

        if not patient:
            return

        patient_dir = self.working_dir / patient
        tooth_path = patient_dir / f'{Mode.TOOTH_CONNECTED_COMPONENT}.npy'
        watershed_path = patient_dir / f'{Mode.WATERSHED}.npy'
        if not (tooth_path.exists() and watershed_path.exists()):
            return

        self.tooth_volume = numpy.load(tooth_path)
        self.watershed_volume = numpy.load(watershed_path)

        self.viewer.views[0].update_volume(self.tooth_volume, include_labels=True)
        self.viewer.views[1].update_volume(self.watershed_volume, include_labels=True)

        labels = numpy.unique(self.tooth_volume)
        labels = labels[labels > 0]

        self.label_selector.update_items(labels)
        self.label_selector.setCurrentIndex(0)

    def _on_label_changed(self, index):
        if self.tooth_volume is None:
            return

        label = self.label_selector.currentData()
        if label is None:
            return

        self.label_volume = self.tooth_volume == label
        volume = (self.label_volume * label).astype(numpy.uint8)
        self.viewer.views[2].update_volume(volume, include_labels=True, reset=False)

        count = self.split_components.get(label)
        if count is None:
            self.count_input.setValue(self.count_input.minimum())
        else:
            self.count_input.setValue(count)

        self._on_execute()

    def _on_execute(self):
        if self.label_volume is None:
            return

        count = self.count_input.value()

        roi = ROIVolume(self.label_volume)
        volume = split_k_component(roi.volume, h=1, k=count)
        volume[volume > 0] += 1
        roi.volume = volume
        volume = roi.restore()
        self.viewer.views[2].update_volume(volume, include_labels=True, reset=False)

    def _on_save(self):
        label = self.label_selector.currentData()
        if label is None:
            return

        self.split_components[label] = self.count_input.value()
        self._update_saved_label()

    def _on_clear(self):
        label = self.label_selector.currentData()
        if label is None:
            return

        self.split_components.pop(label, None)
        self._update_saved_label()

    def _update_saved_label(self):
        if not self.split_components:
            self.saved_label.setText('Saved: (none)')
            return

        items = ', '.join(f'{label}→{count}' for label, count in sorted(self.split_components.items()))
        self.saved_label.setText(f'Saved: {items}')

    def _on_next(self):
        if self.tooth_volume is None or self.watershed_volume is None:
            return

        patient_dir = self.working_dir / self.patient
        volume = self.watershed_volume
        if self.split_components:
            volume = refine_split(self.tooth_volume, self.watershed_volume, self.split_components)

        numpy.save(patient_dir / f'{Mode.REFINE_SPLIT}.npy', volume)

        self.next_requested.emit(self.patient)

    def _on_picked(self, actor):
        label = actor._label
        index = self.label_selector.findData(label)
        self.label_selector.setCurrentIndex(index)
