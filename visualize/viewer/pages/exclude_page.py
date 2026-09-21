from pathlib import Path

import numpy
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from constants.label import Label
from constants.mode import Mode
from post_processing.auto_detect import detect_fake_labels, detect_unerupted_labels
from post_processing.exclude_tooth import exclude_tooth
from post_processing.fill_holes import fill_holes
from post_processing.relabel import relabel_volume
from visualize.ui.buttons import RadioGroup
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.selectors import IconLabelSelector
from visualize.ui.views import VolumeViewer

EXCLUDE_TYPES = ['Normal Tooth', 'Unerupted Tooth', 'Residual Root', 'Fake Tooth', 'Noise']

EXCLUDE_COLORS = {
    'Unerupted Tooth': (220, 70, 70),
    'Residual Root': (70, 220, 70),
    'Fake Tooth': (70, 70, 220),
    'Noise': (0, 0, 0),
}


class ExcludeControls(HorizontalLayout):
    def _init_widgets(self):
        return [
            IconLabelSelector(),
            RadioGroup(EXCLUDE_TYPES),
            QPushButton('Save'),
            QLabel('Saved: (none)'),
            None,
            QPushButton('Save and Next'),
        ]

class ExcludePage(QWidget):
    next_requested = Signal(str)

    def __init__(self, working_dir=Path('working')):
        super().__init__()
        self.working_dir = working_dir

        self.controls = ExcludeControls()
        self.viewer = VolumeViewer()
        self.viewer.set_titles(Mode.get_title(Mode.UNTRUNCATED), 'Instance')

        layout = QVBoxLayout(self)
        layout.addLayout(self.controls)
        layout.addWidget(self.viewer)

        self.label_selector, self.exclude_selector, self.save_button, self.saved_label, self.next_button = self.controls.widgets
        self.exclude_selector.select_first()

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.save_button.clicked.connect(self._on_save)
        self.next_button.clicked.connect(self._on_next)
        self.viewer.views[0].picked.connect(self._on_picked)

        self.patient = None
        self.volume = None
        self.bone_volume = None
        self.exclude_labels = {}

    @property
    def is_ready(self):
        return self.volume is not None

    def closeEvent(self, event):
        super().closeEvent(event)
        self.viewer.close()

    def set_patient(self, patient):
        self.patient = patient
        self.volume = None
        self.bone_volume = None
        self.exclude_labels = {}
        self._update_saved_label()
        self.viewer.views[0].clear_actors()
        self.viewer.views[1].clear_actors()

        if not patient:
            return

        patient_dir = self.working_dir / patient
        untruncated_path = patient_dir / f'{Mode.UNTRUNCATED}.npy'
        bone_path = patient_dir / f'{Mode.BONE_CONNECTED_COMPONENT}.npy'
        if not (untruncated_path.is_file() and bone_path.is_file()):
            return

        self.volume = numpy.load(untruncated_path)
        self.bone_volume = numpy.load(bone_path)
        bone_mask = self.bone_volume > 0

        display = numpy.where(self.volume >= Label.TRUNCATED, 0, self.volume)
        display = display + bone_mask
        self.viewer.views[0].update_volume(display, include_labels=True)

        labels = numpy.unique(self.volume)
        labels = labels[(labels > 0) & (labels < Label.TRUNCATED)]

        unmarked_labels = [int(label) for label in labels if self._find_exclude_type(int(label)) is None]
        for label in detect_unerupted_labels(self.volume, bone_mask, unmarked_labels):
            self.exclude_labels.setdefault('Unerupted Tooth', []).append(label)

        for label in detect_fake_labels(self.volume, bone_mask, unmarked_labels):
            self.exclude_labels.setdefault('Fake Tooth', []).append(label)

        for label in labels:
            color = EXCLUDE_COLORS.get(self._find_exclude_type(int(label)))
            if color is None:
                continue

            actor = self.viewer.views[0].actors.get(f'label {int(label)}')
            if actor is not None:
                actor.prop.color = tuple(c / 255 for c in color)

        self.viewer.views[0].render()

        self._update_saved_label()

        self.label_selector.update_items(labels)
        self.label_selector.setCurrentIndex(0)

    def _on_label_changed(self, index):
        if self.volume is None:
            return

        label = self.label_selector.currentData()
        if label is None:
            return

        volume = self.volume == label
        volume = volume * Label.TOOTH
        self.viewer.views[1].update_volume(volume, reset=False)

        exclude_type = self._find_exclude_type(label)
        if exclude_type is None:
            exclude_type = 'Normal Tooth'
        self.exclude_selector.select_title(exclude_type)

    def _find_exclude_type(self, label):
        for exclude_type, labels in self.exclude_labels.items():
            if label in labels:
                return exclude_type

        return None

    def _on_save(self):
        if self.volume is None:
            return

        label = self.label_selector.currentData()
        if label is None:
            return

        exclude_type = self.exclude_selector.checked_title()
        origin_type = self._find_exclude_type(label)
        if origin_type is not None:
            self.exclude_labels[origin_type].remove(label)
            if not self.exclude_labels[origin_type]:
                del self.exclude_labels[origin_type]

        if exclude_type != 'Normal Tooth':
            self.exclude_labels.setdefault(exclude_type, [])
            self.exclude_labels[exclude_type].append(label)

        self._update_saved_label()

    def _update_saved_label(self):
        if not self.exclude_labels:
            self.saved_label.setText('Saved: (none)')
            return

        parts = ', '.join(f'{exclude_type}{labels}' for exclude_type, labels in self.exclude_labels.items())
        self.saved_label.setText(f'Saved: {parts}')

    def _on_next(self):
        if self.volume is None:
            return

        patient_dir = self.working_dir / self.patient
        volume = self.volume
        if self.exclude_labels:
            volume = exclude_tooth(self.volume, self.exclude_labels)

        tooth_filled, bone_filled = fill_holes(volume, self.bone_volume)
        numpy.save(patient_dir / f'{Mode.TOOTH_FILLED}.npy', tooth_filled)
        numpy.save(patient_dir / f'{Mode.BONE_FILLED}.npy', bone_filled)

        relabeled = relabel_volume(tooth_filled, bone_filled)
        numpy.save(patient_dir / f'{Mode.RELABELED}.npy', relabeled)

        self.next_requested.emit(self.patient)

    def _on_picked(self, actor):
        label = actor._label
        index = self.label_selector.findData(label)
        self.label_selector.setCurrentIndex(index)
