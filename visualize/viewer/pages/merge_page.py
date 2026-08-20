from pathlib import Path

import numpy
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from constants.label import Label
from constants.mode import Mode
from post_processing.exclude_truncated import exclude_truncated
from post_processing.refine_merge import refine_merge
from visualize.ui.checkboxes import IconCheckboxTable
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.views import VolumeViewer


class MergeControls(HorizontalLayout):
    def _init_widgets(self):
        return [
            QPushButton('Preview'),
            QPushButton('Clear'),
            QPushButton('Save'),
            QLabel('Saved: (none)'),
            None,
            QPushButton('Save and Next'),
        ]

class MergeContent(HorizontalLayout):
    def _init_widgets(self):
        return [
            VolumeViewer(2, (512, 512)),
            IconCheckboxTable(),
        ]

class MergePage(QWidget):
    next_requested = Signal(str)

    def __init__(self, working_dir=Path('working')):
        super().__init__()
        self.working_dir = working_dir

        self.controls = MergeControls()
        self.content = MergeContent()
        self.viewer, self.label_table = self.content.widgets
        self.viewer.set_titles(Mode.get_title(Mode.REFINE_SPLIT), 'Preview')

        layout = QVBoxLayout(self)
        layout.addLayout(self.controls)
        layout.addLayout(self.content)

        self.preview_button, self.clear_button, self.save_button, self.saved_label, self.next_button = self.controls.widgets

        self.preview_button.clicked.connect(self._on_preview)
        self.clear_button.clicked.connect(self._on_clear)
        self.save_button.clicked.connect(self._on_save)
        self.next_button.clicked.connect(self._on_next)
        self.viewer.views[0].picked.connect(self._on_picked)

        self.patient = None
        self.volume = None
        self.merge_components = []

    @property
    def is_ready(self):
        return self.volume is not None

    def closeEvent(self, event):
        super().closeEvent(event)
        self.viewer.close()

    def set_patient(self, patient):
        self.patient = patient
        self.volume = None
        self.merge_components = []
        self._update_saved_label()
        self.viewer.views[0].clear_actors()
        self.viewer.views[1].clear_actors()

        if not patient:
            return

        volume_path = self.working_dir / patient / f'{Mode.REFINE_SPLIT}.npy'
        if not volume_path.is_file():
            return

        self.volume = numpy.load(volume_path)
        self.viewer.views[0].update_volume(self.volume, include_labels=True)

        labels = numpy.unique(self.volume)
        labels = labels[labels > 0]

        self.label_table.update_items(labels)

        self._on_preview()

    def _on_preview(self):
        if self.volume is None:
            return

        labels = self.label_table.selected_labels()
        volume = numpy.where(numpy.isin(self.volume, labels), Label.TOOTH, Label.BACKGROUND)
        self.viewer.views[1].update_volume(volume, reset=False)

    def _on_clear(self):
        if self.volume is None:
            return

        self.label_table.clear_selected()
        self._on_preview()

    def _on_save(self):
        labels = self.label_table.selected_labels()
        if len(labels) < 2:
            return

        self.merge_components.append(sorted(labels))
        self._update_saved_label()

    def _update_saved_label(self):
        if not self.merge_components:
            self.saved_label.setText('Saved: (none)')
            return

        groups = ', '.join('+'.join(str(label) for label in group) for group in self.merge_components)
        self.saved_label.setText(f'Saved: {groups}')

    def _on_next(self):
        if self.volume is None:
            return

        patient_dir = self.working_dir / self.patient
        volume = self.volume
        if self.merge_components:
            volume = refine_merge(self.volume, self.merge_components)

        untruncated_volume = exclude_truncated(volume)
        numpy.save(patient_dir / f'{Mode.UNTRUNCATED}.npy', untruncated_volume)

        self.next_requested.emit(self.patient)

    def _on_picked(self, actor):
        label = actor._label
        checkbox = self.label_table.checkboxes[label]
        checkbox.setChecked(True)
