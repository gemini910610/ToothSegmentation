from pathlib import Path

import numpy
from PySide6.QtWidgets import QPushButton, QVBoxLayout

from constants.color import GLASBEY_PALETTE, Color
from post_processing.refine_merge import refine_merge
from post_processing.tooth_data import load_tooth_data
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.views import VolumeViewer
from visualize.viewer.pages.page import VolumeViewerPage
from visualize.viewer.pipeline import BONE_FILENAME, TOOTH_FILENAME, apply_manual_edit


class MergeControls(HorizontalLayout):
    def _init_widgets(self):
        return [
            QPushButton('Clear'),
            None,
            QPushButton('Save'),
        ]

class MergePage(VolumeViewerPage):
    def __init__(self, working_dir=Path('working')):
        super().__init__(working_dir)

        self.controls = MergeControls()
        self.viewer = VolumeViewer(2, (512, 512))
        self.viewer.set_titles('Tooth', 'Preview')

        layout = QVBoxLayout(self)
        layout.addLayout(self.controls)
        layout.addWidget(self.viewer)

        self.clear_button, self.save_button = self.controls.widgets

        self.clear_button.clicked.connect(self._on_clear)
        self.save_button.clicked.connect(self._on_save)
        self.viewer.views[0].picked.connect(self._on_picked)
        self.viewer.views[1].picked.connect(self._on_picked)

        self.bone_cc_volume = None
        self.volume = None
        self.selected_labels = set()
        self.tooth_data = {}

    def set_patient(self, patient):
        self.patient = patient
        self.bone_cc_volume = None
        self.volume = None
        self.selected_labels = set()
        self.tooth_data = {}
        self.dirty = False
        for view in self.viewer.views:
            view.clear_actors()

        if not patient:
            return

        patient_dir = self.working_dir / patient
        tooth_path = patient_dir / TOOTH_FILENAME
        bone_path = patient_dir / BONE_FILENAME
        if not (tooth_path.is_file() and bone_path.is_file()):
            return

        self.volume = numpy.load(tooth_path)
        self.bone_cc_volume = numpy.load(bone_path)
        self.tooth_data = load_tooth_data(patient_dir)

        self._refresh_display()

    def finalize(self, recompute=True):
        if not self.dirty:
            return

        apply_manual_edit(self.working_dir / self.patient, self.bone_cc_volume, self.volume, recompute=recompute)
        self.dirty = False

    def _refresh_display(self, reset=True):
        self.selected_labels = set()

        render_settings = {}
        for label in range(2, int(self.volume.max()) + 1):
            opacity = 0 if self.tooth_data.get(label) == 'Noise' else 1
            render_settings[label] = {
                'name': f'label {label}',
                'color': tuple(c / 255 for c in GLASBEY_PALETTE[label]),
                'opacity': opacity,
            }
        self.viewer.views[0].update_volume(self.volume, include_labels=True, render_settings=render_settings, reset=reset)

        self.viewer.views[1].clear_actors()

    def _on_clear(self):
        if self.volume is None:
            return

        for label in self.selected_labels:
            self.viewer.views[1].actors[f'label {label}'].prop.opacity = 0

        self.selected_labels.clear()
        self.viewer.views[1].render()

    def _on_save(self):
        labels = sorted(self.selected_labels)
        if len(labels) < 2:
            return

        survivor = labels[0]
        if survivor not in self.tooth_data:
            for label in labels[1:]:
                if label in self.tooth_data:
                    self.tooth_data[survivor] = self.tooth_data[label]
                    break
        for label in labels[1:]:
            self.tooth_data.pop(label, None)

        self.volume = refine_merge(self.volume, [labels])
        self.dirty = True

        self._refresh_display(reset=False)

    def _on_picked(self, actor):
        label = actor._label

        preview_actor = self.viewer.views[1].actors.get(f'label {label}')

        if label in self.selected_labels:
            self.selected_labels.remove(label)
            preview_actor.prop.opacity = 0
        else:
            self.selected_labels.add(label)
            if preview_actor is None:
                self.viewer.views[1].add_label(self.volume, label, name=f'label {label}', color=tuple(c / 255 for c in Color.TOOTH))
            else:
                preview_actor.prop.opacity = 1

        self.viewer.views[1].render()
