from pathlib import Path

import numpy
from PySide6.QtWidgets import QPushButton, QSpinBox, QVBoxLayout

from constants.color import GLASBEY_PALETTE
from post_processing.refine_split import refine_split
from post_processing.utils import ROIVolume
from post_processing.watershed import split_k_component
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.selectors import IconLabelSelector
from visualize.ui.views import VolumeViewer
from visualize.viewer.pages.page import VolumeViewerPage
from visualize.viewer.pipeline import BONE_FILENAME, TOOTH_FILENAME, apply_manual_edit
from visualize.viewer.tooth_data import load_tooth_data


class SplitControls(HorizontalLayout):
    def _init_widgets(self):
        return [
            IconLabelSelector(),
            QSpinBox(suffix=' Components', minimum=2),
            QPushButton('Execute'),
            None,
            QPushButton('Save'),
        ]

class SplitPage(VolumeViewerPage):
    def __init__(self, working_dir=Path('working')):
        super().__init__(working_dir)

        self.controls = SplitControls()
        self.viewer = VolumeViewer(2, (512, 512))
        self.viewer.set_titles('Tooth', 'Instance')

        layout = QVBoxLayout(self)
        layout.addLayout(self.controls)
        layout.addWidget(self.viewer)

        self.label_selector, self.count_input, self.execute_button, self.save_button = self.controls.widgets

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.execute_button.clicked.connect(self._on_execute)
        self.save_button.clicked.connect(self._on_save)
        self.viewer.views[0].picked.connect(self._on_picked)

        self.bone_cc_volume = None
        self.volume = None
        self.label_volume = None
        self.tooth_data = {}

    def set_patient(self, patient):
        self.patient = patient
        self.bone_cc_volume = None
        self.volume = None
        self.label_volume = None
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

    def _refresh_display(self, reset=True, select_label=None):
        render_settings = {}
        for label in range(2, int(self.volume.max()) + 1):
            opacity = 0 if self.tooth_data.get(label) == 'Noise' else 1
            render_settings[label] = {
                'name': f'label {label}',
                'color': tuple(c / 255 for c in GLASBEY_PALETTE[label]),
                'opacity': opacity,
            }
        self.viewer.views[0].update_volume(self.volume, include_labels=True, render_settings=render_settings, reset=reset)

        labels = numpy.unique(self.volume)
        labels = numpy.array([label for label in labels if label > 0 and self.tooth_data.get(int(label)) != 'Noise'])

        self.label_selector.update_items(labels)
        index = 0 if select_label is None else self.label_selector.findData(select_label)
        self.label_selector.setCurrentIndex(index)

    def _on_label_changed(self, index):
        if self.volume is None:
            return

        label = self.label_selector.currentData()
        if label is None:
            return

        self.label_volume = self.volume == label
        volume = (self.label_volume * label).astype(numpy.uint8)
        self.viewer.views[1].update_volume(volume, include_labels=True, reset=False)

        self.count_input.setValue(self.count_input.minimum())

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
        self.viewer.views[1].update_volume(volume, include_labels=True, reset=False)

    def _on_save(self):
        label = self.label_selector.currentData()
        if label is None:
            return

        count = self.count_input.value()
        self.volume, new_labels = refine_split(self.volume, self.volume, {label: count})
        self.tooth_data.pop(label, None)
        self.dirty = True

        self._refresh_display(reset=False, select_label=new_labels[label][0])
