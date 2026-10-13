from pathlib import Path

import cc3d
import numpy
from PySide6.QtWidgets import QPushButton, QVBoxLayout

from constants.color import GLASBEY_PALETTE
from constants.label import Label
from post_processing.relabel import compute_tooth_positions
from post_processing.tooth_data import (
    is_excluded,
    load_tooth_data,
    prune_stale_ids,
    save_tooth_data,
    set_entry,
)
from post_processing.utils import next_available_label
from visualize.ui.buttons import RadioGroup
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.selectors import IconLabelSelector
from visualize.ui.views import VolumeViewer
from visualize.viewer.pages.page import VolumeViewerPage
from visualize.viewer.pipeline import BONE_FILENAME, PREDICT_FILENAME, TOOTH_FILENAME

NOISE_THRESHOLD = 1000
NOISE_OPACITY = 0.2

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
            None,
            QPushButton('Save'),
        ]

class ExcludePage(VolumeViewerPage):
    def __init__(self, working_dir=Path('working')):
        super().__init__(working_dir)

        self.controls = ExcludeControls()
        self.viewer = VolumeViewer(2, (512, 512))
        self.viewer.set_titles('Tooth', 'Instance')

        layout = QVBoxLayout(self)
        layout.addLayout(self.controls)
        layout.addWidget(self.viewer)

        self.label_selector, self.exclude_selector, self.save_button = self.controls.widgets

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.save_button.clicked.connect(self._on_save)
        self.viewer.views[0].picked.connect(self._on_picked)

        self.bone_cc_volume = None
        self.volume = None
        self.tooth_data = {}

    def set_patient(self, patient):
        self.patient = patient
        self.bone_cc_volume = None
        self.volume = None
        self.tooth_data = {}
        self.dirty = False
        for view in self.viewer.views:
            view.clear_actors()

        if not patient:
            return

        patient_dir = self.working_dir / patient
        tooth_path = patient_dir / TOOTH_FILENAME
        bone_path = patient_dir / BONE_FILENAME
        predict_path = patient_dir / PREDICT_FILENAME
        if not (tooth_path.is_file() and bone_path.is_file() and predict_path.is_file()):
            return

        tooth_volume = numpy.load(tooth_path)
        bone_cc_volume = numpy.load(bone_path)
        predict_volume = numpy.load(predict_path)
        tooth_data = load_tooth_data(patient_dir)

        excluded_mask = (predict_volume == Label.TOOTH) & (tooth_volume == 0)
        if excluded_mask.any():
            components = cc3d.connected_components(excluded_mask, connectivity=6, binary_image=True)
            components = cc3d.dust(components, NOISE_THRESHOLD, connectivity=6, precomputed_ccl=True)

            noise_mask = components > 0
            if noise_mask.any():
                noise_source_labels = numpy.unique(components[noise_mask])

                used = {int(value) for value in numpy.unique(tooth_volume)}
                new_labels = []
                for _ in noise_source_labels:
                    new_label = next_available_label(used, 2)
                    used.add(new_label)
                    new_labels.append(new_label)

                lookup_table = numpy.zeros(components.max() + 1, dtype=numpy.uint8)
                lookup_table[noise_source_labels] = new_labels

                tooth_volume = tooth_volume.copy()
                tooth_volume[noise_mask] = lookup_table[components[noise_mask]]

                for new_label in new_labels:
                    tooth_data[new_label] = 'Filtered'

                numpy.save(tooth_path, tooth_volume)
                save_tooth_data(patient_dir, tooth_data)

        self.volume = tooth_volume
        self.bone_cc_volume = bone_cc_volume
        self.tooth_data = tooth_data

        self._refresh_display()

    def finalize(self, recompute=True):
        if not self.dirty:
            return

        if recompute:
            excluded_ids = {label for label, value in self.tooth_data.items() if is_excluded(value)}
            positions = compute_tooth_positions(self.volume, self.bone_cc_volume, excluded_ids=excluded_ids)
            self.tooth_data.update(positions)
            prune_stale_ids(self.tooth_data, {int(label) for label in numpy.unique(self.volume)})

        save_tooth_data(self.working_dir / self.patient, self.tooth_data)
        self.dirty = False

    def _display_exclude_type(self, label):
        exclude_type = self.tooth_data.get(label, 'Normal Tooth')
        return None if exclude_type == 'Filtered' else exclude_type

    def _refresh_display(self):
        self.viewer.views[0].update_volume(self.volume, include_labels=True)

        labels = numpy.unique(self.volume)
        labels = labels[labels > 0]

        for label in labels:
            exclude_type = self._display_exclude_type(int(label))
            color = EXCLUDE_COLORS.get(exclude_type)

            actor = self.viewer.views[0].actors.get(f'label {int(label)}')
            if actor is None:
                continue

            if color is not None:
                actor.prop.color = tuple(c / 255 for c in color)
            actor.prop.opacity = NOISE_OPACITY if exclude_type == 'Noise' else 1

        self.viewer.views[0].render()

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

        self.exclude_selector.select_title(self._display_exclude_type(label))

    def _on_save(self):
        if self.volume is None:
            return

        label = self.label_selector.currentData()
        if label is None:
            return

        exclude_type = self.exclude_selector.checked_title()
        if exclude_type is None:
            return

        set_entry(self.tooth_data, label, exclude_type)
        self.dirty = True

        actor = self.viewer.views[0].actors.get(f'label {label}')
        if actor is not None:
            color = EXCLUDE_COLORS.get(exclude_type)
            actor.prop.color = tuple(c / 255 for c in color) if color is not None else tuple(c / 255 for c in GLASBEY_PALETTE[label])
            actor.prop.opacity = NOISE_OPACITY if exclude_type == 'Noise' else 1
            self.viewer.views[0].render()
