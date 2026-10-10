from pathlib import Path

import numpy
from PySide6.QtWidgets import QPushButton

from constants.label import Label
from constants.mode import Mode
from post_processing.auto_detect import detect_exclude_labels
from visualize.ui.buttons import RadioGroup
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.selectors import IconLabelSelector, PatientSelector
from visualize.ui.views import VolumeViewer
from visualize.ui.windows import BaseWindow
from visualize.utils import DataHandler, JsonHandler


class ExcludeDataHandler(DataHandler):
    def find(self, patient, label):
        if patient not in self.data:
            return None

        for exclude_type in self.data[patient]:
            if label in self.data[patient][exclude_type]:
                return exclude_type

        return None

    def set_data(self, patient, exclude_type, label):
        origin_exclude_type = self.find(patient, label)
        if origin_exclude_type is not None:
            if origin_exclude_type == exclude_type:
                return

            self.remove_data(patient, label)

        data = self.data
        if patient not in data:
            data[patient] = {}

        data = data[patient]
        if exclude_type not in data:
            data[exclude_type] = []

        data = data[exclude_type]
        if label not in data:
            data.append(label)

    def remove_data(self, patient, label):
        exclude_type = self.find(patient, label)
        if exclude_type is None:
            return

        self.data[patient][exclude_type].remove(label)

        if not self.data[patient][exclude_type]:
            del self.data[patient][exclude_type]

        if not self.data[patient]:
            del self.data[patient]

EXCLUDE_COLORS = {
    'Unerupted Tooth': (220, 70, 70),
    'Residual Root': (70, 220, 70),
    'Fake Tooth': (70, 70, 220),
    'Noise': (0, 0, 0),
}

class TopLayout(HorizontalLayout):
    def _init_widgets(self):
        return [
            PatientSelector(),
            IconLabelSelector(),
            None,
            RadioGroup(['Normal Tooth', 'Unerupted Tooth', 'Residual Root', 'Fake Tooth', 'Noise']),
            QPushButton('Save')
        ]

class BottomLayout(HorizontalLayout):
    def _init_widgets(self):
        return [
            VolumeViewer()
        ]

class MainWindow(BaseWindow):
    def __init__(self, data_manager):
        super().__init__(TopLayout, BottomLayout)
        self.data_manager = data_manager

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector, self.label_selector, self.exclude_selector, self.save_button = self.layouts[0].widgets
        self.volume_viewer, = self.layouts[1].widgets

        self.patient_selector.setup(data_manager.patients, self._on_patient_changed)
        self.exclude_selector.select_first()

        self.volume_viewer.set_titles(Mode.get_title(Mode.UNTRUNCATED), 'Instance')

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.save_button.clicked.connect(self._on_save)
        self.volume_viewer.views[0].picked.connect(self._on_picked)

        self.volume = None

    def _on_patient_changed(self, index):
        volumes = self.data_manager.load_data(index)
        tooth_volume = volumes[Mode.UNTRUNCATED]
        bone_volume = volumes[Mode.BONE_CONNECTED_COMPONENT]
        bone_mask = bone_volume > 0
        self.volume = tooth_volume

        volume = numpy.where(tooth_volume >= Label.TRUNCATED, 0, tooth_volume)
        volume = volume + bone_mask
        self.volume_viewer.views[0].update_volume(volume, include_labels=True)

        labels = numpy.unique(self.volume)
        labels = labels[(labels > 0) & (labels < Label.TRUNCATED)]

        patient = self.patient_selector.currentText()
        with JsonHandler(Path('edit') / self.data_manager.experiment_name / 'exclude.json', ExcludeDataHandler) as handler:
            unmarked_labels = [int(label) for label in labels if handler.find(patient, int(label)) is None]
            exclude_labels = detect_exclude_labels(tooth_volume, bone_mask, unmarked_labels)
            for exclude_type, detected_labels in exclude_labels.items():
                for label in detected_labels:
                    handler.set_data(patient, exclude_type, label)

            for label in labels:
                color = EXCLUDE_COLORS.get(handler.find(patient, int(label)))
                if color is None:
                    continue

                actor = self.volume_viewer.views[0].actors.get(f'label {int(label)}')
                if actor is not None:
                    actor.prop.color = tuple(c / 255 for c in color)

        self.volume_viewer.views[0].render()

        self.label_selector.update_items(labels)

        self.label_selector.setCurrentIndex(0)

    def _on_label_changed(self, index):
        if self.volume is None:
            return

        label = self.label_selector.currentData()

        volume = self.volume == label
        volume = volume * Label.TOOTH
        self.volume_viewer.views[1].update_volume(volume, reset=False)

        patient = self.patient_selector.currentText()
        with JsonHandler(Path('edit') / self.data_manager.experiment_name / 'exclude.json', ExcludeDataHandler) as handler:
            exclude_type = handler.find(patient, label)

        if exclude_type is None:
            self.exclude_selector.select_first()
        else:
            self.exclude_selector.select_title(exclude_type)

    def _on_save(self):
        if self.volume is None:
            return

        patient = self.patient_selector.currentText()
        label = self.label_selector.currentData()
        exclude_type = self.exclude_selector.checked_title()

        with JsonHandler(Path('edit') / self.data_manager.experiment_name / 'exclude.json', ExcludeDataHandler) as handler:
            if exclude_type == 'Normal Tooth':
                handler.remove_data(patient, label)
            else: # Unerupted Tooth, Residual Root, Fake Tooth, Noise
                handler.set_data(patient, exclude_type, label)

    def _on_picked(self, actor):
        label = actor._label
        index = self.label_selector.findData(label)
        self.label_selector.setCurrentIndex(index)

if __name__ == '__main__':
    import sys

    from post_processing.utils import iterate_datasets
    from tui.apps import BaseApp
    from visualize.utils import DataManager, run_app

    args = BaseApp(required_mode=Mode.UNTRUNCATED).run()
    if args is None:
        sys.exit()

    segmentation_experiment_name = args['exp']
    datasets = args['datasets']

    patients = [f'{dataset}/{patient}' for dataset, patient in iterate_datasets(segmentation_experiment_name, datasets)]
    data_manager = DataManager(segmentation_experiment_name, patients, [Mode.UNTRUNCATED, Mode.BONE_CONNECTED_COMPONENT])
    run_app(MainWindow, data_manager)
