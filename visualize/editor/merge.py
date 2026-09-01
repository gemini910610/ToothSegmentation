from pathlib import Path

import numpy
from PySide6.QtWidgets import QPushButton

from constants.label import Label
from constants.mode import Mode
from visualize.ui.checkboxes import IconCheckboxTable
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.selectors import PatientSelector
from visualize.ui.views import VolumeViewer
from visualize.ui.windows import BaseWindow
from visualize.utils import DataHandler, JsonHandler


class MergeDataHandler(DataHandler):
    def find(self, patient, labels):
        if patient not in self.data:
            return None

        exists = set().union(*self.data[patient])
        labels = set(labels)
        duplicated = exists & labels
        if not duplicated:
            return None

        return sorted(duplicated)

    def set_data(self, patient, labels):
        duplicated = self.find(patient, labels)
        if duplicated is not None:
            raise ValueError(f'label {", ".join(str(label) for label in duplicated)} already exists')

        if patient not in self.data:
            self.data[patient] = []
        self.data[patient].append(labels)

class TopLayout(HorizontalLayout):
    def _init_widgets(self):
        return [
            PatientSelector(),
            None,
            QPushButton('Preview'),
            QPushButton('Clear'),
            QPushButton('Save')
        ]

class BottomLayout(HorizontalLayout):
    def _init_widgets(self):
        return [
            VolumeViewer(size=(512, 512)),
            IconCheckboxTable()
        ]

class MainWindow(BaseWindow):
    def __init__(self, data_manager):
        super().__init__(TopLayout, BottomLayout)
        self.data_manager = data_manager

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector, self.preview_button, self.clear_button, self.save_button = self.layouts[0].widgets
        self.volume_viewer, self.label_table = self.layouts[1].widgets

        self.patient_selector.setup(data_manager.patients, self._on_patient_changed)

        self.volume_viewer.set_titles(Mode.get_title(Mode.REFINE_SPLIT), 'Preview')

        self.preview_button.clicked.connect(self._on_preview)
        self.clear_button.clicked.connect(self._on_clear)
        self.save_button.clicked.connect(self._on_save)
        self.volume_viewer.views[0].picked.connect(self._on_picked)

        self.volume = None

    def _on_patient_changed(self, index):
        self.volume = self.data_manager.load_data(index)[Mode.REFINE_SPLIT]

        self.volume_viewer.views[0].update_volume(self.volume, include_labels=True)

        labels = numpy.unique(self.volume)
        labels = labels[labels > 0]

        self.label_table.update_items(labels)

        self._on_preview()

    def _on_preview(self):
        if self.volume is None:
            return

        labels = self.label_table.selected_labels()
        volume = numpy.where(numpy.isin(self.volume, labels), Label.TOOTH, Label.BACKGROUND)
        self.volume_viewer.views[1].update_volume(volume, reset=False)

    def _on_clear(self):
        if self.volume is None:
            return

        self.label_table.clear_selected()
        self._on_preview()

    def _on_save(self):
        if self.volume is None:
            return

        patient = self.patient_selector.currentText()
        labels = self.label_table.selected_labels()
        if len(labels) < 2:
            return

        with JsonHandler(Path('edit') / self.data_manager.experiment_name / 'merge.json', MergeDataHandler) as handler:
            handler.set_data(patient, labels)

    def _on_picked(self, actor):
        label = actor._label
        checkbox = self.label_table.checkboxes[label]
        checkbox.setChecked(True)

if __name__ == '__main__':
    import sys

    from post_processing.utils import iterate_datasets
    from tui.apps import BaseApp
    from visualize.utils import DataManager, run_app

    args = BaseApp().run()
    if args is None:
        sys.exit()

    segmentation_experiment_name = args['exp']
    datasets = args['datasets']

    patients = [f'{dataset}/{patient}' for dataset, patient in iterate_datasets(segmentation_experiment_name, datasets)]
    data_manager = DataManager(segmentation_experiment_name, patients, [Mode.REFINE_SPLIT])
    run_app(MainWindow, data_manager)
