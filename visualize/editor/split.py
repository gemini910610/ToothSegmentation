from pathlib import Path

import numpy
from PySide6.QtWidgets import QPushButton, QSpinBox

from constants.mode import Mode
from post_processing.utils import ROIVolume
from post_processing.watershed import split_k_component
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.selectors import IconLabelSelector, PatientSelector
from visualize.ui.views import VolumeViewer
from visualize.ui.windows import BaseWindow
from visualize.utils import DataHandler, JsonHandler


class SplitDataHandler(DataHandler):
    def set_data(self, patient, label, cluster):
        if patient not in self.data:
            self.data[patient] = {}
        self.data[patient][str(label)] = cluster

class TopLayout(HorizontalLayout):
    def _init_widgets(self):
        return [
            PatientSelector(),
            IconLabelSelector(),
            QSpinBox(suffix=' Cluster', minimum=2),
            QPushButton('Execute'),
            QPushButton('Save'),
            None
        ]

class BottomLayout(HorizontalLayout):
    def _init_widgets(self):
        return [
            VolumeViewer(3, (512, 512))
        ]

class MainWindow(BaseWindow):
    def __init__(self, data_manager, default_patient=None, default_label=None):
        super().__init__(TopLayout, BottomLayout)
        self.data_manager = data_manager
        self.default_label = default_label

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector, self.label_selector, self.cluster_input, self.execute_button, self.save_button = self.layouts[0].widgets
        self.volume_viewer, = self.layouts[1].widgets

        self.patient_selector.setup(data_manager.patients, self._on_patient_changed)

        self.volume_viewer.set_titles(Mode.get_title(Mode.TOOTH_CONNECTED_COMPONENT), Mode.get_title(Mode.WATERSHED), 'Instance')

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.execute_button.clicked.connect(self._on_execute)
        self.save_button.clicked.connect(self._on_save)
        self.volume_viewer.views[0].picked.connect(self._on_picked)

        self.volume = None
        self.label_volume = None

        if default_patient is not None:
            index = self.data_manager.patients.index(default_patient)
            self.patient_selector.setCurrentIndex(index)

    def _on_patient_changed(self, index):
        volumes = self.data_manager.load_data(index)
        cc_volume = volumes[Mode.TOOTH_CONNECTED_COMPONENT]
        watershed_volume = volumes[Mode.WATERSHED]
        self.volume = cc_volume

        self.volume_viewer.views[0].update_volume(cc_volume, include_labels=True)
        self.volume_viewer.views[1].update_volume(watershed_volume, include_labels=True)

        labels = numpy.unique(cc_volume)
        labels = labels[labels > 0]

        self.label_selector.update_items(labels)

        index = 0 if self.default_label is None or self.default_label not in labels else labels.tolist().index(self.default_label)
        self.label_selector.setCurrentIndex(index)
        self.default_label = None

    def _on_label_changed(self, index):
        if self.volume is None:
            return

        label = self.label_selector.currentData()

        self.label_volume = self.volume == label
        volume = (self.label_volume * label).astype(numpy.uint8)
        self.volume_viewer.views[2].update_volume(volume, include_labels=True, reset=False)

        self.cluster_input.setValue(self.cluster_input.minimum())

    def _on_execute(self):
        if self.label_volume is None:
            return

        cluster = self.cluster_input.value()

        roi = ROIVolume(self.label_volume)
        volume = split_k_component(roi.volume, h=1, k=cluster)
        volume[volume > 0] += 1
        roi.volume = volume
        volume = roi.restore()
        self.volume_viewer.views[2].update_volume(volume, include_labels=True, reset=False)

    def _on_save(self):
        patient = self.patient_selector.currentText()
        label = self.label_selector.currentData()
        cluster = self.cluster_input.value()

        if patient == '' or label is None:
            return

        path = Path('split') / f'{self.data_manager.experiment_name}.json'
        with JsonHandler(path, SplitDataHandler) as handler:
            handler.set_data(patient, label, cluster)

    def _on_picked(self, actor):
        label = actor._label
        index = self.label_selector.findData(label)
        self.label_selector.setCurrentIndex(index)

if __name__ == '__main__':
    from argparse import ArgumentParser

    from post_processing.utils import iterate_datasets
    from visualize.utils import DataManager, run_app

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('datasets', nargs='+', type=str)
    parser.add_argument('--patient', type=str)
    parser.add_argument('--label', type=int)
    args = parser.parse_args()

    experiment_name = args.exp
    datasets = args.datasets
    default_patient = args.patient
    default_label = args.label

    patients = [f'{dataset}/{patient}' for dataset, patient in iterate_datasets(experiment_name, datasets)]
    if default_patient is not None and default_patient not in patients:
        raise ValueError(f'patient {default_patient} not found')

    data_manager = DataManager(experiment_name, patients, [Mode.TOOTH_CONNECTED_COMPONENT, Mode.WATERSHED])
    run_app(MainWindow, data_manager, default_patient, default_label)
