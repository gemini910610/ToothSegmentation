import numpy
import os

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QSpinBox

from scripts.post_processing.watershed import split_k_component
from scripts.tools.widgets import DataHandler, IconLabelSelector, JsonHandler, MainWindowUI, Mode, PatientSelector, VolumeViewer

class SplitDataHandler(DataHandler):
    def set_data(self, patient, label, cluster):
        if patient not in self.data:
            self.data[patient] = {}
        self.data[patient][label] = cluster

class TopLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.patient_selector = PatientSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.label_selector = IconLabelSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.cluster_input = QSpinBox(suffix=' Cluster', minimum=2)
        self.execute_button = QPushButton('Execute')
        self.save_button = QPushButton('Save')
        for widget in [self.patient_selector, self.label_selector, self.cluster_input, self.execute_button, self.save_button]:
            self.addWidget(widget)
        self.addStretch()

    def get_widgets(self):
        return self.patient_selector, self.label_selector, self.cluster_input, self.execute_button, self.save_button

class BottomLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.volume_viewer = VolumeViewer(3, (512, 512))
        self.addWidget(self.volume_viewer)

    def get_widgets(self):
        return self.volume_viewer

class MainWindow(MainWindowUI):
    def __init__(self, data_manager, default_patient=None, default_label=None):
        super().__init__(TopLayout, BottomLayout)
        self.data_manager = data_manager
        self.default_label = default_label if default_patient is not None else None

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector, self.label_selector, self.cluster_input, self.execute_button, self.save_button = self.layouts[0].get_widgets()
        self.volume_viewer = self.layouts[1].get_widgets()

        self.patient_selector.setup(data_manager.patients, self._on_patient_changed)

        self.volume_viewer.set_titles(Mode.get_title(Mode.TOOTH_CONNECTED_COMPONENT), Mode.get_title(Mode.WATERSHED), 'Instance')

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.execute_button.clicked.connect(self._split_component)
        self.save_button.clicked.connect(self._on_save)
        self.volume_viewer.boxes[0].view.picked.connect(self._on_picking)

        self.volume = None
        self.label_volume = None

        if default_patient is not None:
            index = self.data_manager.patients.index(default_patient)
            self.patient_selector.setCurrentIndex(index)

    def _on_picking(self, actor):
        label = actor._label
        index = self.label_selector.findData(label)
        self.label_selector.setCurrentIndex(index)

    def _on_patient_changed(self, index):
        volumes = self.data_manager.load_data(index)
        cc_volume = volumes[Mode.TOOTH_CONNECTED_COMPONENT]
        watershed_volume = volumes[Mode.WATERSHED]
        self.volume = cc_volume

        self.volume_viewer.boxes[0].view.update_volume(cc_volume, Mode.TOOTH_CONNECTED_COMPONENT)
        self.volume_viewer.boxes[1].view.update_volume(watershed_volume, Mode.WATERSHED)

        labels = numpy.unique(cc_volume)
        labels = labels[labels > 0]

        self.label_selector.update_items(labels, labels.max(), -1)

        if self.default_label is not None and self.default_label in labels:
            index = labels.tolist().index(self.default_label)
            self.label_selector.setCurrentIndex(index)
        else:
            self._on_label_changed(0)
        self.default_label = None

    def _on_label_changed(self, index):
        if self.volume is None:
            return

        label = self.label_selector.current_label()

        self.label_volume = self.volume == label
        volume = self.label_volume * label
        volume = volume.astype(numpy.int32, copy=False)
        self.volume_viewer.boxes[2].view.update_volume(volume, Mode.TOOTH_CONNECTED_COMPONENT, reset=False)

        self.cluster_input.setValue(self.cluster_input.minimum())

    def _split_component(self):
        if self.label_volume is None:
            return

        cluster = self.cluster_input.value()

        volume = split_k_component(self.label_volume, h=1, k=cluster, crop=True)
        volume = volume.astype(numpy.int32, copy=False)
        self.volume_viewer.boxes[2].view.update_volume(volume, Mode.TOOTH_CONNECTED_COMPONENT, reset=False)

    def _on_save(self):
        patient = self.patient_selector.currentText()
        label = self.label_selector.current_label()
        cluster = self.cluster_input.value()

        if patient == '' or label is None:
            return

        path = os.path.join('split', f'{self.data_manager.experiment_name}.json')
        with JsonHandler(path, SplitDataHandler) as handler:
            handler.set_data(patient, str(label), cluster)

    def closeEvent(self, event):
        super().closeEvent(event)
        for box in self.volume_viewer.boxes:
            box.view.close()

if __name__ == '__main__':
    from argparse import ArgumentParser
    from PySide6.QtWidgets import QApplication

    from scripts.post_processing.iterators import iterate_datasets
    from scripts.tools.widgets import DataManager

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('--patient', type=str)
    parser.add_argument('--label', type=int)
    args = parser.parse_args()

    experiment_name = args.exp
    default_patient = args.patient
    default_label = args.label

    datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
    patients = [f'{dataset}/{patient}' for dataset, patient in iterate_datasets(experiment_name, datasets)]

    if default_patient is not None and default_patient not in patients:
        raise ValueError(f'Patient {default_patient} not found')

    app = QApplication([])

    data_manager = DataManager(experiment_name, patients, [Mode.TOOTH_CONNECTED_COMPONENT, Mode.WATERSHED])
    window = MainWindow(data_manager, default_patient, default_label)

    window.show()

    app.exec()
