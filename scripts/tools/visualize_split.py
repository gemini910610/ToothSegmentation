import numpy
import os

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QSpinBox

from scripts.post_processing.watershed import split_k_component
from scripts.tools.widgets import DataHandler, IconLabelSelector, JsonHandler, MainWindowUI, Mode, PatientSelector, VolumeColorizer, VolumeLoader, VolumeViewer

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
        self.cluster_input = QSpinBox(suffix=' Cluster', minimum=1)
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
    def __init__(self, data_manager):
        super().__init__(TopLayout, BottomLayout)
        self.loader = VolumeLoader(data_manager, self._handle_volumes, keep_origin=True)

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector, self.label_selector, self.cluster_input, self.execute_button, self.save_button = self.layouts[0].get_widgets()
        self.volume_viewer = self.layouts[1].get_widgets()

        self.patient_selector.setup(data_manager.patients, self.loader.load_patient)

        self.volume_viewer.set_titles(Mode.get_title(Mode.TOOTH_CONNECTED_COMPONENT), Mode.get_title(Mode.WATERSHED), 'Instance')

        self.loader.setup(self.patient_selector, self.label_selector, self.cluster_input, self.execute_button, self.save_button)

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.execute_button.clicked.connect(self._split_component)
        self.save_button.clicked.connect(self._on_save)

        self.volume = None

    def _handle_volumes(self, volumes):
        left_volume, tooth_count = volumes[Mode.TOOTH_CONNECTED_COMPONENT]
        right_volume, _ = volumes[Mode.WATERSHED]
        self.volume = volumes['origin'][Mode.TOOTH_CONNECTED_COMPONENT]

        self.volume_viewer.views[0].view.update_volume(left_volume)
        self.volume_viewer.views[1].view.update_volume(right_volume)

        labels = numpy.unique(self.volume)
        labels = labels[labels > 0]

        self.label_selector.update_items(labels, tooth_count, -1)
        self._on_label_changed(0, reset=True)

    def _on_label_changed(self, index, reset=False):
        if self.volume is None:
            return

        label = self.label_selector.current_label()

        volume = self.volume == label
        volume = volume * label
        volume = volume.astype(numpy.int32, copy=False)
        volume, _ = VolumeColorizer.color_components(volume)
        self.volume_viewer.views[2].view.update_volume(volume, reset)

        self.cluster_input.setValue(self.cluster_input.minimum())

    def _split_component(self):
        if self.volume is None:
            return

        label = self.label_selector.current_label()
        cluster = self.cluster_input.value()

        volume = self.volume == label
        volume = split_k_component(volume, h=1, k=cluster, crop=True)
        volume = volume.astype(numpy.int32, copy=False)
        volume, _ = VolumeColorizer.color_components(volume)
        self.volume_viewer.views[2].view.update_volume(volume, reset=False)

    def _on_save(self):
        if self.volume is None:
            return

        patient = self.patient_selector.currentText()
        label = self.label_selector.current_label()
        cluster = self.cluster_input.value()

        path = os.path.join('split', f'{self.loader.data_manager.experiment_name}.json')
        with JsonHandler(path, SplitDataHandler) as handler:
            handler.set_data(patient, str(label), cluster)

if __name__ == '__main__':
    from argparse import ArgumentParser
    from PySide6.QtWidgets import QApplication

    from scripts.post_processing.iterators import iterate_datasets
    from scripts.tools.widgets import DataManager

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
    patients = [f'{dataset}/{patient}' for dataset, patient in iterate_datasets(experiment_name, datasets)]

    app = QApplication([])

    data_manager = DataManager(experiment_name, patients, [Mode.TOOTH_CONNECTED_COMPONENT, Mode.WATERSHED])
    window = MainWindow(data_manager)

    window.show()

    app.exec()
