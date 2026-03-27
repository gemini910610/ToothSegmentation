import numpy

from PySide6.QtWidgets import QHBoxLayout, QComboBox, QSpinBox, QPushButton
from .widgets import VolumeViewer, Mode, VolumeColorizer, IconLabelSelector, VolumeLoader, PatientSelector, MainWindowUI
from scripts.post_processing.watershed import split_k_component

class TopLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.patient_selector = PatientSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.label_selector = IconLabelSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.cluster_input = QSpinBox(suffix=' Cluster', minimum=1, maximum=6)
        self.execute_button = QPushButton('Execute')
        for widget in [self.patient_selector, self.label_selector, self.cluster_input, self.execute_button]:
            self.addWidget(widget)
        self.addStretch()
    def get_widgets(self):
        return self.patient_selector, self.label_selector, self.cluster_input, self.execute_button

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

        self.patient_selector, self.label_selector, self.cluster_input, self.execute_button = self.top_layout.get_widgets()
        self.volume_viewer = self.bottom_layout.get_widgets()

        self.patient_selector.setup(data_manager.patients, self.loader.load_patient)

        self.volume_viewer.set_titles(Mode.get_title(Mode.TOOTH_CONNECTED_COMPONENT), Mode.get_title(Mode.WATERSHED), 'Instance')

        self.loader.setup(self.patient_selector, self.label_selector, self.cluster_input, self.execute_button)

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.execute_button.clicked.connect(self._split_component)

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

if __name__ == '__main__':
    import os

    from argparse import ArgumentParser
    from PySide6.QtWidgets import QApplication
    from src.config import load_config
    from .widgets import get_patient_fold_mapping, DataManager

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    patient_fold_map = get_patient_fold_mapping(config)

    app = QApplication([])

    data_manager = DataManager(experiment_name, patient_fold_map, [Mode.TOOTH_CONNECTED_COMPONENT, Mode.WATERSHED])
    window = MainWindow(data_manager)

    window.show()

    app.exec()
