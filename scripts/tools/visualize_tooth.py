import numpy

from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QSpinBox, QPushButton
from PySide6.QtGui import QColor, QPixmap, QIcon
from scripts.tools.visualize import VolumeViewer, VolumeLoader, Mode, VolumeColorizer
from scripts.post_processing.watershed import split_k_component

class MainWindowUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.move(0, 0)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setCentralWidget(widget)

        top_layout = QHBoxLayout()
        self.patient_selector = QComboBox()
        self.patient_selector.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.label_selector = QComboBox()
        self.label_selector.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.cluster_input = QSpinBox(suffix=' Cluster', minimum=1, maximum=5)
        self.execute_button = QPushButton('Execute')
        for widget in [self.patient_selector, self.label_selector, self.cluster_input, self.execute_button]:
            top_layout.addWidget(widget)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        self.volume_viewer = VolumeViewer(3, (512, 512))
        layout.addWidget(self.volume_viewer)

class MainWindow(MainWindowUI):
    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager

        self.setWindowTitle(self.data_manager.experiment_name)

        self.patient_selector.addItems(self.data_manager.patients)
        self.patient_selector.setCurrentIndex(-1)
        self.patient_selector.currentIndexChanged.connect(self._load_patient)

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.execute_button.clicked.connect(self._split_component)

        for view, title in zip(self.volume_viewer.views, ['Connected Component', 'Watershed', 'Instance']):
            view.setTitle(title)

        self.volume = None

    def _load_patient(self, index):
        self.patient_selector.setEnabled(False)
        self.label_selector.setEnabled(False)
        self.cluster_input.setEnabled(False)
        self.execute_button.setEnabled(False)

        patient = self.data_manager.patients[index]
        self.volume, _ = self.data_manager.load_data(patient)

        self.thread = VolumeLoader(self.data_manager, patient, Mode.CONNECTED_COMPONENT, Mode.WATERSHED)
        self.thread.finished.connect(self._on_volume_loaded)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _on_volume_loaded(self, left_volume, right_volume, tooth_count):
        self.volume_viewer.views[0].view.update_volume(left_volume)
        self.volume_viewer.views[1].view.update_volume(right_volume)

        self.label_selector.blockSignals(True)
        self.label_selector.clear()
        palette = VolumeColorizer.glasbey_palette(tooth_count)
        size = self.label_selector.font().pointSize()
        for index in range(tooth_count):
            r, g, b, _ = palette[index]
            color = QColor(r, g, b)

            pixmap = QPixmap(size, size)
            pixmap.fill(color)
            icon = QIcon(pixmap)

            self.label_selector.addItem(icon, f'Label {index + 1}')
        self.label_selector.setCurrentIndex(0)
        self.label_selector.blockSignals(False)
        self._on_label_changed(0, reset=True)

        self.cluster_input.setValue(self.cluster_input.minimum())

        self.patient_selector.setEnabled(True)
        self.label_selector.setEnabled(True)
        self.cluster_input.setEnabled(True)
        self.execute_button.setEnabled(True)

    def _on_label_changed(self, index, reset=False):
        if self.volume is None:
            return

        label = index + 1

        volume = self.volume == label
        volume = volume * label
        volume = volume.astype(numpy.int32, copy=False)
        volume, _ = VolumeColorizer.color_components(volume)
        self.volume_viewer.views[2].view.update_volume(volume, reset)

    def _split_component(self):
        if self.volume is None:
            return

        self.patient_selector.setEnabled(False)
        self.label_selector.setEnabled(False)
        self.cluster_input.setEnabled(False)
        self.execute_button.setEnabled(False)

        label = self.label_selector.currentIndex() + 1
        cluster = self.cluster_input.value()

        volume = self.volume == label
        volume = split_k_component(volume, h=1, k=cluster, crop=True)
        volume = volume.astype(numpy.int32, copy=False)
        volume, _ = VolumeColorizer.color_components(volume)
        self.volume_viewer.views[2].view.update_volume(volume, reset=False)

        self.patient_selector.setEnabled(True)
        self.label_selector.setEnabled(True)
        self.cluster_input.setEnabled(True)
        self.execute_button.setEnabled(True)

if __name__ == '__main__':
    import os

    from argparse import ArgumentParser
    from PySide6.QtWidgets import QApplication
    from src.config import load_config
    from scripts.tools.visualize import get_patient_fold_mapping, DataManager

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    patient_fold_map = get_patient_fold_mapping(config)

    app = QApplication([])

    data_manager = DataManager(experiment_name, patient_fold_map, [Mode.CONNECTED_COMPONENT, Mode.WATERSHED], [1])
    window = MainWindow(data_manager)

    window.show()

    app.exec()
