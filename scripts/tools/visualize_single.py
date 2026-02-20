import numpy

from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox
from PySide6.QtGui import QColor, QPixmap, QIcon
from scripts.tools.visualize import VolumeViewer, Mode, VolumeColorizer

class MainWindowUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.move(0, 0)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setCentralWidget(widget)

        top_layout = QHBoxLayout()
        self.patient_selector = QComboBox(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.label_selector = QComboBox(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        for widget in [self.patient_selector, self.label_selector]:
            top_layout.addWidget(widget)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        self.volume_viewer = VolumeViewer(2)
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

        for view, title in zip(self.volume_viewer.views, ['Post Processing', 'Instance']):
            view.setTitle(title)

        self.volume = None

    def _load_patient(self, index):
        self.patient_selector.setEnabled(False)
        self.label_selector.setEnabled(False)

        patient = self.data_manager.patients[index]
        self.volume = self.data_manager.load_data(patient)[0]
        self._on_volume_loaded()

    def _on_volume_loaded(self):
        volume, _ = VolumeColorizer.color_components(self.volume, display_bone=True)
        self.volume_viewer.views[0].view.update_volume(volume)

        self.label_selector.blockSignals(True)
        self.label_selector.clear()
        tooth_count = self.volume.max() - 1
        palette = VolumeColorizer.glasbey_palette(tooth_count)
        size = self.label_selector.font().pointSize()
        labels = numpy.unique(self.volume)
        labels = labels[labels > 1]
        for label in labels:
            index = label - 2
            r, g, b, _ = palette[index]
            color = QColor(r, g, b)

            pixmap = QPixmap(size, size)
            pixmap.fill(color)
            icon = QIcon(pixmap)

            self.label_selector.addItem(icon, f'Label {label}')
        self.label_selector.setCurrentIndex(0)
        self.label_selector.blockSignals(False)
        self._on_label_changed(0, reset=True)

        self.patient_selector.setEnabled(True)
        self.label_selector.setEnabled(True)

    def _on_label_changed(self, index, reset=False):
        if self.volume is None:
            return

        label = int(self.label_selector.currentText()[6:])

        volume = self.volume == label
        volume = volume * (label - 1)
        volume = volume.astype(numpy.int32, copy=False)
        volume, _ = VolumeColorizer.color_components(volume)
        self.volume_viewer.views[1].view.update_volume(volume, reset)

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

    data_manager = DataManager(experiment_name, patient_fold_map, [Mode.POST_PROCESSING], [1])
    window = MainWindow(data_manager)

    window.show()

    app.exec()
