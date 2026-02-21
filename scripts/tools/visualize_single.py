import numpy

from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox
from .widgets import VolumeViewer, Mode, VolumeColorizer, IconLabelSelector, VolumeLoader, PatientSelector

class MainWindowUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.move(0, 0)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setCentralWidget(widget)

        top_layout = QHBoxLayout()
        self.patient_selector = PatientSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.label_selector = IconLabelSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        for widget in [self.patient_selector, self.label_selector]:
            top_layout.addWidget(widget)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        self.volume_viewer = VolumeViewer(2)
        layout.addWidget(self.volume_viewer)

class MainWindow(MainWindowUI):
    def __init__(self, data_manager):
        super().__init__()
        self.loader = VolumeLoader(data_manager, self._handle_volumes, keep_origin=True)

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector.setup(data_manager.patients, self.loader.load_patient)

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)

        self.volume_viewer.set_titles(Mode.get_title(Mode.POST_PROCESSING), 'Instance')

        self.loader.setup(self.patient_selector, self.label_selector)

        self.volume = None

    def _handle_volumes(self, volumes):
        self.volume = volumes['origin'][Mode.POST_PROCESSING]
        volume, tooth_count = volumes[Mode.POST_PROCESSING]
        self.volume_viewer.views[0].view.update_volume(volume)

        labels = numpy.unique(self.volume)
        labels = labels[labels > 1]

        self.label_selector.update_items(labels, tooth_count, -2)
        self._on_label_changed(0, reset=True)

    def _on_label_changed(self, index, reset=False):
        if self.volume is None:
            return

        label = self.label_selector.current_label()

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
    from .widgets import get_patient_fold_mapping, DataManager

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
