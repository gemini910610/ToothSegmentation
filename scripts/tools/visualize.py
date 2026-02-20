import os

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel

from .widgets import Mode, VolumeLoader, VolumeViewer

class MainWindowUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.move(0, 0)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setCentralWidget(widget)

        top_layout = QHBoxLayout()
        self.patient_selector = QComboBox()
        self.tooth_count_label = QLabel('Tooth Count: -/-')
        top_layout.addWidget(self.patient_selector)
        top_layout.addWidget(self.tooth_count_label)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        self.volume_viewer = VolumeViewer()
        layout.addWidget(self.volume_viewer)

class MainWindow(MainWindowUI):
    def __init__(self, data_manager, left_mode, right_mode):
        super().__init__()
        self.data_manager = data_manager
        self.left_mode = left_mode
        self.right_mode = right_mode

        self.setWindowTitle(self.data_manager.experiment_name)

        self.patient_selector.addItems(self.data_manager.patients)
        self.patient_selector.setCurrentIndex(-1)
        self.patient_selector.currentIndexChanged.connect(self._load_patient)

        titles = {
            Mode.GROUND_TRUTH: 'Ground Truth',
            Mode.PREDICT: 'Predict',
            Mode.CONNECTED_COMPONENT: 'Connected Component',
            Mode.CLEANED: 'Cleaned',
            Mode.WATERSHED: 'Watershed',
            Mode.REFINE: 'Refine',
            Mode.POST_PROCESSING: 'Post Processing',
            Mode.REMOVED: 'Removed',
            Mode.RELABELED: 'Relabeled'
        }
        self.volume_viewer.views[0].setTitle(titles[left_mode])
        self.volume_viewer.views[1].setTitle(titles[right_mode])

    def _load_patient(self, index):
        patient = self.data_manager.patients[index]
        self.patient_selector.setEnabled(False)
        self.thread = VolumeLoader(self.data_manager, patient)
        self.thread.finished.connect(self._on_volume_loaded)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _on_volume_loaded(self, volumes):
        left_volume, left_count = volumes[self.left_mode]
        right_volume, right_count = volumes[self.right_mode]

        self.volume_viewer.views[0].view.update_volume(left_volume)
        self.volume_viewer.views[1].view.update_volume(right_volume)
        self.tooth_count_label.setText(f'Tooth Count: {left_count if left_count is not None else "-"}/{right_count if right_count is not None else "-"}')
        self.patient_selector.setEnabled(True)

if __name__ == '__main__':
    from argparse import ArgumentParser
    from src.config import load_config
    from .widgets import DataManager, get_patient_fold_mapping

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('--left', default=Mode.PREDICT, choices=Mode.items())
    parser.add_argument('--right', default=Mode.GROUND_TRUTH, choices=Mode.items())
    parser.add_argument('--cc-label', nargs='+', default=[1, 2], type=int)
    args = parser.parse_args()

    experiment_name = args.exp
    left_mode = args.left
    right_mode = args.right
    cc_label = args.cc_label
    if not os.path.exists(os.path.join('outputs', experiment_name)):
        raise FileNotFoundError(
            f'Output of experiment "{experiment_name}" not found.\033[0m\n'
            'Try using the following command to generating segmentation results:\n'
            f'\033[36mpython scripts/compare.py {experiment_name}\033[0m'
        )

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    patient_fold_map = get_patient_fold_mapping(config)

    app = QApplication([])

    data_manager = DataManager(experiment_name, patient_fold_map, [left_mode, right_mode], cc_label)
    window = MainWindow(data_manager, left_mode, right_mode)

    window.show()

    app.exec()
