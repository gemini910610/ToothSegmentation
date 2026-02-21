import os

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel

from .widgets import Mode, VolumeViewer, VolumeLoader, PatientSelector

class MainWindowUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.move(0, 0)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setCentralWidget(widget)

        top_layout = QHBoxLayout()
        self.patient_selector = PatientSelector()
        self.tooth_count_label = QLabel('Tooth Count: -/-')
        for widget in [self.patient_selector, self.tooth_count_label]:
            top_layout.addWidget(widget)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        self.volume_viewer = VolumeViewer()
        layout.addWidget(self.volume_viewer)

class MainWindow(MainWindowUI):
    def __init__(self, data_manager):
        super().__init__()
        self.loader = VolumeLoader(data_manager, self._handle_volumes)

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector.setup(data_manager.patients, self.loader.load_patient)

        for view, title in zip(self.volume_viewer.views, [Mode.get_title(data_manager.modes[0]), Mode.get_title(data_manager.modes[1])]):
            view.setTitle(title)

        self.loader.setup(self.patient_selector)

    def _handle_volumes(self, volumes):
        left_volume, left_count = volumes[self.loader.data_manager.modes[0]]
        right_volume, right_count = volumes[self.loader.data_manager.modes[1]]

        self.volume_viewer.views[0].view.update_volume(left_volume)
        self.volume_viewer.views[1].view.update_volume(right_volume)
        self.tooth_count_label.setText(f'Tooth Count: {left_count if left_count is not None else "-"}/{right_count if right_count is not None else "-"}')

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
    window = MainWindow(data_manager)

    window.show()

    app.exec()
