import os

from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel
from .widgets import Mode, VolumeViewer, VolumeLoader, PatientSelector, MainWindowUI

class TopLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.patient_selector = PatientSelector()
        self.tooth_count_label = QLabel('Tooth Count: -')
        for widget in [self.patient_selector, self.tooth_count_label]:
            self.addWidget(widget)
        self.addStretch()
    def get_widgets(self):
        return self.patient_selector, self.tooth_count_label

class BottomLayout(QHBoxLayout):
    def __init__(self, count, size=(640, 640)):
        super().__init__()
        self.volume_viewer = VolumeViewer(count, size)
        self.addWidget(self.volume_viewer)
    def get_widgets(self):
        return self.volume_viewer

class MainWindow(MainWindowUI):
    def __init__(self, data_manager):
        super().__init__(TopLayout, BottomLayout, bottom_kwargs={'count': len(data_manager.modes), 'size': (640, 640) if len(data_manager.modes) <= 2 else (512, 512)})
        self.loader = VolumeLoader(data_manager, self._handle_volumes)

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector, self.tooth_count_label = self.top_layout.get_widgets()
        self.volume_viewer = self.bottom_layout.get_widgets()

        self.patient_selector.setup(data_manager.patients, self.loader.load_patient)

        self.volume_viewer.set_titles(*[
            Mode.get_title(mode)
            for mode in data_manager.modes
        ])

        self.loader.setup(self.patient_selector)

    def _handle_volumes(self, volumes):
        counts = []
        for mode, view in zip(self.loader.data_manager.modes, self.volume_viewer.views):
            volume, count = volumes[mode]
            view.view.update_volume(volume)
            counts.append(str(count) if count is not None else '-')

        self.tooth_count_label.setText(f'Tooth Count: {"/".join(counts)}')

if __name__ == '__main__':
    from argparse import ArgumentParser
    from src.config import load_config
    from .widgets import DataManager, get_patient_fold_mapping

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('modes', nargs='+', choices=Mode.items())
    args = parser.parse_args()

    experiment_name = args.exp
    modes = args.modes

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    patient_fold_map = get_patient_fold_mapping(config)

    app = QApplication([])

    data_manager = DataManager(experiment_name, patient_fold_map, modes)
    window = MainWindow(data_manager)

    window.show()

    app.exec()
