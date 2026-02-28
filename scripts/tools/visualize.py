import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import QApplication, QHBoxLayout
from .widgets import Mode, VolumeViewer, VolumeLoader, PatientSelector, MainWindowUI, Color, Slider

class TopLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.patient_selector = PatientSelector()
        self.addWidget(self.patient_selector)
        self.slice_slider = Slider()
        self.addWidget(self.slice_slider)
    def get_widgets(self):
        return self.patient_selector, self.slice_slider

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

        self.patient_selector, self.slice_slider = self.top_layout.get_widgets()
        self.volume_viewer = self.bottom_layout.get_widgets()

        self.patient_selector.setup(data_manager.patients, self.loader.load_patient)

        self.volume_viewer.set_titles(*[
            Mode.get_title(mode)
            for mode in data_manager.modes
        ])

        self.loader.setup(self.patient_selector, self.slice_slider)

        self.slice_slider.valueChanged.connect(self._on_slice_changed)

        shortcut_enter = QShortcut(Qt.Key.Key_Return, self)
        shortcut_enter.activated.connect(lambda: self._on_slice_changed(self.slice_slider.get_value()))

        self.volumes = None

    def _handle_volumes(self, volumes):
        self.volumes = volumes

        self._on_slice_changed(1, reset=True)

        self.slice_slider.set_maximum(volumes[self.loader.data_manager.modes[0]][0].shape[2])
        self.slice_slider.set_value(1)

    def _on_slice_changed(self, value, reset=False):
        if self.volumes is None:
            return

        z = value - 1

        for mode, view in zip(self.loader.data_manager.modes, self.volume_viewer.views):
            volume, _ = self.volumes[mode]
            volume = volume.copy()

            z_slice = volume[:, :, z]
            tooth_area = ~(z_slice == Color.BONE).all(-1) & ~(z_slice == (0, 0, 0, 0)).all(-1)
            bone_area = (z_slice == Color.BONE).all(-1)
            z_slice[tooth_area] = (0, 0, 0, 255)
            z_slice[bone_area] = (255, 0, 0, 75)

            view.view.update_volume(volume, reset=reset)

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
