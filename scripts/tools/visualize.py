from PySide6.QtWidgets import QApplication, QHBoxLayout

from scripts.tools.widgets import MainWindowUI, Mode, PatientSelector, VolumeViewer

class TopLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.patient_selector = PatientSelector()
        self.addWidget(self.patient_selector)
        self.addStretch()

    def get_widgets(self):
        return self.patient_selector

class BottomLayout(QHBoxLayout):
    def __init__(self, count, size=(640, 640)):
        super().__init__()
        self.volume_viewer = VolumeViewer(count, size)
        self.addWidget(self.volume_viewer)

    def get_widgets(self):
        return self.volume_viewer

class MainWindow(MainWindowUI):
    def __init__(self, data_manager, default_patient=None):
        super().__init__(TopLayout, (BottomLayout, {'count': len(data_manager.modes), 'size': (640, 640) if len(data_manager.modes) <= 2 else (512, 512)}))
        self.data_manager = data_manager

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector = self.layouts[0].get_widgets()
        self.volume_viewer = self.layouts[1].get_widgets()

        self.patient_selector.setup(data_manager.patients, self._on_patient_changed)

        self.volume_viewer.set_titles(*[
            Mode.get_title(mode)
            for mode in data_manager.modes
        ])

        if default_patient is not None:
            index = self.data_manager.patients.index(default_patient)
            self.patient_selector.setCurrentIndex(index)

    def _on_patient_changed(self, index):
        volumes = self.data_manager.load_data(index)

        reset = True
        for mode, box in zip(self.data_manager.modes, self.volume_viewer.boxes):
            volume = volumes[mode]
            box.view.update_volume(volume, mode, reset=reset)
            reset = False

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
    parser.add_argument('modes', nargs='+', choices=Mode.items())
    parser.add_argument('--patient', type=str)
    args = parser.parse_args()

    experiment_name = args.exp
    modes = args.modes
    default_patient = args.patient

    datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
    patients = [
        f'{dataset}/{patient}'
        for dataset, patient in iterate_datasets(experiment_name, datasets)
    ]

    if default_patient is not None and default_patient not in patients:
        raise ValueError(f'Patient {default_patient} not found')

    app = QApplication([])

    data_manager = DataManager(experiment_name, patients, modes)
    window = MainWindow(data_manager, default_patient)

    window.show()

    app.exec()
