import numpy

from PySide6.QtWidgets import QHBoxLayout, QComboBox
from .widgets import VolumeViewer, Mode, VolumeColorizer, IconLabelSelector, VolumeLoader, PatientSelector, MainWindowUI, Color
from scipy import ndimage

class TopLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.patient_selector = PatientSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.label_selector = IconLabelSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        for widget in [self.patient_selector, self.label_selector]:
            self.addWidget(widget)
        self.addStretch()
    def get_widgets(self):
        return self.patient_selector, self.label_selector

class BottomLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.volume_viewer = VolumeViewer()
        self.addWidget(self.volume_viewer)
    def get_widgets(self):
        return self.volume_viewer

class MainWindow(MainWindowUI):
    def __init__(self, data_manager):
        super().__init__(TopLayout, BottomLayout)
        self.loader = VolumeLoader(data_manager, self._handle_volumes, keep_origin=True)

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector, self.label_selector = self.top_layout.get_widgets()
        self.volume_viewer = self.bottom_layout.get_widgets()

        self.patient_selector.setup(data_manager.patients, self.loader.load_patient)

        self.volume_viewer.set_titles(Mode.get_title(Mode.REFINE), 'Instance')

        self.loader.setup(self.patient_selector, self.label_selector)

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)

        self.volume = None

    def _handle_volumes(self, volumes):
        self.volume = volumes['origin'][Mode.REFINE]
        volume, tooth_count = volumes[Mode.REFINE]
        bone_volume = volumes['origin'][Mode.BONE_CONNECTED_COMPONENT]

        erosion = ndimage.binary_erosion(bone_volume > 0)
        bone_surface = (bone_volume > 0) & (~erosion)
        volume[bone_surface] = Color.BONE

        self.volume_viewer.views[0].view.update_volume(volume)

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

    data_manager = DataManager(experiment_name, patient_fold_map, [Mode.REFINE, Mode.BONE_CONNECTED_COMPONENT])
    window = MainWindow(data_manager)

    window.show()

    app.exec()
