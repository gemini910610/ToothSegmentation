import numpy

from PySide6.QtGui import Qt, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QComboBox
from .widgets import VolumeViewer, Label, Mode, VolumeColorizer, IconLabelSelector, VolumeLoader, PatientSelector, MainWindowUI
from scripts.post_processing.tooth_slice import crop_single_tooth
from scripts.post_processing.find_points import ensure_upward, find_root
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
        self.volume_viewer = VolumeViewer(1)
        for widget in [self.volume_viewer]:
            self.addWidget(widget)
    def get_widgets(self):
        return self.volume_viewer

class MainWindow(MainWindowUI):
    def __init__(self, data_manager):
        super().__init__(TopLayout, BottomLayout)
        self.loader = VolumeLoader(data_manager, self._handle_volumes, colorize=False, keep_origin=True)

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector, self.label_selector = self.top_layout.get_widgets()
        self.volume_viewer = self.bottom_layout.get_widgets()

        self.patient_selector.setup(data_manager.patients, self.loader.load_patient)

        self.loader.setup(self.patient_selector, self.label_selector)

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)

        shortcut_left = QShortcut(Qt.Key.Key_Left, self)
        shortcut_left.activated.connect(lambda: self._on_label_step(-1))
        shortcut_right = QShortcut(Qt.Key.Key_Right, self)
        shortcut_right.activated.connect(lambda: self._on_label_step(1))

        self.volumes = None

    def _handle_volumes(self, volumes):
        self.volumes = [volumes['origin'][Mode.POST_PROCESSING], volumes['origin'][Mode.IMAGE]]

        tooth_count = self.volumes[0].max() - 1

        self.label_selector.update_items(range(2, tooth_count + 2), tooth_count, -2)
        self._on_label_changed(0)

    def _on_label_changed(self, index):
        if self.volumes is None:
            return

        label = self.label_selector.current_label()
        segmentation_volume, image_volume = self.volumes
        segmentation_volume, image_volume, tooth_volume, center = crop_single_tooth(segmentation_volume, image_volume, tooth_label=label, bone_label=1)
        segmentation_volume, image_volume, tooth_volume, center = ensure_upward(segmentation_volume, image_volume, tooth_volume, center)

        roots = find_root(segmentation_volume)
        if len(roots) > 1:
            distances = [
                numpy.linalg.norm(roots[i] - roots[j])
                for i in range(len(roots) - 1)
                for j in range(i + 1, len(roots))
            ]
            distances.sort()
            print(f'[{", ".join([f"{distance:.2f}" for distance in distances])}]')

        self.volume_viewer.set_titles(f'Label {label} ({len(roots)} Root)')

        mask = segmentation_volume == Label.TOOTH
        tooth_volume = ndimage.binary_opening(mask, iterations=3)
        volume = numpy.where(mask, tooth_volume * Label.TOOTH, segmentation_volume)

        volume = VolumeColorizer.color_volume(volume)
        palette = VolumeColorizer.glasbey_palette(len(roots))
        for (x, y, z), color in zip(roots, palette):
            volume[x-1:x+2, y-1:y+2, z-1:z+2] = color
        self.volume_viewer.views[0].view.update_volume(volume)

    def _on_label_step(self, step):
        if self.volumes is None:
            return

        count = self.label_selector.count()
        index = self.label_selector.currentIndex()
        if index + step < 0 or index + step >= count:
            print('\a', end='', flush=True)
        index = max(0, min(count - 1, index + step))
        self.label_selector.setCurrentIndex(index)

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

    data_manager = DataManager(experiment_name, patient_fold_map, [Mode.POST_PROCESSING, Mode.IMAGE])
    window = MainWindow(data_manager)

    window.show()

    app.exec()
