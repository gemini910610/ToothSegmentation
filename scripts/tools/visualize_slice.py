import cv2

from PySide6.QtGui import Qt, QShortcut
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QButtonGroup, QRadioButton, QCheckBox
from .widgets import VolumeViewer, Mode, VolumeColorizer, ImageTable, IconLabelSelector, VolumeLoader, PatientSelector
from scripts.post_processing.tooth_slice import get_slices, crop_single_tooth, normalize_slice
from scripts.post_processing.find_points import ensure_upward, CEJFinder, find_bone_point

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
        self.point_toggle = QCheckBox('Display Points')
        top_layout.addWidget(self.point_toggle)
        self.slice_selector = QButtonGroup()
        group_layout = QHBoxLayout()
        for index, title in enumerate(['Segmentation', 'Image', 'Tooth']):
            radio = QRadioButton(title)
            self.slice_selector.addButton(radio, index)
            group_layout.addWidget(radio)
        self.slice_selector.buttons()[0].setChecked(True)
        top_layout.addLayout(group_layout)
        layout.addLayout(top_layout)

        bottom_layout = QHBoxLayout()
        self.volume_viewer = VolumeViewer(1)
        self.image_table = ImageTable()
        for widget in [self.volume_viewer, self.image_table]:
            bottom_layout.addWidget(widget)
        layout.addLayout(bottom_layout)

class MainWindow(MainWindowUI):
    def __init__(self, data_manager):
        super().__init__()
        self.loader = VolumeLoader(data_manager, self._handle_volumes, colorize=False, keep_origin=True)

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector.setup(data_manager.patients, self.loader.load_patient)

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.point_toggle.stateChanged.connect(lambda: self._on_slice_changed(self.slice_selector.checkedId()))
        self.slice_selector.idClicked.connect(self._on_slice_changed)

        self.loader.setup(self.patient_selector, self.label_selector, self.point_toggle, *self.slice_selector.buttons())

        shortcut_left = QShortcut(Qt.Key.Key_Left, self)
        shortcut_left.activated.connect(lambda: self._on_label_step(-1))
        shortcut_right = QShortcut(Qt.Key.Key_Right, self)
        shortcut_right.activated.connect(lambda: self._on_label_step(1))

        self.volumes = None
        self.slices = None
        self.cej_finder = None

    def _handle_volumes(self, volumes):
        self.volumes = [volumes['origin'][Mode.RELABELED], volumes['origin'][Mode.IMAGE]]

        self.slice_selector.blockSignals(True)
        tooth_count = self.volumes[0].max() - 1

        self.label_selector.update_items(range(2, tooth_count + 2), tooth_count, -2)
        self.slice_selector.buttons()[0].setChecked(True)
        self.slice_selector.blockSignals(False)
        self._on_label_changed(0)

    def _on_label_changed(self, index):
        if self.volumes is None:
            return

        label = self.label_selector.current_label()
        segmentation_volume, image_volume = self.volumes
        segmentation_volume, image_volume, tooth_volume, center = crop_single_tooth(segmentation_volume, image_volume, tooth_label=label, bone_label=1)
        segmentation_volume, image_volume, tooth_volume, center = ensure_upward(segmentation_volume, image_volume, tooth_volume, center)

        volume = VolumeColorizer.color_volume(segmentation_volume, display_bone=True)

        self.volume_viewer.set_titles(f'Label {label}')
        self.volume_viewer.views[0].view.update_volume(volume)

        self.slices = list(get_slices(segmentation_volume, image_volume, tooth_volume, center))
        self._on_slice_changed(self.slice_selector.checkedId())

        self.cej_finder = CEJFinder(tooth_volume)

    def _on_slice_changed(self, index):
        if self.slices is None:
            return

        slices = []
        for segmentation_slice, image_slice, tooth_slice in self.slices:
            volume_slice = normalize_slice(segmentation_slice, image_slice, tooth_slice)[index]
            volume_slice = cv2.cvtColor(volume_slice, cv2.COLOR_GRAY2BGR)

            if self.point_toggle.isChecked():
                if index in [1, 2]:
                    left_cej, right_cej = self.cej_finder.find(tooth_slice)
                    volume_slice = cv2.circle(volume_slice, left_cej, 2, (0, 255, 255), -1)
                    volume_slice = cv2.circle(volume_slice, right_cej, 2, (0, 255, 255), -1)

                if index in [0, 1]:
                    left_bone, right_bone = find_bone_point(segmentation_slice)
                    volume_slice = cv2.circle(volume_slice, left_bone, 2, (255, 0, 0), -1)
                    volume_slice = cv2.circle(volume_slice, right_bone, 2, (255, 0, 0), -1)

            slices.append(volume_slice)

        self.image_table.update_images(slices)

    def _on_label_step(self, step):
        if self.volumes is None:
            return

        count = self.label_selector.count()
        index = self.label_selector.currentIndex()
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

    data_manager = DataManager(experiment_name, patient_fold_map, [Mode.RELABELED, Mode.IMAGE], [])
    window = MainWindow(data_manager)

    window.show()

    app.exec()
