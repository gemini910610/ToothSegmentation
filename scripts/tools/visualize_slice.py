from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QButtonGroup, QRadioButton
from .widgets import VolumeViewer, Mode, VolumeColorizer, ImageTable, IconLabelSelector, VolumeLoader
from scripts.post_processing.tooth_slice import get_slices, crop_single_tooth, normalize_slice

class MainWindowUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.move(0, 0)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setCentralWidget(widget)

        top_layout = QHBoxLayout()
        self.patient_selector = QComboBox(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.label_selector = IconLabelSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        for widget in [self.patient_selector, self.label_selector]:
            top_layout.addWidget(widget)
        top_layout.addStretch()
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
        self.data_manager = data_manager

        self.setWindowTitle(self.data_manager.experiment_name)

        self.patient_selector.addItems(self.data_manager.patients)
        self.patient_selector.setCurrentIndex(-1)
        self.patient_selector.currentIndexChanged.connect(self._load_patient)

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)

        self.slice_selector.idClicked.connect(self._on_slice_changed)

        self.volumes = None
        self.slices = None

    def _load_patient(self, index):
        self.patient_selector.setEnabled(False)
        self.label_selector.setEnabled(False)
        for radio in self.slice_selector.buttons():
            radio.setEnabled(False)

        patient = self.data_manager.patients[index]
        self.thread = VolumeLoader(self.data_manager, patient, colorize=False, keep_origin=True)
        self.thread.finished.connect(self._on_volume_loaded)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _on_volume_loaded(self, volumes):
        self.volumes = [volumes['origin'][Mode.RELABELED], volumes['origin'][Mode.IMAGE]]

        self.slice_selector.blockSignals(True)
        tooth_count = self.volumes[0].max() - 1

        self.label_selector.update_items(range(2, tooth_count + 2), tooth_count, -2)
        self.slice_selector.buttons()[0].setChecked(True)
        self.slice_selector.blockSignals(False)
        self._on_label_changed(0)

        self.patient_selector.setEnabled(True)
        self.label_selector.setEnabled(True)
        for radio in self.slice_selector.buttons():
            radio.setEnabled(True)

    def _on_label_changed(self, index):
        if self.volumes is None:
            return

        label = self.label_selector.current_label()
        segmentation_volume, image_volume = self.volumes
        segmentation_volume, image_volume, tooth_volume, center = crop_single_tooth(segmentation_volume, image_volume, tooth_label=label, bone_label=1)
        volume = VolumeColorizer.color_volume(segmentation_volume, display_bone=True)
        self.volume_viewer.views[0].setTitle(f'Label {label}')
        self.volume_viewer.views[0].view.update_volume(volume)

        self.slices = list(get_slices(segmentation_volume, image_volume, tooth_volume, center))
        self._on_slice_changed(self.slice_selector.checkedId())

    def _on_slice_changed(self, index):
        if self.slices is None:
            return

        slices = [normalize_slice(*volume_slice)[index] for volume_slice in self.slices]
        self.image_table.update_images(slices)

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
