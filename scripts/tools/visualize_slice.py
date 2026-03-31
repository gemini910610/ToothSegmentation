import numpy

from PySide6.QtWidgets import QHBoxLayout, QComboBox, QButtonGroup, QRadioButton
from PySide6.QtGui import QShortcut, Qt
from .widgets import PatientSelector, IconLabelSelector, VolumeViewer, MainWindowUI, VolumeLoader, Mode, VolumeColorizer, Label, Color, AxisItem, ImageTable
from scripts.post_processing.tooth_slice import extract_single_tooth, align_crop_tooth, get_slices, normalize_slice, find_normal_vectors

class TopLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.patient_selector = PatientSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.label_selector = IconLabelSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        for widget in [self.patient_selector, self.label_selector]:
            self.addWidget(widget)
        self.addStretch()

        self.slice_selector = QButtonGroup()
        group_layout = QHBoxLayout()
        for index, title in enumerate(['Segmentation', 'Image', 'Tooth']):
            radio = QRadioButton(title)
            self.slice_selector.addButton(radio, index)
            group_layout.addWidget(radio)
        self.addLayout(group_layout)
    def get_widgets(self):
        return self.patient_selector, self.label_selector, self.slice_selector

class BottomLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.volume_viewer = VolumeViewer()
        self.image_table = ImageTable()
        for widget in [self.volume_viewer, self.image_table]:
            self.addWidget(widget)
        self.axis_item = AxisItem(self.volume_viewer.views[1].view)
    def get_widgets(self):
        return self.volume_viewer, self.image_table, self.axis_item

class MainWindow(MainWindowUI):
    def __init__(self, data_manager):
        super().__init__(TopLayout, BottomLayout)
        self.loader = VolumeLoader(data_manager, self._handle_volumes, colorize=False, keep_origin=True)

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector, self.label_selector, self.slice_selector = self.top_layout.get_widgets()
        self.volume_viewer, self.image_table, self.axis_item = self.bottom_layout.get_widgets()

        self.patient_selector.setup(data_manager.patients, self.loader.load_patient)
        self.slice_selector.buttons()[0].setChecked(True)

        self.loader.setup(self.patient_selector, self.label_selector, *self.slice_selector.buttons())

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.slice_selector.idClicked.connect(self._on_slice_changed)
        for view in self.volume_viewer.views:
            view.view.viewChanged.connect(self.axis_item.sync_camera)

        shortcut_left = QShortcut(Qt.Key.Key_Left, self)
        shortcut_left.activated.connect(lambda: self._on_label_step(-1))
        shortcut_right = QShortcut(Qt.Key.Key_Right, self)
        shortcut_right.activated.connect(lambda: self._on_label_step(1))

        self.volumes = None
        self.display_volume = None
        self.previous_label_mask = None
        self.colors = None
        self.slices = None
        self.normal_vectors = None

    def _handle_volumes(self, volumes):
        self.volumes = [volumes['origin'][Mode.POST_PROCESSING], volumes['origin'][Mode.IMAGE]]

        volume = self.volumes[0]
        volume = numpy.select([volume == 1, (volume > 1) & (volume < Label.CROPPED)], [Label.BONE, Label.TOOTH])
        self.display_volume = VolumeColorizer.color_volume(volume, display_bone=True)

        self.normal_vectors = find_normal_vectors(self.volumes[0])

        patient = self.patient_selector.currentText()
        self.volume_viewer.views[0].setTitle(patient)

        labels = numpy.unique(self.volumes[0])
        labels = labels[(labels > 1) & (labels < Label.CROPPED)] # exclude background, bone and removed teeth

        tooth_count = self.volumes[0].max() - 1
        self.colors = VolumeColorizer.glasbey_palette(tooth_count)

        self.label_selector.update_items(labels, tooth_count, -2)
        self.previous_label_mask = None
        self._on_label_changed(0, reset=True)

    def _on_label_changed(self, index, reset=False):
        if self.volumes is None or self.display_volume is None or self.normal_vectors is None:
            return

        label = self.label_selector.current_label()
        segmentation_volume, image_volume = self.volumes
        normal_vector = self.normal_vectors[label]
        filtered_segmentation, transform_meta = extract_single_tooth(segmentation_volume, normal_vector, tooth_label=label, bone_label=1)
        aligned_segmentation, aligned_image = align_crop_tooth(filtered_segmentation, image_volume, transform_meta)
        self.slices = get_slices(aligned_segmentation, aligned_image, reverse=label // 10 in {2, 3})

        volume = VolumeColorizer.color_volume(filtered_segmentation, display_bone=True)
        self.volume_viewer.views[1].view.update_volume(volume, reset=reset)

        self.volume_viewer.views[1].setTitle(f'Label {label}')

        mask = self.volumes[0] == label
        color = self.colors[label - 2]
        if self.previous_label_mask is not None:
            self.display_volume[self.previous_label_mask] = Color.TOOTH
        self.display_volume[mask] = color
        self.previous_label_mask = mask

        self.volume_viewer.views[0].view.update_volume(self.display_volume, reset=reset)

        self.axis_item.update_axes(*transform_meta['axes'])
        self.axis_item.sync_camera(self.volume_viewer.views[1].view.opts)

        self._on_slice_changed(0)

    def _on_label_step(self, step):
        if self.volumes is None:
            return

        count = self.label_selector.count()
        index = self.label_selector.currentIndex()
        if index + step < 0 or index + step >= count:
            print('\a', end='', flush=True)
        index = max(0, min(count - 1, index + step))
        self.label_selector.setCurrentIndex(index)

    def _on_slice_changed(self, index):
        if self.slices is None:
            return

        slices = []
        for segmentation_slice, image_slice, tooth_slice in self.slices:
            volume_slice = normalize_slice(segmentation_slice, image_slice, tooth_slice)[index]
            slices.append(volume_slice)

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

    data_manager = DataManager(experiment_name, patient_fold_map, [Mode.POST_PROCESSING, Mode.IMAGE])
    window = MainWindow(data_manager)

    window.show()

    app.exec()
