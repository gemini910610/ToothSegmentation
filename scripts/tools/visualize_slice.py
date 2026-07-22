import cv2
import numpy
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QRadioButton

from keypoint.predict import CEJFinder
from scripts.post_processing.bone_loss import restore_coordinates, slice_bone_loss, slice_area_bone_loss, split_volume
from scripts.post_processing.tooth_slice import align_crop_tooth, extract_single_tooth, find_normal_vectors, get_slices, normalize_slice
from scripts.tools.widgets import Color, glasbey_palette, IconLabelSelector, ImageCell, ImageTable, Label, MainWindowUI, Mode, PatientSelector, SyncView, VolumeViewer

class TopLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.patient_selector = PatientSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.label_selector = IconLabelSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        for widget in [self.patient_selector, self.label_selector]:
            self.addWidget(widget)
        self.addStretch()

        self.point_toggle = QCheckBox('Display Points')
        self.addWidget(self.point_toggle)

        self.slice_selector = QButtonGroup()
        group_layout = QHBoxLayout()
        for index, title in enumerate(['Segmentation', 'Image', 'Tooth']):
            radio = QRadioButton(title)
            self.slice_selector.addButton(radio, index)
            group_layout.addWidget(radio)
        self.addLayout(group_layout)

    def get_widgets(self):
        return self.patient_selector, self.label_selector, self.point_toggle, self.slice_selector

class BottomLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.volume_viewer = VolumeViewer()
        self.image_table = ImageTable([
            ImageCell('0 (B)', 0, 0),
            ImageCell('45 (MB)', 1, 0),
            ImageCell('90 (M)', 2, 0),
            ImageCell('135 (ML)', 3, 0),
            ImageCell('180 (L)', 0, 1),
            ImageCell('225 (DL)', 1, 1),
            ImageCell('270 (D)', 2, 1),
            ImageCell('315 (DB)', 3, 1),
        ])
        for widget in [self.volume_viewer, self.image_table]:
            self.addWidget(widget)

    def get_widgets(self):
        return self.volume_viewer, self.image_table

class MainWindow(MainWindowUI):
    def __init__(self, data_manager, keypoint_experiment_name, default_patient=None, default_label=None):
        super().__init__(TopLayout, BottomLayout)
        self.data_manager = data_manager
        self.default_label = default_label if default_patient is not None else None

        keypoint_model_path = os.path.join('logs', keypoint_experiment_name, 'best.pth')
        self.cej_finder = CEJFinder(keypoint_model_path)

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector, self.label_selector, self.point_toggle, self.slice_selector = self.layouts[0].get_widgets()
        self.volume_viewer, self.image_table = self.layouts[1].get_widgets()

        self.patient_selector.setup(data_manager.patients, self._on_patient_changed)
        self.point_toggle.setChecked(True)
        self.slice_selector.buttons()[0].setChecked(True)

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.point_toggle.stateChanged.connect(lambda x: self._on_slice_changed(self.slice_selector.checkedId()))
        self.slice_selector.idClicked.connect(self._on_slice_changed)
        self.volume_viewer.boxes[0].view.picked.connect(self._on_picked)

        shortcut_left = QShortcut(Qt.Key.Key_Left, self)
        shortcut_left.activated.connect(lambda: self._on_label_step(-1))
        shortcut_right = QShortcut(Qt.Key.Key_Right, self)
        shortcut_right.activated.connect(lambda: self._on_label_step(1))

        self.fdi_volume = None
        self.image_volume = None
        self.normal_vectors = None
        self.previous_label = None
        self.slices = None
        self.points = None
        self.area_masks = None

        if default_patient is not None:
            index = self.data_manager.patients.index(default_patient)
            self.patient_selector.setCurrentIndex(index)

    def _on_picked(self, actor):
        label = actor._label
        index = self.label_selector.findData(label)
        self.label_selector.setCurrentIndex(index)

    def _on_patient_changed(self, index):
        volumes = self.data_manager.load_data(index)
        self.fdi_volume = volumes[Mode.FDI]
        self.image_volume = volumes[Mode.IMAGE]

        settings = {
            1: {
                'name': 'bone',
                'color': Color.BONE,
                'opacity': 0.5
            },
            **{
                label: {
                    'name': str(label),
                    'color': Color.TOOTH,
                    'opacity': 1 if label < Label.CROPPED else 0
                }
                for label in numpy.unique(self.fdi_volume)
                if label > 1
            }
        }
        self.volume_viewer.boxes[0].view.update_volume(self.fdi_volume, render_settings=settings)

        self.normal_vectors = find_normal_vectors(self.fdi_volume)

        patient = self.patient_selector.currentText()
        self.volume_viewer.boxes[0].setTitle(patient)

        labels = numpy.unique(self.fdi_volume)
        labels = labels[(labels > 1) & (labels < Label.CROPPED)] # exclude background, bone, removed teeth and missing FDI teeth

        self.label_selector.update_items(labels, self.fdi_volume.max(), -2)

        self.previous_label = None

        if self.default_label is not None and self.default_label in labels:
            index = labels.tolist().index(self.default_label)
            self.label_selector.blockSignals(True)
            self.label_selector.setCurrentIndex(index)
            self.label_selector.blockSignals(False)
            self._on_label_changed(index, reset=True)
        else:
            self._on_label_changed(0, reset=True)
        self.default_label = None

    def _on_label_changed(self, index, reset=False):
        if self.fdi_volume is None or self.image_volume is None or self.normal_vectors is None:
            return

        label = self.label_selector.current_label()
        normal_vector = self.normal_vectors[label]

        filtered_segmentation, transform_meta = extract_single_tooth(self.fdi_volume, normal_vector, tooth_label=label, bone_label=1)
        aligned_segmentation, aligned_image = align_crop_tooth(filtered_segmentation, self.image_volume, transform_meta)
        slices = get_slices(aligned_segmentation, aligned_image, reverse=label // 10 in {2, 3})

        self.volume_viewer.boxes[1].setTitle(f'Label {label}')

        if self.previous_label is not None:
            self.volume_viewer.boxes[0].view.actors[str(self.previous_label)].prop.color = tuple(c / 255 for c in Color.TOOTH)
        palette = glasbey_palette(48)
        self.volume_viewer.boxes[0].view.actors[str(label)].prop.color = tuple(c / 255 for c in palette[label-2])
        self.previous_label = label

        origin_segmentation_slices = []
        segmentation_slices = []
        image_slices = []
        tooth_slices = []
        for origin_segmentation_slice, image_slice, tooth_slice in slices:
            segmentation_slice, image_slice, tooth_slice = normalize_slice(origin_segmentation_slice, image_slice, tooth_slice)

            origin_segmentation_slices.append(origin_segmentation_slice)
            segmentation_slices.append(segmentation_slice)
            image_slices.append(image_slice)
            tooth_slices.append(tooth_slice)

        self.slices = [segmentation_slices, image_slices, tooth_slices]
        self.points = self.cej_finder.find(origin_segmentation_slices, tooth_slices)

        self.area_masks = []
        for box, segmentation_slice, point in zip(self.image_table.boxes, origin_segmentation_slices, self.points):
            bone_loss = slice_bone_loss(segmentation_slice, point, label)
            area_bone_loss, area_mask = slice_area_bone_loss(segmentation_slice, point, label)
            title = box.title()
            index = title.find(')')
            box.setTitle(f'{title[:index+1]} {bone_loss}%/{area_bone_loss}%')
            self.area_masks.append(area_mask)

        points = restore_coordinates(self.points, aligned_segmentation.shape, transform_meta)
        upper_volume, lower_volume = split_volume(points, filtered_segmentation, transform_meta)
        root_volume = upper_volume if label // 10 in {1, 2} else lower_volume

        filtered_segmentation[*root_volume.T] = 3
        render_settings = SyncView.get_settings()
        render_settings[3] = {
            'name': 'surface',
            'color': (255, 0, 0),
            'opacity': 1
        }
        self.volume_viewer.boxes[1].view.update_volume(filtered_segmentation, reset=False, render_settings=render_settings)
        # self.volume_viewer.boxes[1].view.update_points(points)

        self._on_slice_changed(self.slice_selector.checkedId())

    def _on_label_step(self, step):
        count = self.label_selector.count()
        index = self.label_selector.currentIndex()
        if index + step < 0 or index + step >= count:
            print('\a', end='', flush=True)
            return

        index = max(0, min(count - 1, index + step))
        self.label_selector.setCurrentIndex(index)

    def _on_slice_changed(self, index):
        if self.slices is None or self.area_masks is None:
            return

        slices = self.slices[index]
        points = None if not self.point_toggle.isChecked() else self.points
        slices = [cv2.cvtColor(slice_image, cv2.COLOR_GRAY2RGB) for slice_image in slices]

        if index == 0:
            colors = numpy.array([(0, 0, 0), (70, 220, 70), Color.BONE], dtype=numpy.uint8)
            for display_slice, area_mask in zip(slices, self.area_masks):
                mask = area_mask > 0
                display_slice[mask] = colors[area_mask[mask]]

        self.image_table.update_images(slices, points)

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
    parser.add_argument('segmentation_exp', type=str)
    parser.add_argument('keypoint_exp', type=str)
    parser.add_argument('--patient', type=str)
    parser.add_argument('--label', type=int)
    args = parser.parse_args()

    segmentation_experiment_name = args.segmentation_exp
    keypoint_experiment_name = args.keypoint_exp
    default_patient = args.patient
    default_label = args.label

    datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
    patients = [f'{dataset}/{patient}' for dataset, patient in iterate_datasets(segmentation_experiment_name, datasets)]

    if default_patient is not None and default_patient not in patients:
        raise ValueError(f'Patient {default_patient} not found')

    app = QApplication([])

    data_manager = DataManager(segmentation_experiment_name, patients, [Mode.FDI, Mode.IMAGE])
    window = MainWindow(data_manager, keypoint_experiment_name, default_patient, default_label)

    window.show()

    app.exec()
