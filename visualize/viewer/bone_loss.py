import cv2
import numpy
from PySide6.QtCore import Qt
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import QCheckBox
from scipy import ndimage

from constants.color import GLASBEY_PALETTE, Color
from constants.label import Label
from constants.mode import Mode
from post_processing.bone_loss import (
    get_bone_surface,
    get_tooth_surface,
    restore_coordinates,
    slice_area_bone_loss,
    slice_contour_bone_loss,
    split_volume,
)
from post_processing.tooth_slice import (
    align_crop_tooth,
    extract_single_tooth,
    find_normal_vectors,
    get_slices,
    normalize_slice,
)
from visualize.ui.buttons import RadioGroup
from visualize.ui.grids import ImageCell, PointImageGrid
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.selectors import IconLabelSelector, PatientSelector
from visualize.ui.sliders import PercentSlider
from visualize.ui.views import SyncView, VolumeViewer
from visualize.ui.windows import BaseWindow


class TopLayout(HorizontalLayout):
    def _init_widgets(self):
        return [
            PatientSelector(),
            IconLabelSelector(),
            None,
            PercentSlider(),
            None,
            QCheckBox('Display Points'),
            QCheckBox('Display Mask'),
            RadioGroup(['Segmentation', 'Image', 'Tooth'])
        ]

class BottomLayout(HorizontalLayout):
    def _init_widgets(self):
        return [
            VolumeViewer(2, (640, 640)),
            PointImageGrid([
                ImageCell('0 (B)', 0, 0),
                ImageCell('45 (MB)', 1, 0),
                ImageCell('90 (M)', 2, 0),
                ImageCell('135 (ML)', 3, 0),
                ImageCell('180 (L)', 0, 1),
                ImageCell('225 (DL)', 1, 1),
                ImageCell('270 (D)', 2, 1),
                ImageCell('315 (DB)', 3, 1)
            ])
        ]

class MainWindow(BaseWindow):
    def __init__(self, data_manager, cej_finder):
        super().__init__(TopLayout, BottomLayout)
        self.data_manager = data_manager
        self.cej_finder = cej_finder

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector, self.label_selector, self.bone_opacity_slider, self.point_toggle, self.mask_toggle, self.slice_selector = self.layouts[0].widgets
        self.volume_viewer, self.image_grid = self.layouts[1].widgets

        self.patient_selector.setup(data_manager.patients, self._on_patient_changed)
        self.point_toggle.setChecked(True)
        self.mask_toggle.setChecked(True)
        self.slice_selector.select_first()

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.bone_opacity_slider.on_release(self._on_bone_opacity_changed)
        self.point_toggle.stateChanged.connect(lambda: self._on_slice_changed(self.slice_selector.checked_title()))
        self.mask_toggle.stateChanged.connect(lambda: self._on_slice_changed(self.slice_selector.checked_title()))
        self.slice_selector.on_change(self._on_slice_changed)
        self.volume_viewer.views[0].picked.connect(self._on_picked)

        shortcut_left = QShortcut(Qt.Key.Key_Left, self)
        shortcut_left.activated.connect(lambda: self._on_label_step(-1))
        shortcut_right = QShortcut(Qt.Key.Key_Right, self)
        shortcut_right.activated.connect(lambda: self._on_label_step(1))

        self.segmentation_volume = None
        self.bone_mask = None
        self.bone_coordinates = None
        self.image_coefficients = None
        self.normal_vectors = None
        self.previous_label = None
        self.slices = None
        self.points = None
        self.area_masks = None

    def _on_patient_changed(self, index):
        volumes = self.data_manager.load_data(index)
        fdi_volume = volumes[Mode.FDI]
        bone_volume = volumes[Mode.BONE_FILLED]
        image_volume = volumes[Mode.IMAGE]

        self.bone_mask = bone_volume > 0
        self.segmentation_volume = fdi_volume + self.bone_mask
        self.bone_coordinates = numpy.argwhere(self.bone_mask)
        self.image_coefficients = ndimage.spline_filter(image_volume, order=3)

        labels = numpy.unique(self.segmentation_volume)

        render_settings = {
            1: {'name': 'bone', 'color': tuple(c / 255 for c in Color.BONE), 'opacity': self.bone_opacity_slider.value()},
            **{
                int(label): {
                    'name': str(int(label)),
                    'color': tuple(c / 255 for c in Color.TOOTH),
                    'opacity': 1 if label < Label.TRUNCATED else 0
                }
                for label in labels
                if label > 1
            }
        }
        self.volume_viewer.views[0].update_volume(self.segmentation_volume, render_settings=render_settings)

        self.normal_vectors = find_normal_vectors(self.segmentation_volume)

        patient = self.patient_selector.currentText()
        self.volume_viewer.boxes[0].setTitle(patient)

        labels = labels[(labels > 1) & (labels < Label.TRUNCATED)] # exclude background, bone, removed teeth and missing FDI teeth

        self.label_selector.update_items(labels)

        self.previous_label = None

        self.label_selector.setCurrentIndex(0)

    def _on_label_changed(self, index):
        if self.segmentation_volume is None:
            return

        label = self.label_selector.currentData()
        if label is None:
            return

        is_upper = label // 10 in {1, 2}
        normal_vector = self.normal_vectors[label]

        filtered_segmentation, transform_meta = extract_single_tooth(self.segmentation_volume == label, self.bone_mask, self.bone_coordinates, normal_vector)
        aligned_segmentation, aligned_image = align_crop_tooth(filtered_segmentation, self.image_coefficients, transform_meta)
        slices = get_slices(aligned_segmentation, aligned_image, reverse=label // 10 in {2, 3})

        self.volume_viewer.boxes[1].setTitle(f'Label {label}')

        if self.previous_label is not None:
            self.volume_viewer.views[0].actors[str(self.previous_label)].prop.color = tuple(c / 255 for c in Color.TOOTH)
        self.volume_viewer.views[0].actors[str(label)].prop.color = tuple(c / 255 for c in GLASBEY_PALETTE[label])
        self.previous_label = label

        raw_segmentation_slices = []
        segmentation_slices = []
        image_slices = []
        tooth_slices = []
        for raw_segmentation_slice, raw_image_slice in slices:
            segmentation_slice, image_slice = normalize_slice(raw_segmentation_slice, raw_image_slice)
            tooth_slice = image_slice * (segmentation_slice == 127).astype(numpy.uint8)

            raw_segmentation_slices.append(raw_segmentation_slice)
            segmentation_slices.append(segmentation_slice)
            image_slices.append(image_slice)
            tooth_slices.append(tooth_slice)

        self.slices = {'Segmentation': segmentation_slices, 'Image': image_slices, 'Tooth': tooth_slices}
        self.points = self.cej_finder.find(raw_segmentation_slices, tooth_slices)

        self.area_masks = []
        titles = []
        for raw_segmentation_slice, point in zip(raw_segmentation_slices, self.points):
            tooth_area = raw_segmentation_slice == Label.TOOTH
            bone_area = raw_segmentation_slice == Label.BONE
            tooth_surface = get_tooth_surface(tooth_area)
            bone_surface = get_bone_surface(tooth_area, bone_area)

            contour_bone_loss = slice_contour_bone_loss(tooth_surface, bone_surface, point, is_upper)
            area_bone_loss, area_mask = slice_area_bone_loss(tooth_area, bone_surface, point, is_upper)

            titles.append(f'{contour_bone_loss}%/{area_bone_loss}%')
            self.area_masks.append(area_mask)

        self.image_grid.update_titles(titles)

        restored_points, _ = restore_coordinates(self.points, aligned_segmentation.shape, transform_meta)
        filtered_tooth_volume = filtered_segmentation == Label.TOOTH
        upper_volume, lower_volume = split_volume(restored_points, filtered_tooth_volume, transform_meta)
        root_volume = upper_volume if is_upper else lower_volume

        filtered_segmentation[*root_volume.T] = 3
        render_settings = SyncView.get_settings()
        render_settings[Label.BONE]['opacity'] = self.bone_opacity_slider.value()
        render_settings[3] = SyncView._color_setting(name='surface', color=(255, 0, 0), opacity=1)
        self.volume_viewer.views[1].update_volume(filtered_segmentation, reset=False, render_settings=render_settings)

        self._on_slice_changed(self.slice_selector.checked_title())

    def _on_bone_opacity_changed(self, opacity):
        if self.segmentation_volume is None:
            return

        for view in self.volume_viewer.views:
            view.actors['bone'].prop.opacity = opacity

    def _on_slice_changed(self, title):
        if self.slices is None or self.area_masks is None or title is None:
            return

        slices = self.slices[title]
        points = None if not self.point_toggle.isChecked() else self.points
        slices = [cv2.cvtColor(slice_image, cv2.COLOR_GRAY2RGB) for slice_image in slices]

        if self.mask_toggle.isChecked():
            colors = numpy.array([(0, 0, 0), (70, 220, 70), Color.BONE], dtype=numpy.uint8)
            for display_slice, area_mask in zip(slices, self.area_masks):
                mask = area_mask > 0
                display_slice[mask] = colors[area_mask[mask]]

        self.image_grid.update_images(slices, points)

    def _on_picked(self, actor):
        label = actor._label
        index = self.label_selector.findData(label)
        self.label_selector.setCurrentIndex(index)

    def _on_label_step(self, step):
        count = self.label_selector.count()
        index = self.label_selector.currentIndex()
        if index + step < 0 or index + step >= count:
            print('\a', end='', flush=True)
            return

        index = max(0, min(count - 1, index + step))
        self.label_selector.setCurrentIndex(index)

if __name__ == '__main__':
    import sys
    from pathlib import Path

    from keypoints.models import load_model
    from keypoints.predict import CEJFinder
    from post_processing.utils import iterate_datasets
    from tui.apps import BoneLossApp
    from utils.config import load_config
    from visualize.utils import DataManager, run_app

    args = BoneLossApp().run()
    if args is None:
        sys.exit()

    segmentation_experiment_name = args['segmentation_experiment_name']
    keypoint_experiment_name = args['keypoint_experiment_name']
    datasets = args['datasets']

    config = load_config(Path('logs') / keypoint_experiment_name / 'keypoints.toml', fold=1)
    model = load_model(config)
    cej_finder = CEJFinder(model, config.device)

    patients = [
        f'{dataset}/{patient}'
        for dataset, patient in iterate_datasets(segmentation_experiment_name, datasets)
        if (Path('outputs') / segmentation_experiment_name / dataset / patient / f'{Mode.FDI}.npy').exists()
    ]
    data_manager = DataManager(segmentation_experiment_name, patients, [Mode.FDI, Mode.BONE_FILLED, Mode.IMAGE])
    run_app(MainWindow, data_manager, cej_finder)
