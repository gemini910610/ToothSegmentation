from collections import defaultdict
from pathlib import Path

import cv2
import numpy
from PySide6.QtCore import Qt
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget
from scipy import ndimage

from constants.color import GLASBEY_PALETTE, Color
from constants.label import Label
from panoramic.predict import predict_panoramic_data
from post_processing.bone_loss import (
    restore_coordinates,
    slice_area_bone_loss,
    slice_contour_bone_loss,
    split_volume,
    volume_bone_loss,
)
from post_processing.panoramic import crop_panoramic, find_panoramic_image
from post_processing.tooth_slice import (
    align_crop_tooth,
    extract_single_tooth,
    find_normal_vectors,
    get_slices,
    normalize_slice,
)
from post_processing.utils import get_bone_surface, get_tooth_surface
from utils.image import imread
from visualize.ui.buttons import RadioGroup
from visualize.ui.grids import ImageCell, PointImageGrid
from visualize.ui.images import AutoFitImage, bgr_to_pixmap
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.selectors import IconLabelSelector
from visualize.ui.sliders import PercentSlider
from visualize.ui.views import SyncView, VolumeViewer
from visualize.viewer.pages.page import Page
from visualize.viewer.pipeline import BONE_FILENAME, IMAGE_FILENAME, TOOTH_FILENAME
from visualize.viewer.tooth_data import is_excluded, load_tooth_data


class BoneLossControls(HorizontalLayout):
    def _init_widgets(self):
        return [
            IconLabelSelector(),
            QCheckBox('Display All Teeth'),
            None,
            PercentSlider(),
            None,
            QCheckBox('Display Points'),
            QCheckBox('Display Mask'),
            RadioGroup(['Segmentation', 'Image', 'Tooth']),
        ]

class BoneLossContent(HorizontalLayout):
    def _init_widgets(self):
        return [
            VolumeViewer(2, (512, 512)),
            PointImageGrid([
                ImageCell('0 (B)', 0, 0),
                ImageCell('45 (MB)', 1, 0),
                ImageCell('90 (M)', 2, 0),
                ImageCell('135 (ML)', 3, 0),
                ImageCell('180 (L)', 0, 1),
                ImageCell('225 (DL)', 1, 1),
                ImageCell('270 (D)', 2, 1),
                ImageCell('315 (DB)', 3, 1),
            ]),
            BoneLossSummary(),
        ]

class BoneLossSummary(QWidget):
    def __init__(self):
        super().__init__()

        self.crop_container = QWidget()
        self.crop_layout = QHBoxLayout(self.crop_container)
        self.crop_layout.setContentsMargins(0, 0, 0, 0)
        self.crop_images = []

        self.volume_bone_loss_label = QLabel('3D Bone Loss: -')
        self.mean_contour_label = QLabel('Mean Contour: -')
        self.mean_area_label = QLabel('Mean Area: -')
        self.max_contour_label = QLabel('Max Contour: -')
        self.max_area_label = QLabel('Max Area: -')
        self.keypoint_bone_loss_label = QLabel('2D Bone Loss: -')

        layout = QVBoxLayout(self)
        layout.addWidget(self.crop_container)
        layout.addWidget(self.volume_bone_loss_label)
        layout.addWidget(self.keypoint_bone_loss_label)
        layout.addWidget(self.mean_contour_label)
        layout.addWidget(self.mean_area_label)
        layout.addWidget(self.max_contour_label)
        layout.addWidget(self.max_area_label)
        layout.addStretch()

    def _update_summary(self, crop_pixmaps, volume_bone_loss, mean_contour, mean_area, max_contour, max_area, keypoint_bone_losses):
        while self.crop_layout.count():
            item = self.crop_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.crop_images = []

        for pixmap in crop_pixmaps:
            image = AutoFitImage()
            image.setFixedWidth(75)
            image.update_image(pixmap)
            self.crop_layout.addWidget(image)
            self.crop_images.append(image)

        self.crop_layout.addStretch()

        self.volume_bone_loss_label.setText(f'3D Bone Loss: {volume_bone_loss}%')
        self.mean_contour_label.setText(f'Mean Contour: {mean_contour:.1f}%')
        self.mean_area_label.setText(f'Mean Area: {mean_area:.1f}%')
        self.max_contour_label.setText(f'Max Contour: {max_contour}%')
        self.max_area_label.setText(f'Max Area: {max_area}%')

        if not keypoint_bone_losses:
            self.keypoint_bone_loss_label.setText('2D Bone Loss: -')
        else:
            joined = '/'.join(str(loss) for loss in keypoint_bone_losses)
            self.keypoint_bone_loss_label.setText(f'2D Bone Loss: {joined}%')

class BoneLossPage(Page):
    def __init__(self, cej_finder, panoramic_models, working_dir=Path('working')):
        super().__init__(working_dir)
        self.cej_finder = cej_finder
        self.panoramic_models = panoramic_models

        self.controls = BoneLossControls()
        self.content = BoneLossContent()

        layout = QVBoxLayout(self)
        layout.addLayout(self.controls)
        layout.addLayout(self.content)

        self.label_selector, self.all_teeth_toggle, self.bone_opacity_slider, self.point_toggle, self.mask_toggle, self.slice_selector = self.controls.widgets
        self.volume_viewer, self.image_grid, self.summary = self.content.widgets

        self.point_toggle.setChecked(True)
        self.mask_toggle.setChecked(True)
        self.all_teeth_toggle.setChecked(True)
        self.slice_selector.select_first()

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.bone_opacity_slider.on_release(self._on_bone_opacity_changed)
        self.point_toggle.stateChanged.connect(lambda: self._on_slice_changed(self.slice_selector.checked_title()))
        self.mask_toggle.stateChanged.connect(lambda: self._on_slice_changed(self.slice_selector.checked_title()))
        self.all_teeth_toggle.stateChanged.connect(self._on_all_teeth_changed)
        self.slice_selector.on_change(self._on_slice_changed)
        self.volume_viewer.views[0].picked.connect(self._on_picked)

        shortcut_left = QShortcut(Qt.Key.Key_Left, self)
        shortcut_left.activated.connect(lambda: self._on_label_step(-1))
        shortcut_right = QShortcut(Qt.Key.Key_Right, self)
        shortcut_right.activated.connect(lambda: self._on_label_step(1))

        self.volume = None
        self.segmentation_volume = None
        self.bone_mask = None
        self.bone_coordinates = None
        self.image_coefficients = None
        self.normal_vectors = None
        self.previous_label = None
        self.slices = None
        self.points = None
        self.area_masks = None
        self.panoramic_image = None
        self.tooth_datum = defaultdict(list)
        self.excluded_labels = []

    @property
    def is_ready(self):
        return self.volume is not None

    def closeEvent(self, event):
        super().closeEvent(event)
        self.volume_viewer.close()

    def set_patient(self, patient):
        self.patient = patient
        self.volume = None
        self.segmentation_volume = None
        self.previous_label = None
        self.slices = None
        self.points = None
        self.area_masks = None
        self.panoramic_image = None
        self.tooth_datum = defaultdict(list)
        self.exclude_types = {}
        self.volume_viewer.views[0].clear_actors()
        self.volume_viewer.views[1].clear_actors()

        if not patient:
            return

        patient_dir = self.working_dir / patient
        tooth_path = patient_dir / TOOTH_FILENAME
        bone_path = patient_dir / BONE_FILENAME
        image_path = patient_dir / IMAGE_FILENAME
        if not (tooth_path.is_file() and bone_path.is_file() and image_path.is_file()):
            return

        tooth_volume = numpy.load(tooth_path)
        bone_volume = numpy.load(bone_path)
        image_volume = numpy.load(image_path)
        tooth_data = load_tooth_data(patient_dir)

        panoramic_image_path = find_panoramic_image(self.working_dir, patient)
        if panoramic_image_path is not None:
            panoramic_image = imread(panoramic_image_path)
            keypoint_tooth_data = predict_panoramic_data(panoramic_image_path, self.panoramic_models)
            self.panoramic_image = cv2.cvtColor(panoramic_image, cv2.COLOR_GRAY2RGB)
            for tooth in keypoint_tooth_data:
                for fdi in tooth['FDI']:
                    self.tooth_datum[int(fdi)].append(tooth)

        lookup_table = numpy.arange(tooth_volume.max() + 1, dtype=numpy.uint8)
        self.exclude_types = {}
        next_excluded_label = 50
        for label in numpy.unique(tooth_volume):
            label = int(label)
            if label <= 1:
                continue

            value = tooth_data.get(label)
            if value is not None and is_excluded(value):
                lookup_table[label] = next_excluded_label
                self.exclude_types[next_excluded_label] = value
                next_excluded_label += 1
            else:
                lookup_table[label] = int(value) if value is not None else label

        self.volume = lookup_table[tooth_volume]
        self.bone_mask = bone_volume > 0
        self.segmentation_volume = self.volume + self.bone_mask
        self.bone_coordinates = numpy.argwhere(self.bone_mask)
        self.image_coefficients = ndimage.spline_filter(image_volume, order=3)

        labels = numpy.unique(self.segmentation_volume)

        show_all = self.all_teeth_toggle.isChecked()
        self.excluded_labels = []
        render_settings = {1: {'name': 'bone', 'color': tuple(c / 255 for c in Color.BONE), 'opacity': self.bone_opacity_slider.value()}}
        for label in labels:
            if label <= 1:
                continue

            exclude_type = self.exclude_types.get(int(label))
            if exclude_type is not None:
                self.excluded_labels.append(int(label))

            render_settings[int(label)] = {
                'name': str(int(label)),
                'color': tuple(c / 255 for c in Color.TOOTH) if exclude_type is None else (0, 0, 0),
                'opacity': self._excluded_tooth_opacity(int(label), show_all)
            }

        self.volume_viewer.views[0].update_volume(self.segmentation_volume, render_settings=render_settings)

        self.normal_vectors = find_normal_vectors(self.segmentation_volume)

        self.volume_viewer.boxes[0].setTitle(patient)

        labels = numpy.array([label for label in labels if label > 1 and int(label) not in self.exclude_types])

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
        contour_bone_losses = []
        area_bone_losses = []
        for raw_segmentation_slice, point in zip(raw_segmentation_slices, self.points):
            tooth_area = raw_segmentation_slice == Label.TOOTH
            bone_area = raw_segmentation_slice == Label.BONE
            tooth_surface = get_tooth_surface(tooth_area)
            bone_surface = get_bone_surface(tooth_area, bone_area)

            contour_bone_loss = slice_contour_bone_loss(tooth_surface, bone_surface, point, is_upper)
            area_bone_loss, area_mask = slice_area_bone_loss(tooth_area, bone_surface, point, is_upper)

            titles.append(f'{contour_bone_loss}%/{area_bone_loss}%')
            self.area_masks.append(area_mask)
            contour_bone_losses.append(contour_bone_loss)
            area_bone_losses.append(area_bone_loss)

        self.image_grid.update_titles(titles)

        restored_points, aligned_points = restore_coordinates(self.points, aligned_segmentation.shape, transform_meta)
        filtered_tooth_volume = filtered_segmentation == Label.TOOTH
        filtered_bone_volume = filtered_segmentation == Label.BONE
        upper_volume, lower_volume = split_volume(restored_points, filtered_tooth_volume, transform_meta)
        root_volume = upper_volume if is_upper else lower_volume

        tooth_volume_bone_loss = volume_bone_loss(filtered_tooth_volume, filtered_bone_volume, aligned_points, is_upper, transform_meta)

        teeth = self.tooth_datum.get(label, [])
        keypoint_bone_losses = [tooth['bone_loss'] for tooth in teeth]
        crop_pixmaps = [] if self.panoramic_image is None else [bgr_to_pixmap(crop_panoramic(self.panoramic_image, tooth)) for tooth in teeth]

        self.summary._update_summary(
            crop_pixmaps,
            tooth_volume_bone_loss,
            sum(contour_bone_losses) / len(contour_bone_losses),
            sum(area_bone_losses) / len(area_bone_losses),
            max(contour_bone_losses),
            max(area_bone_losses),
            keypoint_bone_losses,
        )

        filtered_segmentation[*root_volume.T] = 3
        render_settings = SyncView.get_settings()
        render_settings[Label.BONE]['opacity'] = self.bone_opacity_slider.value()
        render_settings[3] = SyncView.color_setting(name='surface', color=(220, 70, 70), opacity=1)
        self.volume_viewer.views[1].update_volume(filtered_segmentation, reset=False, render_settings=render_settings)

        self._on_slice_changed(self.slice_selector.checked_title())

    def _on_bone_opacity_changed(self, opacity):
        if self.segmentation_volume is None:
            return

        for view in self.volume_viewer.views:
            view.actors['bone'].prop.opacity = opacity

    def _excluded_tooth_opacity(self, label, show_all):
        exclude_type = self.exclude_types.get(label)
        if exclude_type is None:
            return 1

        if exclude_type == 'Noise':
            return 0

        if exclude_type == 'Filtered':
            return 0.5

        return 0.5 if show_all else 0

    def _on_all_teeth_changed(self):
        if self.segmentation_volume is None:
            return

        show_all = self.all_teeth_toggle.isChecked()
        for label in self.excluded_labels:
            actor = self.volume_viewer.views[0].actors.get(str(label))
            if actor is not None:
                actor.prop.opacity = self._excluded_tooth_opacity(label, show_all)

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
