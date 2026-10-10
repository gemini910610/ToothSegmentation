from pathlib import Path

import cv2
import numpy
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton, QVBoxLayout

from constants.label import Label
from panoramic.predict import predict_panoramic_data
from post_processing.panoramic import (
    draw_label,
    draw_panoramic,
    find_panoramic_image,
    get_panoramic,
)
from post_processing.relabel_fdi import relabel_fdi
from utils.image import imread
from visualize.ui.grids import LabelGrid
from visualize.ui.images import DestinationImage, SourceImage, bgr_to_pixmap
from visualize.ui.layouts import HorizontalLayout
from visualize.viewer.pages.page import Page
from visualize.viewer.pipeline import BONE_FILENAME, IMAGE_FILENAME, TOOTH_FILENAME


class FDIControls(HorizontalLayout):
    def _init_widgets(self):
        return [
            None,
            QPushButton('Save'),
        ]

class FDICentral(HorizontalLayout):
    def _init_widgets(self):
        return [
            SourceImage(),
            DestinationImage(),
        ]

class FDIBottom(HorizontalLayout):
    def _init_widgets(self):
        return [
            LabelGrid(),
        ]

class FDIPage(Page):
    def __init__(self, panoramic_models, working_dir=Path('working')):
        super().__init__(working_dir)
        self.panoramic_models = panoramic_models

        self.controls = FDIControls()
        self.central = FDICentral()
        self.bottom = FDIBottom()

        layout = QVBoxLayout(self)
        layout.addLayout(self.controls)
        layout.addLayout(self.central)
        layout.addLayout(self.bottom)

        self.save_button, = self.controls.widgets
        self.source_image, self.destination_image = self.central.widgets
        self.label_grid, = self.bottom.widgets

        self.save_button.clicked.connect(self._on_save)
        self.destination_image.fdi_updated.connect(self._on_fdi_update)

        self.volume = None
        self.label_editors = None

    def sizeHint(self):
        size = super().sizeHint()
        screen_width = QApplication.primaryScreen().availableGeometry().width()

        return QSize(screen_width, size.height())

    def set_patient(self, patient):
        self.patient = patient
        self.volume = None
        self.label_editors = None
        self.dirty = False

        if not patient:
            return

        patient_dir = self.working_dir / patient
        image_path = patient_dir / IMAGE_FILENAME
        tooth_path = patient_dir / TOOTH_FILENAME
        bone_path = patient_dir / BONE_FILENAME
        if not (image_path.is_file() and tooth_path.is_file() and bone_path.is_file()):
            return

        panoramic_image_path = find_panoramic_image(self.working_dir, patient)
        if panoramic_image_path is None:
            return

        image_volume = numpy.load(image_path)
        self.volume = numpy.load(tooth_path)
        bone_volume = numpy.load(bone_path)

        segmentation_volume = numpy.where(self.volume >= Label.TRUNCATED, 0, self.volume)
        segmentation_volume = segmentation_volume + (bone_volume > 0)

        jaw_labels, image_panoramic, segmentation_panoramic_mask = get_panoramic(image_volume, segmentation_volume)
        rate = 4
        image_panoramic = cv2.resize(image_panoramic, (0, 0), fx=rate, fy=rate)
        segmentation_panoramic_mask = cv2.resize(segmentation_panoramic_mask, (0, 0), fx=rate, fy=rate, interpolation=cv2.INTER_NEAREST)
        segmentation_panoramic = draw_label(image_panoramic, segmentation_panoramic_mask)

        keypoint_panoramic = imread(panoramic_image_path)
        tooth_data = predict_panoramic_data(panoramic_image_path, self.panoramic_models)
        keypoint_panoramic = draw_panoramic(keypoint_panoramic, tooth_data)

        target_height, target_width, _ = keypoint_panoramic.shape
        height, width, _ = segmentation_panoramic.shape
        padding_top = (target_height - height) // 2
        padding_left = (target_width - width) // 2
        padding_bottom = target_height - height - padding_top
        padding_right = target_width - width - padding_left
        segmentation_panoramic = cv2.copyMakeBorder(segmentation_panoramic, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT, value=0)
        segmentation_panoramic_mask = cv2.copyMakeBorder(segmentation_panoramic_mask, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT, value=0)

        pixmap = bgr_to_pixmap(segmentation_panoramic)
        self.source_image.update_image(pixmap, segmentation_panoramic_mask)

        pixmap = bgr_to_pixmap(keypoint_panoramic)
        self.destination_image.update_image(pixmap, tooth_data)

        segmentation_labels = [[], [], [], []]
        for labels in jaw_labels:
            for label in labels:
                segmentation_labels[label // 10 - 1].append(label)

        keypoint_labels = [[], [], [], []]
        for tooth in tooth_data:
            fdi = int(tooth['FDI'][0])
            keypoint_labels[fdi // 10 - 1].append([int(fdi) for fdi in tooth['FDI']])

        self.label_editors = self.label_grid.update_labels(segmentation_labels, keypoint_labels=keypoint_labels)

    def _on_fdi_update(self, label, fdi):
        if self.label_editors is None:
            return

        self.label_editors[label].setText(fdi)

    def _on_save(self):
        if self.volume is None:
            return

        data = {}
        exists = set()
        empty_labels = []
        duplicated_fdis = []
        for label, fdi_editor in self.label_editors.items():
            fdi = fdi_editor.text()
            if fdi == '-':
                data[label] = None
                continue

            if fdi == '':
                empty_labels.append(label)
            elif fdi in exists:
                duplicated_fdis.append(fdi)
            data[label] = fdi
            exists.add(fdi)

        volume_labels = numpy.unique(self.volume)
        volume_labels = volume_labels[(volume_labels > 1) & (volume_labels < Label.TRUNCATED)]
        missing_labels = [int(label) for label in volume_labels if int(label) not in data]

        messages = []
        if empty_labels:
            messages.append(f'Empty Labels: {", ".join(str(label) for label in empty_labels)}')
        if duplicated_fdis:
            messages.append(f'Duplicated FDI: {", ".join(str(fdi) for fdi in duplicated_fdis)}')
        if missing_labels:
            messages.append(f'Unassigned Labels: {", ".join(str(label) for label in missing_labels)}')

        if messages:
            QMessageBox.warning(self, 'Warning', '\n'.join(messages))
            return

        fdi_mapping = {str(label): fdi for label, fdi in data.items()}
        volume = relabel_fdi(self.volume, fdi_mapping)

        numpy.save(self.working_dir / self.patient / TOOTH_FILENAME, volume)

        self.set_patient(self.patient)
        self.dirty = True
