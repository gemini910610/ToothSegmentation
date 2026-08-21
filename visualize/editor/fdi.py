from pathlib import Path

import cv2
import numpy
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

from constants.label import Label
from constants.mode import Mode
from post_processing.panoramic import (
    draw_label,
    draw_panoramic,
    find_panoramic_paths,
    get_panoramic,
    load_panoramic,
)
from visualize.ui.grids import LabelGrid
from visualize.ui.images import DestinationImage, SourceImage, bgr_to_pixmap
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.selectors import PatientSelector
from visualize.ui.windows import BaseWindow
from visualize.utils import DataHandler, JsonHandler


class FDIDataHandler(DataHandler):
    def set_data(self, patient, data):
        self.data[patient] = data

class TopLayout(HorizontalLayout):
    def _init_widgets(self):
        return [
            PatientSelector(),
            None
        ]

class CentralLayout(HorizontalLayout):
    def _init_widgets(self):
        return [
            SourceImage(),
            DestinationImage()
        ]

class BottomLayout(HorizontalLayout):
    def _init_widgets(self):
        return [
            LabelGrid(),
            None,
            QPushButton('Save')
        ]

class MainWindow(BaseWindow):
    def __init__(self, data_manager, panoramic_paths):
        super().__init__(TopLayout, CentralLayout, BottomLayout)
        self.data_manager = data_manager
        self.panoramic_paths = panoramic_paths

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector, = self.layouts[0].widgets
        self.source_image, self.destination_image = self.layouts[1].widgets
        self.label_grid, self.save_button = self.layouts[2].widgets

        self.patient_selector.setup(data_manager.patients, self._on_patient_changed)

        self.destination_image.fdi_updated.connect(self._on_fdi_update)
        self.save_button.clicked.connect(self._on_save)

        self.label_editors = None

    def showEvent(self, event):
        width = QApplication.primaryScreen().availableGeometry().width()
        self.setFixedWidth(width)
        self.centralWidget().layout().activate()
        self.adjustSize()

    def _on_patient_changed(self, index):
        patient_key = self.data_manager.patients[index]

        volumes = self.data_manager.load_data(index)
        image_volume = volumes[Mode.IMAGE]
        relabeled_volume = volumes[Mode.RELABELED]
        bone_volume = volumes[Mode.BONE_FILLED]

        segmentation_volume = numpy.where(relabeled_volume >= Label.TRUNCATED, 0, relabeled_volume)
        segmentation_volume = segmentation_volume + (bone_volume > 0)

        jaw_labels, image_panoramic, segmentation_panoramic_mask = get_panoramic(image_volume, segmentation_volume)
        rate = 4
        image_panoramic = cv2.resize(image_panoramic, (0, 0), fx=rate, fy=rate)
        segmentation_panoramic_mask = cv2.resize(segmentation_panoramic_mask, (0, 0), fx=rate, fy=rate, interpolation=cv2.INTER_NEAREST)
        segmentation_panoramic = draw_label(image_panoramic, segmentation_panoramic_mask)

        image_path, prediction_path = self.panoramic_paths[patient_key]
        keypoint_panoramic, tooth_data = load_panoramic(image_path, prediction_path)
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
                segmentation_labels[label//10-1].append(label)

        with JsonHandler(Path('fdi') / f'{self.data_manager.experiment_name}.json', FDIDataHandler) as handler:
            data = handler.data.get(patient_key)

        if data is not None:
            self.label_editors = self.label_grid.update_labels(segmentation_labels, saved_data=data)
        else:
            keypoint_labels = [[], [], [], []]
            for tooth in tooth_data:
                fdi = int(tooth['FDI'][0])
                keypoint_labels[fdi//10-1].append([int(fdi) for fdi in tooth['FDI']])

            self.label_editors = self.label_grid.update_labels(segmentation_labels, keypoint_labels=keypoint_labels)

    def _on_fdi_update(self, label, fdi):
        if self.label_editors is None:
            return

        self.label_editors[label].setText(fdi)

    def _on_save(self):
        if self.label_editors is None:
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

        messages = []
        if empty_labels:
            messages.append(f'Empty Labels: {", ".join([str(label) for label in empty_labels])}')
        if duplicated_fdis:
            messages.append(f'Duplicated FDI: {", ".join([str(fdi) for fdi in duplicated_fdis])}')

        if messages:
            QMessageBox.warning(self, 'Warning', '\n'.join(messages))
            return

        patient = self.patient_selector.currentText()

        with JsonHandler(Path('fdi') / f'{self.data_manager.experiment_name}.json', FDIDataHandler) as handler:
            handler.set_data(patient, data)

if __name__ == '__main__':
    import sys

    from post_processing.utils import iterate_datasets
    from tui.apps import BaseApp
    from visualize.utils import DataManager, run_app

    args = BaseApp().run()
    if args is None:
        sys.exit()

    segmentation_experiment_name = args['exp']
    datasets = args['datasets']

    panoramic_paths = {}
    for dataset, patient in iterate_datasets(segmentation_experiment_name, datasets):
        paths = find_panoramic_paths(dataset, patient)
        if paths is not None:
            panoramic_paths[f'{dataset}/{patient}'] = paths

    patients = list(panoramic_paths.keys())
    data_manager = DataManager(segmentation_experiment_name, patients, [Mode.IMAGE, Mode.RELABELED, Mode.BONE_FILLED])
    run_app(MainWindow, data_manager, panoramic_paths)
