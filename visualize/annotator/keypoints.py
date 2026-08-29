import shutil
from pathlib import Path

import cv2
import numpy
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut

from utils.image import imread
from visualize.ui.grids import ImageCell, PointImageGrid
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.selectors import LabelSelector, PatientSelector
from visualize.ui.windows import BaseWindow
from visualize.utils import DataHandler, JsonHandler


class KeypointDataHandler(DataHandler):
    def find(self, patient, label, degree):
        if patient not in self.data:
            return None

        if label not in self.data[patient]:
            return None

        if degree not in self.data[patient][label]:
            return None

        return self.data[patient][label][degree]

    def set_data(self, patient, label, degree, left_point, right_point):
        if patient not in self.data:
            self.data[patient] = {}

        if label not in self.data[patient]:
            self.data[patient][label] = {}

        self.data[patient][label][degree] = [left_point, right_point]

    def remove_data(self, patient, label, degree):
        points = self.find(patient, label, degree)
        if points is None:
            return

        del self.data[patient][label][degree]

        if not self.data[patient][label]:
            del self.data[patient][label]

        if not self.data[patient]:
            del self.data[patient]

class TopLayout(HorizontalLayout):
    def _init_widgets(self):
        return [
            PatientSelector(),
            LabelSelector(),
            None
        ]

class BottomLayout(HorizontalLayout):
    def _init_widgets(self):
        return [
            PointImageGrid([
                ImageCell('0 (B)', 0, 0),
                ImageCell('45 (MB)', 0, 1),
                ImageCell('90 (M)', 0, 2),
                ImageCell('135 (ML)', 0, 3)
            ], (384, 384), clickable=True)
        ]

class MainWindow(BaseWindow):
    def __init__(self, experiment_name, slice_dir, annotation_path, patients):
        super().__init__(TopLayout, BottomLayout)
        self.slice_dir = slice_dir
        self.annotation_path = annotation_path

        self.setWindowTitle(experiment_name)

        self.patient_selector, self.label_selector = self.layouts[0].widgets
        self.image_grid, = self.layouts[1].widgets

        self.patient_selector.setup(patients, self._on_patient_changed)

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)

        shortcut_save = QShortcut(QKeySequence('Ctrl+S'), self)
        shortcut_save.activated.connect(self._on_save)
        shortcut_left = QShortcut(Qt.Key.Key_Left, self)
        shortcut_left.activated.connect(lambda: self._on_label_step(-1))
        shortcut_right = QShortcut(Qt.Key.Key_Right, self)
        shortcut_right.activated.connect(lambda: self._on_label_step(1))

    def _on_patient_changed(self, index):
        patient = self.patient_selector.currentText()
        labels = sorted((path.name for path in (self.slice_dir / patient).iterdir()), key=int)

        self.label_selector.update_items(labels)
        self.label_selector.setCurrentIndex(0)

    def _on_label_changed(self, index):
        patient = self.patient_selector.currentText()
        label = self.label_selector.currentData()
        image_dir = self.slice_dir / patient / str(label)

        slices = []
        for degree in range(0, 180, 45):
            image_path = image_dir / f'image_{degree}.png'
            image = imread(image_path)
            segmentation_path = image_dir / f'segmentation_{degree}.png'
            mask = imread(segmentation_path)

            foreground = image[mask == 127]
            foreground = foreground[foreground > 0]
            low, high = numpy.quantile(foreground, [0.05, 0.95])
            image = image.clip(low, high)
            image = (image - low) / (high - low)
            image = (image * 255).astype(numpy.uint8)
            slices.append(image)
        self.image_grid.update_images(slices)

        with JsonHandler(self.annotation_path, KeypointDataHandler) as handler:
            for image, degree in zip(self.image_grid.images, range(0, 180, 45)):
                points = handler.find(patient, str(label), str(degree))
                if points is None:
                    continue

                image.left_point = [point * size for point, size in zip(points[0], image.size)]
                image.right_point = [point * size for point, size in zip(points[1], image.size)]
                image.update()

    def _on_save(self):
        patient = self.patient_selector.currentText()
        label = self.label_selector.currentData()
        if patient == '' or label is None:
            return

        source_dir = self.slice_dir / patient / str(label)
        destination_dir = self.annotation_path.parent / patient / str(label)

        with JsonHandler(self.annotation_path, KeypointDataHandler) as handler:
            for image, degree in zip(self.image_grid.images, range(0, 180, 45)):
                filenames = [f'image_{degree}.png', f'segmentation_{degree}.png']
                left_point = image.left_point
                right_point = image.right_point
                if left_point is None or right_point is None:
                    handler.remove_data(patient, str(label), str(degree))
                    for filename in filenames:
                        (destination_dir / filename).unlink(missing_ok=True)
                    continue

                left_point = [point / size for point, size in zip(left_point, image.size)]
                right_point = [point / size for point, size in zip(right_point, image.size)]
                handler.set_data(patient, str(label), str(degree), left_point, right_point)

                destination_dir.mkdir(parents=True, exist_ok=True)
                for filename in filenames:
                    shutil.copy2(source_dir / filename, destination_dir / filename)

    def _on_label_step(self, step):
        self._on_save()

        count = self.label_selector.count()
        index = self.label_selector.currentIndex()
        if index + step < 0 or index + step >= count:
            print('\a', end='', flush=True)
            return

        self.label_selector.setCurrentIndex(index + step)

if __name__ == '__main__':
    import sys

    from post_processing.utils import iterate_datasets
    from tui.apps import BaseApp
    from visualize.utils import run_app

    args = BaseApp().run()
    if args is None:
        sys.exit()

    segmentation_experiment_name = args['exp']
    datasets = args['datasets']

    slice_dir = Path('output_slices') / segmentation_experiment_name

    patients = [
        f'{dataset}/{patient}'
        for dataset, patient in iterate_datasets(segmentation_experiment_name, datasets)
        if (slice_dir / dataset / patient).is_dir()
    ]

    run_app(MainWindow, segmentation_experiment_name, slice_dir, Path('datasets') / 'slices' / 'keypoints.json', patients)
