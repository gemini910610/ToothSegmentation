import os
import cv2
import numpy

from PySide6.QtGui import QShortcut, QKeySequence, Qt
from PySide6.QtWidgets import QHBoxLayout, QComboBox
from .widgets import PatientSelector, LabelSelector, ImageTable, MainWindowUI, ImageCell, DataHandler, JsonHandler

class TopLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.patient_selector = PatientSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.label_selector = LabelSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        for width in [self.patient_selector, self.label_selector]:
            self.addWidget(width)
        self.addStretch()
    def get_widgets(self):
        return self.patient_selector, self.label_selector

class BottomLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.image_table = ImageTable([
            ImageCell('0 (B)', 0, 0),
            ImageCell('45 (MB)', 0, 1),
            ImageCell('90 (M)', 0, 2),
            ImageCell('135 (ML)', 0, 3)
        ], (384, 384), clickable=True)
        self.addWidget(self.image_table)
    def get_widgets(self):
        return self.image_table

class MainWindow(MainWindowUI):
    def __init__(self, experiment_name, slice_folder):
        super().__init__(TopLayout, BottomLayout)

        self.slice_folder = slice_folder
        self.patients = [
            f'{dataset}/{patient}'
            for dataset in os.listdir(slice_folder)
            if os.path.isdir(os.path.join(slice_folder, dataset))
            for patient in sorted(os.listdir(os.path.join(slice_folder, dataset)), key=lambda x: int(x[5:]))
        ]

        self.setWindowTitle(experiment_name)

        self.patient_selector, self.label_selector = self.top_layout.get_widgets()
        self.image_table = self.bottom_layout.get_widgets()

        self.patient_selector.setup(self.patients, self.load_patient)

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)

        shortcut_save = QShortcut(QKeySequence('Ctrl+S'), self)
        shortcut_save.activated.connect(self._on_save)
        shortcut_left = QShortcut(Qt.Key.Key_Left, self)
        shortcut_left.activated.connect(lambda: self._on_label_step(-1))
        shortcut_right = QShortcut(Qt.Key.Key_Right, self)
        shortcut_right.activated.connect(lambda: self._on_label_step(1))
    def load_patient(self, index):
        patient = self.patient_selector.currentText()
        labels = os.listdir(os.path.join(self.slice_folder, patient))
        labels.sort(key=int)

        self.label_selector.update_items(labels)
        self._on_label_changed(0)
    def _on_label_changed(self, index):
        patient = self.patient_selector.currentText()
        label = self.label_selector.current_label()
        image_folder = os.path.join(self.slice_folder, patient, label)
        slices = []
        for degree in range(0, 180, 45):
            image = cv2.imread(os.path.join(image_folder, f'image_{degree}.png'))
            low = image.min()
            high = image.max()
            image = (image - low) / (high - low)
            image = (image * 255).astype(numpy.uint8)
            slices.append(image)
        self.image_table.update_images(slices)

        with JsonHandler(f'{self.slice_folder}/points.json', KeypointHandler) as handler:
            for i, box in enumerate(self.image_table.boxes):
                degree = i * 45
                points = handler.find(patient, label, str(degree))
                if points is None:
                    continue
                box.image.left_point = [point * size for point, size in zip(points[0], box.image.size)]
                box.image.right_point = [point * size for point, size in zip(points[1], box.image.size)]
                box.image.update()
    def _on_save(self):
        patient = self.patient_selector.currentText()
        label = self.label_selector.current_label()
        with JsonHandler(f'{self.slice_folder}/points.json', KeypointHandler) as handler:
            for i, box in enumerate(self.image_table.boxes):
                degree = i * 45
                left_point = box.image.left_point
                right_point = box.image.right_point
                if left_point is None or right_point is None:
                    handler.remove_data(patient, label, str(degree))
                    continue

                left_point = [point / size for point, size in zip(left_point, box.image.size)]
                right_point = [point / size for point, size in zip(right_point, box.image.size)]
                handler.set_data(patient, label, str(degree), left_point, right_point)
    def _on_label_step(self, step):
        self._on_save()

        count = self.label_selector.count()
        index = self.label_selector.currentIndex()
        if index + step < 0 or index + step >= count:
            print('\a', end='', flush=True)
            return

        index = max(0, min(count - 1, index + step))
        self.label_selector.setCurrentIndex(index)

class KeypointHandler(DataHandler):
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

        if len(self.data[patient][label]) == 0:
            del self.data[patient][label]

        if len(self.data[patient]) == 0:
            del self.data[patient]

if __name__ == '__main__':
    from argparse import ArgumentParser
    from PySide6.QtWidgets import QApplication

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('slice_folder', type=str)
    args = parser.parse_args()

    experiment_name = args.exp
    slice_folder = args.slice_folder

    app = QApplication([])

    window = MainWindow(experiment_name, slice_folder)

    window.show()

    app.exec()
