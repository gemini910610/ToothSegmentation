import os
import numpy
import cv2

from scripts.tools.widgets import PatientSelector, IconLabelSelector, ImageTable, ImageCell, MainWindowUI
from PySide6.QtCore import Qt
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import QHBoxLayout, QComboBox, QCheckBox, QButtonGroup, QRadioButton
from keypoint.predict import CEJFinder

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
        self.image_table = ImageTable([
            ImageCell('0 (B)', 0, 0),
            ImageCell('45 (MB)', 0, 1),
            ImageCell('90 (M)', 0, 2),
            ImageCell('135 (ML)', 0, 3),
            ImageCell('180 (L)', 1, 3),
            ImageCell('225 (DL)', 1, 2),
            ImageCell('270 (D)', 1, 1),
            ImageCell('315 (DB)', 1, 0),
        ], size=(224, 224))
        self.addWidget(self.image_table)
    def get_widgets(self):
        return self.image_table

class MainWindow(MainWindowUI):
    def __init__(self):
        super().__init__(TopLayout, BottomLayout)
        self.cej_finder = CEJFinder('logs/keypoint_baseline/best.pth')

        self.setWindowTitle('keypoint_baseline')

        self.patient_selector, self.label_selector, self.point_toggle, self.slice_selector = self.top_layout.get_widgets()
        self.image_table = self.bottom_layout.get_widgets()

        for dataset in os.listdir('datasets/slices'):
            if not os.path.isdir(f'datasets/slices/{dataset}'):
                continue
            for patient in sorted(os.listdir(f'datasets/slices/{dataset}'), key=lambda x: int(x[5:])):
                self.patient_selector.addItem(f'{dataset}/{patient}')
        self.patient_selector.setCurrentIndex(-1)

        self.point_toggle.setChecked(True)
        self.slice_selector.buttons()[0].setChecked(True)

        self.patient_selector.currentIndexChanged.connect(self._on_patient_changed)
        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.point_toggle.stateChanged.connect(lambda x: self._on_slice_changed(self.slice_selector.checkedId()))
        self.slice_selector.idClicked.connect(self._on_slice_changed)

        shortcut_up = QShortcut(Qt.Key.Key_Up, self)
        shortcut_up.activated.connect(lambda: self._on_patient_step(-1))
        shortcut_down = QShortcut(Qt.Key.Key_Down, self)
        shortcut_down.activated.connect(lambda: self._on_patient_step(1))
        shortcut_left = QShortcut(Qt.Key.Key_Left, self)
        shortcut_left.activated.connect(lambda: self._on_label_step(-1))
        shortcut_right = QShortcut(Qt.Key.Key_Right, self)
        shortcut_right.activated.connect(lambda: self._on_label_step(1))

        self.slices = None
        self.points = None

    def _on_patient_changed(self, index):
        patient = self.patient_selector.currentText()
        labels = os.listdir(f'datasets/slices/{patient}')
        labels = [int(label) for label in labels]
        labels.sort()
        labels = numpy.array(labels)

        tooth_count = int(labels[-1]) - 1
        self.label_selector.update_items(labels, tooth_count, -2)
        self._on_label_changed(0)

    def _on_label_changed(self, index):
        patient = self.patient_selector.currentText()
        label = self.label_selector.current_label()

        origin_segmentation_slices = []
        segmentation_slices = []
        image_slices = []
        tooth_slices = []
        for degree in range(0, 360, 45):
            segmentation_slice = cv2.imread(f'datasets/slices/{patient}/{label}/segmentation_{degree}.png', cv2.IMREAD_GRAYSCALE)
            image_slice = cv2.imread(f'datasets/slices/{patient}/{label}/image_{degree}.png', cv2.IMREAD_GRAYSCALE)
            tooth_slice = cv2.imread(f'datasets/slices/{patient}/{label}/tooth_{degree}.png', cv2.IMREAD_GRAYSCALE)

            lookup_table = numpy.zeros(256, dtype=numpy.uint8)
            lookup_table[127] = 1
            lookup_table[255] = 2
            origin_segmentation_slice = lookup_table[segmentation_slice]

            origin_segmentation_slices.append(origin_segmentation_slice)
            segmentation_slices.append(segmentation_slice)
            image_slices.append(image_slice)
            tooth_slices.append(tooth_slice)

        self.slices = [segmentation_slices, image_slices, tooth_slices]
        self.points = self.cej_finder.find(origin_segmentation_slices, tooth_slices)

        self._on_slice_changed(self.slice_selector.checkedId())

    def _on_patient_step(self, step):
        count = self.patient_selector.count()
        index = self.patient_selector.currentIndex()
        if index + step < 0 or index + step >= count:
            print('\a', end='', flush=True)
            return

        index = max(0, min(count - 1, index + step))
        self.patient_selector.setCurrentIndex(index)

    def _on_label_step(self, step):
        count = self.label_selector.count()
        index = self.label_selector.currentIndex()
        if index + step < 0 or index + step >= count:
            print('\a', end='', flush=True)
            return

        index = max(0, min(count - 1, index + step))
        self.label_selector.setCurrentIndex(index)

    def _on_slice_changed(self, index):
        if self.slices is None:
            return

        slices = self.slices[index]
        points = None if not self.point_toggle.isChecked() else self.points
        self.image_table.update_images(slices, points)

if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication

    app = QApplication([])

    window = MainWindow()

    window.show()

    app.exec()
