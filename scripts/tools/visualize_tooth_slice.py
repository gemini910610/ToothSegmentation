from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QGridLayout, QLabel, QGroupBox
from PySide6.QtGui import QColor, QPixmap, QIcon
from scripts.tools.visualize import VolumeViewer, Mode, VolumeColorizer
from scripts.post_processing.single_tooth import get_slices, crop_single_tooth
from PIL import Image
from PIL.ImageQt import ImageQt

class ImageBox(QGroupBox):
    def __init__(self, title, size=(128, 128)):
        super().__init__(title)

        self.size = size

        layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setFixedSize(*size)
        layout.addWidget(self.label)
        self.setLayout(layout)
    def update_image(self, image):
        image = Image.fromarray(image)
        image = image.resize(self.size, Image.Resampling.NEAREST)
        image = ImageQt(image)
        pixmap = QPixmap.fromImage(image)
        self.label.setPixmap(pixmap)

class ImageTable(QWidget):
    def __init__(self, rows=4, columns=3):
        super().__init__()

        self.rows = rows
        self.columns = columns

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        for row in range(rows):
            for column in range(columns):
                title = str((row * columns + column) * 30)
                self.layout.addWidget(ImageBox(title), row, column)

        self.setFixedSize(self.sizeHint())
    def update_images(self, images):
        for i, image in enumerate(images):
            row = i // self.columns
            column = i % self.columns
            self.layout.itemAtPosition(row, column).widget().update_image(image)

class MainWindowUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.move(0, 0)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setCentralWidget(widget)

        top_layout = QHBoxLayout()
        self.patient_selector = QComboBox(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.label_selector = QComboBox(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        for widget in [self.patient_selector, self.label_selector]:
            top_layout.addWidget(widget)
        top_layout.addStretch()
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

        self.volume = None

    def _load_patient(self, index):
        self.patient_selector.setEnabled(False)
        self.label_selector.setEnabled(False)

        patient = self.data_manager.patients[index]
        self.volume = self.data_manager.load_data(patient)[0]
        self._on_volume_loaded()

    def _on_volume_loaded(self):
        self.label_selector.blockSignals(True)
        self.label_selector.clear()
        tooth_count = self.volume.max() - 1
        palette = VolumeColorizer.glasbey_palette(tooth_count)
        size = self.label_selector.font().pointSize()
        for index in range(tooth_count):
            r, g, b, _ = palette[index]
            color = QColor(r, g, b)

            pixmap = QPixmap(size, size)
            pixmap.fill(color)
            icon = QIcon(pixmap)

            self.label_selector.addItem(icon, f'Label {index + 1}')
        self.label_selector.setCurrentIndex(0)
        self.label_selector.blockSignals(False)
        self._on_label_changed(0)

        self.patient_selector.setEnabled(True)
        self.label_selector.setEnabled(True)

    def _on_label_changed(self, index):
        if self.volume is None:
            return

        origin_volume, center = crop_single_tooth(self.volume, tooth_label=index + 2, bone_label=1)
        volume = VolumeColorizer.color_volume(origin_volume, display_bone=True)
        self.volume_viewer.views[0].setTitle(f'Label {index + 1}')
        self.volume_viewer.views[0].view.update_volume(volume)

        slices = get_slices(origin_volume, center)
        self.image_table.update_images(slices)

if __name__ == '__main__':
    import os

    from argparse import ArgumentParser
    from PySide6.QtWidgets import QApplication
    from src.config import load_config
    from scripts.tools.visualize import get_patient_fold_mapping, DataManager

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    patient_fold_map = get_patient_fold_mapping(config)

    app = QApplication([])

    data_manager = DataManager(experiment_name, patient_fold_map, [Mode.POST_PROCESSING], [])
    window = MainWindow(data_manager)

    window.show()

    app.exec()
