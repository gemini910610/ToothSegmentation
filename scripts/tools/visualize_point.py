import numpy
import cv2

from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QGridLayout, QLabel, QGroupBox, QButtonGroup, QRadioButton
from PySide6.QtGui import QColor, QPixmap, QIcon, Qt, QShortcut
from scripts.tools.visualize import VolumeViewer, Mode, VolumeColorizer, Label
from scripts.post_processing.tooth_slice import get_slices, crop_single_tooth, normalize_slice
from PIL import Image
from PIL.ImageQt import ImageQt
from sklearn.cluster import KMeans
from scipy import ndimage
from skimage.morphology import h_minima

class CEJFinder:
    def __init__(self, tooth_volume):
        mask = tooth_volume != 0
        colors = tooth_volume[mask].reshape(-1, 1)
        self.k_means = KMeans(n_clusters=3, random_state=42).fit(colors)
    def find(self, tooth_slice):
        mask = tooth_slice != 0
        colors = tooth_slice[mask].reshape(-1, 1)

        labels = self.k_means.predict(colors)
        centers = self.k_means.cluster_centers_.reshape(-1)

        order = numpy.argsort(centers)

        ys, xs = numpy.where(mask)
        label = labels == order[-1]
        y = ys[label]
        x = xs[label]

        median = numpy.median(x)
        index_left = x <= median
        index_right = x > median

        x_left = x[index_left]
        y_left = y[index_left]
        index_lb = numpy.argmax(y_left - x_left)
        left_bottom = [x_left[index_lb], y_left[index_lb]]

        x_right = x[index_right]
        y_right = y[index_right]
        index_rb = numpy.argmax(y_right + x_right)
        right_bottom = [x_right[index_rb], y_right[index_rb]]

        return left_bottom, right_bottom

# def find_cej(tooth_slice):
#     mask = tooth_slice != 0
#     colors = tooth_slice[mask].reshape(-1, 1)

#     k_means = KMeans(n_clusters=3, random_state=42)

#     labels = k_means.fit_predict(colors)
#     centers = k_means.cluster_centers_.reshape(-1)

#     order = numpy.argsort(centers)
#     label = order[-1]

#     ys, xs = numpy.where(mask)
#     y = ys[labels == label]
#     x = xs[labels == label]

#     index_lb = numpy.argmax(y - x)
#     left_bottom = [x[index_lb], y[index_lb]]

#     index_rb = numpy.argmax(y + x)
#     right_bottom = [x[index_rb], y[index_rb]]

#     return left_bottom, right_bottom

def find_bone_point(segmentation_slice):
    tooth_mask = segmentation_slice == Label.TOOTH
    bone_mask = segmentation_slice == Label.BONE
    background_mask = segmentation_slice == Label.BACKGROUND

    tooth_dilation = ndimage.binary_dilation(tooth_mask)
    bone_dilation = ndimage.binary_dilation(bone_mask)
    background_dilation = ndimage.binary_dilation(background_mask)
    bone_points = tooth_dilation & bone_dilation & background_dilation

    labels, count = ndimage.label(bone_points)

    centroids = ndimage.center_of_mass(bone_points, labels, range(1, count + 1))
    centroids = numpy.array(centroids)

    xs = centroids[:, 1]

    mean = (min(xs) + max(xs)) / 2
    left_mask = xs <= mean
    right_mask = xs > mean

    left_centroids = centroids[left_mask]
    right_centroids = centroids[right_mask]

    left_index = numpy.argmin(left_centroids[:, 0])
    right_index = numpy.argmin(right_centroids[:, 0])

    left_bone = numpy.round(left_centroids[left_index][::-1]).astype(numpy.int32)
    right_bone = numpy.ceil(right_centroids[right_index][::-1]).astype(numpy.int32)

    return left_bone, right_bone

def ensure_upward(segmentation_volume, image_volume, tooth_volume, center):
    tooth_mask = segmentation_volume == Label.TOOTH
    z = tooth_mask.any((0, 1)).argmax()
    x, y = numpy.argwhere(tooth_mask[:, :, z])[0]

    if z != 0 and segmentation_volume[x, y, z - 1] != Label.BONE:
        segmentation_volume = segmentation_volume[:,:,::-1]
        image_volume = image_volume[:,:,::-1]
        tooth_volume = tooth_volume[:,:,::-1]
        center = center.copy()
        center[2] = segmentation_volume.shape[2] - center[2] - 1

    return segmentation_volume, image_volume, tooth_volume, center

def find_root(segmentation_volume):
    tooth_volume = segmentation_volume == Label.TOOTH

    any_xy = tooth_volume.any(2) # z
    z_min_index = numpy.argmax(tooth_volume, 2).astype(numpy.int32)
    z_min_index[~any_xy] = segmentation_volume.shape[2]

    z_min = ndimage.gaussian_filter(z_min_index, sigma=1)
    min_mask = h_minima(z_min, h=3) & any_xy

    labels, count = ndimage.label(min_mask)
    min_points = []
    for label in range(1, count + 1):
        xs, ys = numpy.where(labels == label)
        index = numpy.argmin(z_min_index[xs, ys])
        x, y = int(xs[index]), int(ys[index])
        z = int(z_min_index[x, y])
        min_points.append([x, y, z])

    return min_points

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
    def update_images(self, slices):
        for i, image in enumerate(slices):
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
        self.slice_selector = QButtonGroup()
        group_layout = QHBoxLayout()
        for index, title in enumerate(['Segmentation', 'Image', 'Tooth']):
            radio = QRadioButton(title)
            self.slice_selector.addButton(radio, index)
            group_layout.addWidget(radio)
        self.slice_selector.buttons()[0].setChecked(True)
        top_layout.addLayout(group_layout)
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

        self.slice_selector.idClicked.connect(self._on_slice_changed)

        shortcut_left = QShortcut(Qt.Key.Key_Left, self)
        shortcut_left.activated.connect(lambda: self._on_label_step(-1))
        shortcut_right = QShortcut(Qt.Key.Key_Right, self)
        shortcut_right.activated.connect(lambda: self._on_label_step(1))

        self.volumes = None
        self.cej_finder = None
        self.slices = None

    def _load_patient(self, index):
        self.patient_selector.setEnabled(False)
        self.label_selector.setEnabled(False)
        for radio in self.slice_selector.buttons():
            radio.setEnabled(False)

        patient = self.data_manager.patients[index]
        self.volumes = self.data_manager.load_data(patient)
        self._on_volume_loaded()

    def _on_volume_loaded(self):
        self.label_selector.blockSignals(True)
        self.slice_selector.blockSignals(True)
        self.label_selector.clear()
        tooth_count = self.volumes[0].max() - 1
        print(tooth_count)
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
        self.slice_selector.buttons()[0].setChecked(True)
        self.label_selector.blockSignals(False)
        self.slice_selector.blockSignals(False)
        for label in range(tooth_count):
            try:
                self._on_label_changed(label)
                break
            except Exception as exception:
                print(exception)

        self.patient_selector.setEnabled(True)
        self.label_selector.setEnabled(True)
        for radio in self.slice_selector.buttons():
            radio.setEnabled(True)

    def _on_label_changed(self, index):
        if self.volumes is None:
            return

        segmentation_volume, image_volume = self.volumes
        segmentation_volume, image_volume, tooth_volume, center = crop_single_tooth(segmentation_volume, image_volume, tooth_label=index + 2, bone_label=1)
        segmentation_volume, image_volume, tooth_volume, center = ensure_upward(segmentation_volume, image_volume, tooth_volume, center)

        volume = VolumeColorizer.color_volume(segmentation_volume, display_bone=True)

        # root_points = find_root(segmentation_volume)
        # print(len(root_points), root_points)
        # palette = VolumeColorizer.glasbey_palette(len(root_points))
        # for index, (x, y, z) in enumerate(root_points):
        #     r, g, b, _ = palette[index]
        #     volume[x, y, z] = (r, g, b, 255)
        #     volume[x + 1, y, z] = (r, g, b, 255)
        #     volume[x - 1, y, z] = (r, g, b, 255)
        #     volume[x, y + 1, z] = (r, g, b, 255)
        #     volume[x, y - 1, z] = (r, g, b, 255)
        #     volume[x, y, z + 1] = (r, g, b, 255)
        #     volume[x, y, z - 1] = (r, g, b, 255)

        self.volume_viewer.views[0].setTitle(f'Label {index + 1}')
        self.volume_viewer.views[0].view.update_volume(volume)

        self.cej_finder = CEJFinder(tooth_volume)

        self.slices = list(get_slices(segmentation_volume, image_volume, tooth_volume, center))
        self._on_slice_changed(self.slice_selector.checkedId())

    def _on_slice_changed(self, index):
        if self.slices is None:
            return

        slices = []
        for segmentation_slice, image_slice, tooth_slice in self.slices:
            volume_slice = normalize_slice(segmentation_slice, image_slice, tooth_slice)[index]
            volume_slice = cv2.cvtColor(volume_slice, cv2.COLOR_GRAY2BGR)

            if index in [1, 2]:
                left_cej, right_cej = self.cej_finder.find(tooth_slice)
                volume_slice = cv2.circle(volume_slice, left_cej, 2, (0, 255, 255), -1)
                volume_slice = cv2.circle(volume_slice, right_cej, 2, (0, 255, 255), -1)

            if index in [0, 1]:
                left_bone, right_bone = find_bone_point(segmentation_slice)
                volume_slice = cv2.circle(volume_slice, left_bone, 2, (255, 0, 0), -1)
                volume_slice = cv2.circle(volume_slice, right_bone, 2, (255, 0, 0), -1)

            slices.append(volume_slice)

        self.image_table.update_images(slices)

    def _on_label_step(self, step):
        if self.volumes is None:
            return

        count = self.label_selector.count()
        index = self.label_selector.currentIndex()
        index = max(0, min(count -1, index + step))
        self.label_selector.setCurrentIndex(index)

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

    data_manager = DataManager(experiment_name, patient_fold_map, [Mode.RELABELED, Mode.IMAGE], [])
    window = MainWindow(data_manager)

    window.show()

    app.exec()
