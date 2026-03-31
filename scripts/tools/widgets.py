import os
import numpy
import colorcet
import json

from collections import defaultdict
from scipy import ndimage
from PySide6.QtCore import Signal, QThread, Qt
from PySide6.QtGui import QVector3D, QPixmap, QColor, QIcon
from PySide6.QtWidgets import QMainWindow, QGroupBox, QVBoxLayout, QHBoxLayout, QWidget, QApplication, QGridLayout, QLabel, QComboBox, QSlider
from pyqtgraph.opengl import GLViewWidget, GLVolumeItem, GLLinePlotItem
from PIL import Image
from PIL.ImageQt import ImageQt

class Label:
    BACKGROUND = 0
    TOOTH = 1
    BONE = 2
    CROPPED = 50
    UNERUPTED = 70
    RESIDUAL = 90
    FAKE = 110

class Color:
    TOOTH = (235, 223, 180, 255)
    BONE = (212, 161, 230, 75)

class Mode:
    GROUND_TRUTH = 'gt'
    IMAGE = 'image'
    PREDICT = 'predict'
    BONE_CONNECTED_COMPONENT = 'bone_cc'
    TOOTH_CONNECTED_COMPONENT = 'tooth_cc'
    CLEANED = 'cleaned'
    WATERSHED = 'watershed'
    REFINE = 'refine'
    POST_PROCESSING = 'pp'
    REMOVED = 'removed'
    BONE_FILLED = 'bone_filled'
    TOOTH_FILLED = 'tooth_filled'
    RELABELED = 'relabeled'

    @staticmethod
    def items():
        return [
            Mode.GROUND_TRUTH,
            Mode.PREDICT,
            Mode.BONE_CONNECTED_COMPONENT,
            Mode.TOOTH_CONNECTED_COMPONENT,
            Mode.CLEANED,
            Mode.WATERSHED,
            Mode.REFINE,
            Mode.POST_PROCESSING,
            Mode.REMOVED,
            Mode.BONE_FILLED,
            Mode.TOOTH_FILLED,
            Mode.RELABELED
        ]

    @staticmethod
    def get_title(mode):
        return {
            Mode.GROUND_TRUTH: 'Ground Truth',
            Mode.IMAGE: 'Image',
            Mode.PREDICT: 'Predict',
            Mode.BONE_CONNECTED_COMPONENT: 'Connected Component (Bone)',
            Mode.TOOTH_CONNECTED_COMPONENT: 'Connected Component (Tooth)',
            Mode.CLEANED: 'Cleaned',
            Mode.WATERSHED: 'Watershed',
            Mode.REFINE: 'Refine',
            Mode.POST_PROCESSING: 'Post Processing',
            Mode.REMOVED: 'Removed',
            Mode.BONE_FILLED: 'Bone Filled',
            Mode.TOOTH_FILLED: 'Tooth Filled',
            Mode.RELABELED: 'Relabeled'
        }[mode]

class DataManager:
    def __init__(self, experiment_name, patient_mapping, modes, base_output_dir='outputs'):
        self.experiment_name = experiment_name
        self.patient_mapping = {
            patient: fold
            for patient, fold in sorted(patient_mapping.items(), key=lambda x: (x[0].split('/')[0], int(x[0].split('_')[-1])))
        }
        self.patients = list(self.patient_mapping.keys())
        self.modes = modes
        self.base_output_dir = base_output_dir

    def load_data(self, patient):
        fold = self.patient_mapping[patient]
        base_dir = os.path.join(self.base_output_dir, self.experiment_name, f'Fold_{fold}', patient)

        volumes = [
            numpy.load(os.path.join(base_dir, f'{mode}.npy'))
            for mode in self.modes
        ]
        return volumes

    def save_data(self, patient, volume, filename):
        fold = self.patient_mapping[patient]
        file_path = os.path.join(self.base_output_dir, self.experiment_name, f'Fold_{fold}', patient, filename)
        numpy.save(file_path, volume)

class VolumeColorizer:
    @staticmethod
    def glasbey_palette(num_colors):
        colors = colorcet.glasbey_light[:num_colors]
        return [
            (int(color[1:3], 16), int(color[3:5], 16), int(color[5:], 16), 255)
            for color in colors
        ]

    @staticmethod
    def color_volume(volume, display_bone=False):
        rgba = numpy.zeros((*volume.shape, 4), dtype=numpy.uint8)

        tooth_mask = volume == Label.TOOTH
        rgba[tooth_mask] = Color.TOOTH

        if display_bone:
            bone_mask = volume == Label.BONE
            erosion = ndimage.binary_erosion(bone_mask)
            bone_surface = bone_mask & (~erosion)
            rgba[bone_surface] = Color.BONE

        return rgba

    @staticmethod
    def color_components(volume, display_bone=False):
        max_label = volume.max()
        print(max_label)

        palette = VolumeColorizer.glasbey_palette(max_label)

        lookup_table = numpy.zeros((max_label + 1, 4), dtype=numpy.uint8)
        lookup_table[1:] = palette[:max_label]
        lookup_table[Label.CROPPED:] = 0
        if display_bone:
            lookup_table[1] = 0
        rgba = lookup_table[volume]

        label_counts = numpy.bincount(volume.ravel())

        counts = []
        for label in range(1, max_label + 1):
            counts.append((label_counts[label], lookup_table[label]))
        counts.sort(key=lambda x: x[0])

        print('─' * 72)
        for i in range(0, len(counts), 9):
            print('\t'.join(
                f'\033[38;2;{r};{g};{b}m{count}\033[0m'
                for count, (r, g, b, _) in counts[i:i+9]
            ))
        print('─' * 72)

        if display_bone:
            bone_mask = volume == 1
            erosion = ndimage.binary_erosion(bone_mask)
            bone_surface = bone_mask & (~erosion)
            rgba[bone_surface] = Color.BONE

        return rgba, max_label

class VolumeLoaderThread(QThread):
    finished = Signal(object) # {mode: rgba_volume, "origin": {mode: volume}}
    def __init__(self, data_manager, patient, colorize=True, keep_origin=False):
        super().__init__()
        self.data_manager = data_manager
        self.patient = patient
        self.colorize = colorize
        self.keep_origin = keep_origin

    def run(self):
        modes = self.data_manager.modes
        volumes = self.data_manager.load_data(self.patient)
        results = {}
        if self.keep_origin:
            results['origin'] = {
                mode: volume
                for mode, volume in zip(modes, volumes)
            }
        if self.colorize:
            for mode, volume in zip(modes, volumes):
                results[mode] = self._make_rgba(volume, mode)

        self.finished.emit(results)

    def _make_rgba(self, volume, mode):
        match mode:
            case Mode.GROUND_TRUTH | Mode.PREDICT:
                return VolumeColorizer.color_volume(volume, display_bone=True), None
            case Mode.TOOTH_CONNECTED_COMPONENT | Mode.WATERSHED | Mode.CLEANED | Mode.REFINE | Mode.REMOVED | Mode.TOOTH_FILLED | Mode.RELABELED:
                return VolumeColorizer.color_components(volume)
            case Mode.BONE_CONNECTED_COMPONENT | Mode.BONE_FILLED | Mode.POST_PROCESSING:
                return VolumeColorizer.color_components(volume, display_bone=True)
            case _:
                return None, None

class SyncGLView(GLViewWidget):
    viewChanged = Signal(object)
    def __init__(self, size=(640, 640), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setBackgroundColor('white')
        self.setFixedSize(*size)

        self.item = None
        self.volume_shape = None

    def update_volume(self, volume, reset=True):
        self.volume_shape = volume.shape[:3]
        width, height, depth = self.volume_shape

        if reset and self.item is not None:
            self.removeItem(self.item)

        if reset or self.item is None:
            self.item = GLVolumeItem(volume, smooth=True)
            self.item.translate(-width / 2, -height / 2, -depth / 2)
            self.item.rotate(90, 0, 0, 1)
            self.addItem(self.item)

            self._reset_camera()
        else:
            self.item.setData(volume)

    def _reset_camera(self):
        if self.volume_shape is None:
            return

        width, height, depth = self.volume_shape
        fov = self.opts['fov']
        self.setCameraPosition(
            pos=QVector3D(0, 0, 0),
            distance=height / 2 + max(width, depth) / 2 / numpy.tan(numpy.radians(fov / 2)),
            elevation=0,
            azimuth=0
        )

    def _sync_change(self):
        self.viewChanged.emit(self.opts)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self._sync_change()

    def wheelEvent(self, event):
        super().wheelEvent(event)
        self._sync_change()

    def contextMenuEvent(self, event):
        self._reset_camera()
        self._sync_change()

    def apply_opts(self, options):
        for key in ['azimuth', 'elevation', 'distance', 'center']:
            self.opts[key] = options[key]
        self.update()

class SyncGLViewBox(QGroupBox):
    def __init__(self, size=(640, 640), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.view = SyncGLView(size)
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)

class AxisItem(GLViewWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.move(0, 0)
        self.setFixedSize(128, 128)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setBackgroundColor('white')
        self.opts['distance'] = 2

        self.item = GLLinePlotItem(mode='lines')
        self.item.rotate(90, 0, 0, 1)
        self.item.setGLOptions('translucent')
        self.addItem(self.item)

    def update_axes(self, axis_x, axis_y, axis_z):
        points = numpy.array([
            [0, 0, 0], axis_x,
            [0, 0, 0], axis_y,
            [0, 0, 0], axis_z
        ])
        colors = numpy.array([
            [1, 0, 0, 1], [1, 0, 0, 1],
            [0, 1, 0, 1], [0, 1, 0, 1],
            [0, 0, 1, 1], [0, 0, 1, 1]
        ])
        self.item.setData(pos=points, color=colors, width=3)

    def sync_camera(self, opts):
        for key in ['elevation', 'azimuth']:
            self.opts[key] = opts[key]
        self.update()

class VolumeViewer(QWidget):
    def __init__(self, count=2, size=(640, 640), sync=True):
        super().__init__()

        layout = QHBoxLayout()
        self.views = [SyncGLViewBox(size) for _ in range(count)]
        for view in self.views:
            layout.addWidget(view)
        self.setLayout(layout)

        if sync:
            for source_view in self.views:
                for destination_view in self.views:
                    if source_view == destination_view:
                        continue
                    source_view.view.viewChanged.connect(destination_view.view.apply_opts)

    def set_titles(self, *titles):
        for view, title in zip(self.views, titles):
            view.setTitle(title)

    @staticmethod
    def display_volumes(volumes, count=None, size=(640, 640), title=None):
        app = QApplication([])

        window = VolumeViewer(count if count is not None else len(volumes), size)
        window.move(0, 0)
        if title is not None:
            window.setWindowTitle(title)
        for view, (key, volume) in zip(window.views, volumes.items()):
            view.setTitle(key)
            view.view.update_volume(volume)

        window.show()

        app.exec()

def get_patient_fold_mapping(config, base_output_dir='outputs'):
    return {
        f'{dataset}/{patient}': fold
        for fold in range(1, config.num_folds + 1)
        for dataset in config.datasets
        for patient in os.listdir(os.path.join(base_output_dir, config.experiment, f'Fold_{fold}', dataset))
    }

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
    def __init__(self, rows=4, columns=2):
        super().__init__()

        self.rows = rows
        self.columns = columns

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        titles = [
            ['0 (B)', '315 (DB)'],
            ['45 (MB)', '270 (D)'],
            ['90 (M)', '225 (DL)'],
            ['135 (ML)', '180 (L)']
        ]
        for row in range(rows):
            for column in range(columns):
                title = titles[row][column]
                self.layout.addWidget(ImageBox(title), row, column)

        self.setFixedSize(self.sizeHint())
    def update_images(self, slices):
        positions = [
            [0, 0], [1, 0], [2, 0], [3, 0],
            [3, 1], [2, 1], [1, 1], [0, 1]
        ]
        for (row, column), image in zip(positions, slices):
            self.layout.itemAtPosition(row, column).widget().update_image(image)

class IconLabelSelector(QComboBox):
    def update_items(self, labels, count, index_offset):
        self.blockSignals(True)
        self.clear()

        palette = VolumeColorizer.glasbey_palette(count)
        size = self.font().pointSize()
        for label in labels:
            index = label.astype(numpy.int32) + index_offset
            r, g, b, _ = palette[index]
            color = QColor(r, g, b)

            pixmap = QPixmap(size, size)
            pixmap.fill(color)
            icon = QIcon(pixmap)

            self.addItem(icon, f'Label {label}', label)

        self.setCurrentIndex(0)
        self.blockSignals(False)

    def current_label(self):
        return self.currentData()

class VolumeLoader:
    def __init__(self, data_manager, handle_volumes, **kwargs):
        self.data_manager = data_manager
        self.handle_volumes = handle_volumes
        self.loader_kwargs = kwargs
        self._loading_widgets = None
    def setup(self, *loading_widgets):
        self._loading_widgets = loading_widgets
    def load_patient(self, index):
        self._set_loading(True)
        patient = self.data_manager.patients[index]
        self.loader = VolumeLoaderThread(self.data_manager, patient, **self.loader_kwargs)
        self.loader.finished.connect(self._on_volume_loaded)
        self.loader.finished.connect(self.loader.deleteLater)
        self.loader.start()
    def _on_volume_loaded(self, volumes):
        self.handle_volumes(volumes)
        self._set_loading(False)
    def _set_loading(self, loading):
        if self._loading_widgets is None:
            return

        for widget in self._loading_widgets:
            widget.setEnabled(not loading)

class PatientSelector(QComboBox):
    def setup(self, patients, current_index_changed):
        self.addItems(patients)
        self.setCurrentIndex(-1)
        self.currentIndexChanged.connect(current_index_changed)

class MainWindowUI(QMainWindow):
    def __init__(self, TopLayout, BottomLayout, top_kwargs={}, bottom_kwargs={}):
        super().__init__()
        self.move(0, 0)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setCentralWidget(widget)

        self.top_layout = TopLayout(**top_kwargs)
        self.bottom_layout = BottomLayout(**bottom_kwargs)
        layout.addLayout(self.top_layout)
        layout.addLayout(self.bottom_layout)

class Slider(QWidget):
    valueChanged = Signal(int)
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.label = QLabel('1')
        self.label.setFixedWidth(30)
        self.label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setPageStep(1)
        for widget in [self.label, self.slider]:
            layout.addWidget(widget)
        self.slider.valueChanged.connect(lambda value: self.label.setText(str(value)))
        self.slider.sliderReleased.connect(lambda: self.valueChanged.emit(self.get_value()))

        self.dragging = False
    def set_value(self, value):
        self.slider.setValue(value)
    def get_value(self):
        return self.slider.value()
    def set_maximum(self, value):
        self.slider.setMaximum(value)
    def get_maximum(self):
        return self.slider.maximum()

class DataHandler:
    def __init__(self, data=None):
        if data is None:
            data = {}
        self.data = data
    def find(self, *args, **kwargs):
        raise NotImplementedError
    def set_data(self, *args, **kwargs):
        raise NotImplementedError
    def remove_data(self, *args, **kwargs):
        raise NotImplementedError

class JsonHandler:
    def __init__(self, path, DataHandler):
        self.path = path
        data = self.load(path)
        self.handler = DataHandler(data)

    def __enter__(self):
        return self.handler

    def __exit__(self, exc_type, exc, tb):
        self.save(self.path, self.handler.data)

    @staticmethod
    def load(path):
        if not os.path.exists(path):
            return defaultdict(dict)

        with open(path) as file:
            data = json.load(file)
            return defaultdict(dict, data)

    @staticmethod
    def save(path, data):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        data = dict(data)
        with open(path, 'w') as file:
            json.dump(data, file, indent=4)
