import os
import numpy
import skimage

from scipy import ndimage
from PySide6.QtCore import Signal, QThread
from PySide6.QtGui import QVector3D, QPixmap, QColor, QIcon
from PySide6.QtWidgets import QMainWindow, QGroupBox, QVBoxLayout, QHBoxLayout, QWidget, QApplication, QGridLayout, QLabel, QComboBox
from pyqtgraph.opengl import GLViewWidget, GLVolumeItem
from PIL import Image
from PIL.ImageQt import ImageQt

class Label:
    BACKGROUND = 0
    TOOTH = 1
    BONE = 2

class Color:
    TOOTH = (235, 223, 180, 255)
    BONE = (212, 161, 230, 75)
    REMOVED = (0, 0, 0, 255)

class Mode:
    GROUND_TRUTH = 'gt'
    IMAGE = 'image'
    PREDICT = 'predict'
    CONNECTED_COMPONENT = 'cc'
    CLEANED = 'cleaned'
    WATERSHED = 'watershed'
    REFINE = 'refine'
    POST_PROCESSING = 'pp'
    REMOVED = 'removed'
    RELABELED = 'relabeled'

    @staticmethod
    def items():
        return [
            Mode.GROUND_TRUTH,
            Mode.PREDICT,
            Mode.CONNECTED_COMPONENT,
            Mode.CLEANED,
            Mode.WATERSHED,
            Mode.REFINE,
            Mode.POST_PROCESSING,
            Mode.REMOVED,
            Mode.RELABELED
        ]

    @staticmethod
    def get_title(mode):
        return {
            Mode.GROUND_TRUTH: 'Ground Truth',
            Mode.IMAGE: 'Image',
            Mode.PREDICT: 'Predict',
            Mode.CONNECTED_COMPONENT: 'Connected Component',
            Mode.CLEANED: 'Cleaned',
            Mode.WATERSHED: 'Watershed',
            Mode.REFINE: 'Refine',
            Mode.POST_PROCESSING: 'Post Processing',
            Mode.REMOVED: 'Removed',
            Mode.RELABELED: 'Relabeled'
        }[mode]

class DataManager:
    def __init__(self, experiment_name, patient_mapping, modes, cc_label, base_output_dir='outputs'):
        self.experiment_name = experiment_name
        self.patient_mapping = {
            patient: fold
            for patient, fold in sorted(patient_mapping.items(), key=lambda x: (x[0].split('/')[0], int(x[0].split('_')[-1])))
            if Mode.REFINE not in modes or os.path.exists(os.path.join(base_output_dir, experiment_name, f'Fold_{fold}', patient, 'refine_volume.npy'))
        }
        self.patients = list(self.patient_mapping.keys())
        self.modes = modes
        self.cc_label = cc_label
        self.base_output_dir = base_output_dir

    def load_data(self, patient):
        fold = self.patient_mapping[patient]
        base_dir = os.path.join(self.base_output_dir, self.experiment_name, f'Fold_{fold}', patient)

        volumes = []
        index = 0
        for mode in self.modes:
            match mode:
                case Mode.GROUND_TRUTH:
                    volume_path = os.path.join(base_dir, 'ground_truth.npy')
                case Mode.IMAGE:
                    volume_path = os.path.join(base_dir, 'image.npy')
                case Mode.PREDICT:
                    volume_path = os.path.join(base_dir, 'volume.npy')
                case Mode.CONNECTED_COMPONENT:
                    volume_path = os.path.join(base_dir, f'cc_volume_{self.cc_label[index]}.npy')
                    index += 1
                case Mode.CLEANED:
                    volume_path = os.path.join(base_dir, 'cleaned_volume.npy')
                case Mode.WATERSHED:
                    volume_path = os.path.join(base_dir, 'watershed_volume.npy')
                case Mode.REFINE:
                    volume_path = os.path.join(base_dir, 'refine_volume.npy')
                case Mode.POST_PROCESSING:
                    volume_path = os.path.join(base_dir, 'pp_volume.npy')
                case Mode.REMOVED:
                    volume_path = os.path.join(base_dir, 'removed_volume.npy')
                case Mode.RELABELED:
                    volume_path = os.path.join(base_dir, 'relabeled_volume.npy')
            volume = numpy.load(volume_path)
            volumes.append(volume)

        return volumes

    def save_data(self, patient, volume, filename):
        fold = self.patient_mapping[patient]
        file_path = os.path.join(self.base_output_dir, self.experiment_name, f'Fold_{fold}', patient, filename)
        numpy.save(file_path, volume)

class VolumeColorizer:
    @staticmethod
    def glasbey_palette(num_colors):
        levels = numpy.linspace(0, 1, 32)
        candidates = numpy.array(numpy.meshgrid(levels, levels, levels)).reshape(3, -1).T
        candidates_lab = skimage.color.rgb2lab(candidates.reshape(-1, 1, 1, 3)).reshape(-1, 3)

        lightness = candidates_lab[:, 0]
        mask = (lightness > 50) & (lightness < 80)
        candidates = candidates[mask]
        candidates_lab = candidates_lab[mask]

        initial = numpy.array([[1, 1, 1]])
        palette_lab = skimage.color.rgb2lab(initial.reshape(-1, 1, 1, 3)).reshape(-1, 3)
        palette = []

        for _ in range(num_colors):
            distance = numpy.min(numpy.linalg.norm(candidates_lab[:,None,:] - palette_lab[None,:,:], axis=2), axis=1)
            index = numpy.argmax(distance)
            palette.append(candidates[index])
            palette_lab = numpy.vstack([palette_lab, candidates_lab[index]])

        return [
            (int(r * 255), int(g * 255), int(b * 255), 255)
            for r, g, b in palette
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
        component_count = max_label - 1 if display_bone else max_label

        palette = VolumeColorizer.glasbey_palette(component_count)

        lookup_table = numpy.zeros((max_label + 1, 4), dtype=numpy.uint8)
        start_label = 2 if display_bone else 1
        lookup_table[start_label:] = palette
        rgba = lookup_table[volume]

        label_counts = numpy.bincount(volume.ravel())

        counts = []
        for label in range(start_label, start_label + component_count):
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

        return rgba, component_count

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
            case Mode.CONNECTED_COMPONENT | Mode.CLEANED | Mode.WATERSHED | Mode.REFINE:
                return VolumeColorizer.color_components(volume)
            case Mode.POST_PROCESSING | Mode.REMOVED | Mode.RELABELED:
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

        if reset:
            self.item = GLVolumeItem(volume, smooth=True)
            self.item.translate(-width / 2, -height / 2, -depth / 2)
            self.item.rotate(90, 0, 0, 1)
            self.addItem(self.item)

            self._reset_camera()
        else:
            self.item.setData(volume)

    def _reset_camera(self):
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

class IconLabelSelector(QComboBox):
    def update_items(self, labels, count, index_offset):
        self.blockSignals(True)
        self.clear()

        palette = VolumeColorizer.glasbey_palette(count)
        size = self.font().pointSize()
        for label in labels:
            index = label + index_offset
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
    def __init__(self, TopLayout, BottomLayout):
        super().__init__()
        self.move(0, 0)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setCentralWidget(widget)

        self.top_layout = TopLayout()
        self.bottom_layout = BottomLayout()
        layout.addLayout(self.top_layout)
        layout.addLayout(self.bottom_layout)
