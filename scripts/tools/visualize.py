import os
import numpy
import skimage

from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QGroupBox
from PySide6.QtCore import Signal, QThread
from PySide6.QtGui import QVector3D
from pyqtgraph.opengl import GLViewWidget, GLVolumeItem
from scipy import ndimage

class Label:
    TOOTH = 1
    BONE = 2

class Color:
    TOOTH = (235, 223, 180, 255)
    BONE = (212, 161, 230, 75)
    REMOVED = (0, 0, 0, 255)

class ViewSetting:
    VIEW_WIDTH = 640
    VIEW_HEIGHT = 640

    BACKGROUND = 'white'

class Mode:
    GROUND_TRUTH = 'gt'
    PREDICT = 'predict'
    CONNECTED_COMPONENT = 'cc'
    POST_PROCESSING = 'pp'
    WATERSHED = 'watershed'
    REFINE = 'refine'

    def items():
        return [
            Mode.GROUND_TRUTH,
            Mode.PREDICT,
            Mode.CONNECTED_COMPONENT,
            Mode.WATERSHED,
            Mode.POST_PROCESSING,
            Mode.REFINE
        ]

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
                case Mode.PREDICT:
                    volume_path = os.path.join(base_dir, 'volume.npy')
                case Mode.CONNECTED_COMPONENT:
                    volume_path = os.path.join(base_dir, f'cc_volume_{self.cc_label[index]}.npy')
                    index += 1
                case Mode.WATERSHED:
                    volume_path = os.path.join(base_dir, 'watershed_volume.npy')
                case Mode.REFINE:
                    volume_path = os.path.join(base_dir, 'refine_volume.npy')
                case Mode.POST_PROCESSING:
                    volume_path = os.path.join(base_dir, 'pp_volume.npy')
            volume = numpy.load(volume_path)
            volumes.append(volume)

        return volumes

    def save_data(self, patient, volume, filename):
        fold = self.patient_mapping[patient]
        file_path = os.path.join(self.base_output_dir, self.experiment_name, f'Fold_{fold}', patient, filename)
        numpy.save(file_path, volume)

class VolumeColorizer:
    def _glasbey_palette(self, num_colors):
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

    def color_volume(self, volume, display_bone=False):
        rgba = numpy.zeros((*volume.shape, 4), dtype=numpy.uint8)

        tooth_mask = volume == Label.TOOTH
        rgba[tooth_mask] = Color.TOOTH

        if display_bone:
            bone_mask = volume == Label.BONE
            erosion = ndimage.binary_erosion(bone_mask)
            bone_surface = bone_mask & (~erosion)
            rgba[bone_surface] = Color.BONE

        return rgba

    def color_components(self, volume, display_bone=False):
        max_label = volume.max()
        component_count = max_label - 1 if display_bone else max_label

        palette = self._glasbey_palette(component_count)

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

class VolumeLoader(QThread):
    finished = Signal(object, object, int) # left volume, right volume, tooth count
    def __init__(self, data_manager, patient, left_mode, right_mode):
        super().__init__()
        self.data_manager = data_manager
        self.patient = patient
        self.left_mode = left_mode
        self.right_mode = right_mode
        self.colorizer = VolumeColorizer()

    def run(self):
        left_volume, right_volume = self.data_manager.load_data(self.patient)
        left_volume, left_count = self._make_rgba(left_volume, self.left_mode)
        right_volume, right_count = self._make_rgba(right_volume, self.right_mode)
        if self.left_mode == Mode.CONNECTED_COMPONENT and self.right_mode == Mode.CONNECTED_COMPONENT:
            if self.data_manager.cc_label[0] == 1:
                count = left_count
            elif self.data_manager.cc_label[1] == 1:
                count = right_count
        elif self.left_mode == Mode.CONNECTED_COMPONENT and self.data_manager.cc_label[0] == 1:
            count = left_count
        elif self.right_mode == Mode.CONNECTED_COMPONENT and self.data_manager.cc_label[0] == 1:
            count = right_count
        elif self.left_mode == Mode.WATERSHED:
            count = left_count
        elif self.right_mode == Mode.WATERSHED:
            count = right_count
        else:
            count = -1
        self.finished.emit(left_volume, right_volume, count)

    def _make_rgba(self, volume, mode):
        match mode:
            case Mode.GROUND_TRUTH:
                return self.colorizer.color_volume(volume, display_bone=True), None
            case Mode.PREDICT:
                return self.colorizer.color_volume(volume, display_bone=True), None
            case Mode.CONNECTED_COMPONENT:
                return self.colorizer.color_components(volume)
            case Mode.WATERSHED:
                return self.colorizer.color_components(volume)
            case Mode.REFINE:
                return self.colorizer.color_components(volume)
            case Mode.POST_PROCESSING:
                return self.colorizer.color_components(volume, display_bone=True)

class SyncGLView(GLViewWidget):
    viewChanged = Signal(object)
    def __init__(self, size=(ViewSetting.VIEW_WIDTH, ViewSetting.VIEW_HEIGHT), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setBackgroundColor(ViewSetting.BACKGROUND)
        self.setFixedSize(*size)

        self.item = None
        self.volume_shape = None

    def update_volume(self, volume):
        self.volume_shape = volume.shape[:3]
        width, height, depth = self.volume_shape

        if self.item is not None:
            self.removeItem(self.item)

        self.item = GLVolumeItem(volume, smooth=True)
        self.item.translate(-width / 2, -height / 2, -depth / 2)
        self.item.rotate(90, 0, 0, 1)
        self.addItem(self.item)

        self._reset_camera()

    def _reset_camera(self):
        width, _, depth = self.volume_shape
        fov = self.opts['fov']
        self.setCameraPosition(
            pos=QVector3D(0, 0, 0),
            distance=max(width, depth) / 2 / numpy.tan(numpy.radians(fov / 2)),
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
    def __init__(self, size=(ViewSetting.VIEW_WIDTH, ViewSetting.VIEW_HEIGHT), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.view = SyncGLView(size)
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)

class VolumeViewer(QWidget):
    def __init__(self, count=2, size=(ViewSetting.VIEW_WIDTH, ViewSetting.VIEW_HEIGHT)):
        super().__init__()

        layout = QHBoxLayout()
        self.views = [SyncGLViewBox(size) for _ in range(count)]
        for view in self.views:
            layout.addWidget(view)
        self.setLayout(layout)

        for source_view in self.views:
            for destination_view in self.views:
                if source_view == destination_view:
                    continue
                source_view.view.viewChanged.connect(destination_view.view.apply_opts)

class MainWindowUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.move(0, 0)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setCentralWidget(widget)

        top_layout = QHBoxLayout()
        self.patient_selector = QComboBox()
        self.tooth_count_label = QLabel('Tooth Count: -')
        top_layout.addWidget(self.patient_selector)
        top_layout.addWidget(self.tooth_count_label)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        self.volume_viewer = VolumeViewer()
        layout.addWidget(self.volume_viewer)

class MainWindow(MainWindowUI):
    def __init__(self, data_manager, left_mode, right_mode):
        super().__init__()
        self.data_manager = data_manager
        self.left_mode = left_mode
        self.right_mode = right_mode

        self.setWindowTitle(self.data_manager.experiment_name)

        self.patient_selector.addItems(self.data_manager.patients)
        self.patient_selector.setCurrentIndex(-1)
        self.patient_selector.currentIndexChanged.connect(self._load_patient)

        titles = {
            Mode.GROUND_TRUTH: 'Ground Truth',
            Mode.PREDICT: 'Predict',
            Mode.CONNECTED_COMPONENT: 'Connected Component',
            Mode.POST_PROCESSING: 'Post Processing',
            Mode.WATERSHED: 'Watershed',
            Mode.REFINE: 'Refine'
        }
        self.volume_viewer.views[0].setTitle(titles[left_mode])
        self.volume_viewer.views[1].setTitle(titles[right_mode])

    def _load_patient(self, index):
        patient = self.data_manager.patients[index]
        self.patient_selector.setEnabled(False)
        self.thread = VolumeLoader(self.data_manager, patient, self.left_mode, self.right_mode)
        self.thread.finished.connect(self._on_volume_loaded)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _on_volume_loaded(self, left_volume, right_volume, tooth_count):
        self.volume_viewer.views[0].view.update_volume(left_volume)
        self.volume_viewer.views[1].view.update_volume(right_volume)
        self.tooth_count_label.setText(f'Tooth Count: {tooth_count if tooth_count > 0 else "-"}')
        self.patient_selector.setEnabled(True)

def get_patient_fold_mapping(config, base_output_dir='outputs'):
    return {
        f'{dataset}/{patient}': fold
        for fold in range(1, config.num_folds + 1)
        for dataset in config.datasets
        for patient in os.listdir(os.path.join(base_output_dir, config.experiment, f'Fold_{fold}', dataset))
    }

if __name__ == '__main__':
    from argparse import ArgumentParser
    from PySide6.QtWidgets import QApplication
    from src.config import load_config

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('--left', default=Mode.PREDICT, choices=Mode.items())
    parser.add_argument('--right', default=Mode.GROUND_TRUTH, choices=Mode.items())
    parser.add_argument('--cc-label', nargs='+', default=[1, 2], type=int)
    args = parser.parse_args()

    experiment_name = args.exp
    left_mode = args.left
    right_mode = args.right
    cc_label = args.cc_label
    if not os.path.exists(os.path.join('outputs', experiment_name)):
        raise FileNotFoundError(
            f'Output of experiment "{experiment_name}" not found.\033[0m\n'
            'Try using the following command to generating segmentation results:\n'
            f'\033[36mpython scripts/compare.py {experiment_name}\033[0m'
        )

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    patient_fold_map = get_patient_fold_mapping(config)

    app = QApplication([])

    data_manager = DataManager(experiment_name, patient_fold_map, [left_mode, right_mode], cc_label)
    window = MainWindow(data_manager, left_mode, right_mode)

    window.show()

    app.exec()
