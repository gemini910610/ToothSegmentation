import colorcet
import json
import numpy
import os

from datetime import datetime
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget
from pyvista import ImageData, PolyData, Sphere
from pyvistaqt import QtInteractor
from scipy import ndimage

class Label:
    BACKGROUND = 0
    TOOTH = 1
    BONE = 2
    CROPPED = 50
    UNERUPTED = 70
    RESIDUAL = 90
    FAKE = 110
    MISSING_FDI = 130

class Color:
    TOOTH = (235, 223, 180)
    BONE = (212, 161, 230)

class Mode:
    GROUND_TRUTH = 'gt'
    IMAGE = 'image'
    PREDICT = 'predict'
    BONE_CONNECTED_COMPONENT = 'bone_cc'
    TOOTH_CONNECTED_COMPONENT = 'tooth_cc'
    CLEANED = 'cleaned'
    WATERSHED = 'watershed'
    REFINE_SPLIT = 'refine_split'
    REFINE_MERGE = 'refine_merge'
    POST_PROCESSING = 'pp'
    REMOVED = 'removed'
    BONE_FILLED = 'bone_filled'
    TOOTH_FILLED = 'tooth_filled'
    RELABELED = 'relabeled'
    FDI = 'fdi'

    @staticmethod
    def items():
        return [
            Mode.GROUND_TRUTH,
            Mode.PREDICT,
            Mode.BONE_CONNECTED_COMPONENT,
            Mode.TOOTH_CONNECTED_COMPONENT,
            Mode.CLEANED,
            Mode.WATERSHED,
            Mode.REFINE_SPLIT,
            Mode.REFINE_MERGE,
            Mode.POST_PROCESSING,
            Mode.REMOVED,
            Mode.BONE_FILLED,
            Mode.TOOTH_FILLED,
            Mode.RELABELED,
            Mode.FDI
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
            Mode.REFINE_SPLIT: 'Refine Split',
            Mode.REFINE_MERGE: 'Refine Merge',
            Mode.POST_PROCESSING: 'Post Processing',
            Mode.REMOVED: 'Removed',
            Mode.BONE_FILLED: 'Bone Filled',
            Mode.TOOTH_FILLED: 'Tooth Filled',
            Mode.RELABELED: 'Relabeled',
            Mode.FDI: 'FDI'
        }[mode]

class DataManager:
    def __init__(self, experiment_name, patients, modes, base_output_dir='outputs'):
        self.experiment_name = experiment_name
        self.patients = patients
        self.modes = modes
        self.base_output_dir = base_output_dir

    def load_data(self, index):
        patient = self.patients[index]
        base_dir = os.path.join(self.base_output_dir, self.experiment_name, patient)

        volumes = {
            mode: numpy.load(os.path.join(base_dir, f'{mode}.npy'))
            for mode in self.modes
        }
        return volumes

def glasbey_palette(num_colors):
    colors = colorcet.glasbey_light[:num_colors]
    return [
        (int(color[1:3], 16), int(color[3:5], 16), int(color[5:], 16), 1)
        for color in colors
    ]

class SyncView(QtInteractor):
    def __init__(self, size=(640, 640)):
        super().__init__()
        self.setFixedSize(*size)

    def get_settings(self, mode, num_colors=None):
        bone_setting = {
            'name': 'bone',
            'color': tuple(color / 255 for color in Color.BONE),
            'opacity': 0.5
        }
        match mode:
            case Mode.GROUND_TRUTH | Mode.PREDICT:
                return {
                    Label.BONE: bone_setting,
                    Label.TOOTH: {
                        'name': 'tooth',
                        'color': tuple(color / 255 for color in Color.TOOTH),
                        'opacity': 1
                    },
                    3: {
                        'name': 'surface',
                        'color': (255, 0, 0),
                        'opacity': 1
                    },
                    4: {
                        'name': 'selected',
                        'color': (255, 165, 0),
                        'opacity': 1
                    }
                }
            case Mode.TOOTH_CONNECTED_COMPONENT | Mode.WATERSHED | Mode.CLEANED | Mode.REFINE_SPLIT | Mode.REFINE_MERGE | Mode.REMOVED | Mode.TOOTH_FILLED | Mode.RELABELED:
                palette = glasbey_palette(num_colors)
                return {
                    label: {
                        'name': f'label {label}',
                        'color': tuple(c / 255 for c in color),
                        'opacity': 1 if label < Label.CROPPED else 0
                    }
                    for label, color in enumerate(palette, 1)
                }
            case Mode.BONE_CONNECTED_COMPONENT | Mode.BONE_FILLED | Mode.POST_PROCESSING | Mode.FDI:
                palette = glasbey_palette(num_colors)
                return {
                    1: bone_setting,
                    **{
                        label: {
                            'name': f'label {label}',
                            'color': tuple(c / 255 for c in color),
                            'opacity': 1 if label < Label.CROPPED else 0
                        }
                        for label, color in enumerate(palette[1:], 2)
                    }
                }

    def update_volume(self, volume, mode, reset=True):
        self.clear_actors()
        render_settings = self.get_settings(mode, volume.max())

        objects = ndimage.find_objects(volume)
        for label, slices in enumerate(objects, 1):
            if slices is None:
                continue

            slices = tuple(
                slice(max(0, s.start - 1), min(shape, s.stop + 1))
                for s, shape in zip(slices, volume.shape)
            )
            roi = (volume[slices] == label).astype(numpy.uint8)
            data = ImageData(dimensions=roi.shape, origin=(s.start for s in slices))
            data.point_data['values'] = roi.flatten(order='F')
            mesh = data.contour([0.5])
            setting = render_settings[label]
            self.add_mesh(mesh, smooth_shading=True, **setting)

        if reset:
            self.view_xz()
            self.reset_camera()

    def update_points(self, points):
        data = PolyData(points)
        sphere = Sphere(radius=1)
        mesh = data.glyph(orient=False, scale=False, geom=sphere)
        self.add_mesh(mesh, color='black', name='cej points')

    def mousePressEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.AltModifier and event.button() == Qt.MouseButton.LeftButton:
            os.makedirs('screenshots', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.screenshot(f'screenshots/{timestamp}.png')

        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        self.view_xz()
        self.reset_camera()
        self.render()

class SyncViewBox(QGroupBox):
    def __init__(self, size=(640, 640)):
        super().__init__()
        self.view = SyncView(size)
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)

class VolumeViewer(QWidget):
    def __init__(self, count=2, size=(640, 640)):
        super().__init__()

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.boxes = [SyncViewBox(size) for _ in range(count)]
        for box in self.boxes:
            layout.addWidget(box)

        for box in self.boxes[1:]:
            box.view.camera = self.boxes[0].view.camera

        for box in self.boxes:
            view = box.view
            callback = self._create_callback(view)
            view.iren.add_observer('InteractionEvent', callback)
            view.iren.add_observer('MouseWheelForwardEvent', callback)
            view.iren.add_observer('MouseWheelBackwardEvent', callback)

    def _create_callback(self, source):
        def callback(caller, event):
            for box in self.boxes:
                view = box.view
                if view is not source:
                    view.render()
        return callback

    def set_titles(self, *titles):
        for box, title in zip(self.boxes, titles):
            box.setTitle(title)

class ClickableImage(QLabel):
    def __init__(self, size=(128, 128), clickable=False):
        super().__init__()
        self.size = size
        self.setFixedSize(*size)
        self.clickable = clickable

        self.dragging = False
        self.image = None
        self.left_point = None
        self.right_point = None

    def update_image(self, origin_image, points=None):
        image = Image.fromarray(origin_image)
        image = image.resize(self.size, Image.Resampling.NEAREST)
        image = ImageQt(image)
        self.image = QPixmap.fromImage(image)

        if points is None:
            self.left_point = None
            self.right_point = None
        else:
            width, height = self.size
            origin_height, origin_width = origin_image.shape[:2]

            left_x, left_y, right_x, right_y = points
            left_x = left_x * width / origin_width
            left_y = left_y * height / origin_height
            right_x = right_x * width / origin_width
            right_y = right_y * height / origin_height

            self.left_point = (left_x, left_y)
            self.right_point = (right_x, right_y)

        self.update()

    def mousePressEvent(self, event):
        if not self.clickable or self.image is None:
            event.ignore()
            return

        x = int(event.position().x())
        y = int(event.position().y())

        button = event.button()
        if button == Qt.MouseButton.LeftButton:
            self.left_point = (x, y)
            self.dragging = button
        elif button == Qt.MouseButton.RightButton:
            self.right_point = (x, y)
            self.dragging = button
        elif button == Qt.MouseButton.MiddleButton:
            self.left_point = None
            self.right_point = None

        self.update()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self.clickable or self.image is None or self.dragging is None:
            return

        x = int(event.position().x())
        y = int(event.position().y())

        if self.dragging == Qt.MouseButton.LeftButton:
            self.left_point = (x, y)
        elif self.dragging == Qt.MouseButton.RightButton:
            self.right_point = (x, y)

        self.update()

    def mouseReleaseEvent(self, event):
        if not self.clickable or self.image is None or self.dragging is None:
            return

        self.dragging = None

    def paintEvent(self, event):
        if self.image is None:
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.image)

        if self.left_point is not None:
            x, y = self.left_point
            painter.setPen(QPen(Qt.GlobalColor.red, 3))
            painter.drawEllipse(x - 2, y - 2, 4, 4)

        if self.right_point is not None:
            x, y = self.right_point
            painter.setPen(QPen(Qt.GlobalColor.green, 3))
            painter.drawEllipse(x - 2, y - 2, 4, 4)

        painter.end()

class ImageBox(QGroupBox):
    def __init__(self, title, size=(128, 128), clickable=False):
        super().__init__(title)

        layout = QVBoxLayout()
        self.image = ClickableImage(size, clickable)
        layout.addWidget(self.image)
        self.setLayout(layout)

    def update_image(self, image, points=None):
        self.image.update_image(image, points)

class ImageCell:
    def __init__(self, title, row, column):
        self.title = title
        self.row = row
        self.column = column

class ImageTable(QWidget):
    def __init__(self, cells, size=(128, 128), clickable=False):
        super().__init__()

        self.cells = cells

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.boxes = []
        for cell in self.cells:
            box = ImageBox(cell.title, size, clickable)
            self.boxes.append(box)
            self.layout.addWidget(box, cell.row, cell.column)

        self.setFixedSize(self.sizeHint())

    def update_images(self, slices, points=None):
        if points is None:
            points = [None] * len(self.cells)

        for cell, image, point in zip(self.cells, slices, points):
            self.layout.itemAtPosition(cell.row, cell.column).widget().update_image(image, point)

    def mousePressEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.AltModifier and event.button() == Qt.MouseButton.LeftButton:
            os.makedirs('screenshots', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pixmap = self.grab()
            pixmap.save(f'screenshots/{timestamp}.png')

        super().mousePressEvent(event)

class LabelSelector(QComboBox):
    def update_items(self, *args):
        self.blockSignals(True)
        self.clear()

        self._add_items(*args)

        self.setCurrentIndex(0)
        self.blockSignals(False)

    def _add_items(self, labels):
        for label in labels:
            self.addItem(f'Label {label}', label)

    def current_label(self):
        return self.currentData()

class IconLabelSelector(LabelSelector):
    def _add_items(self, labels, count, index_offset):
        palette = glasbey_palette(count)
        size = self.font().pointSize()
        for label in labels:
            index = label.astype(numpy.int32) + index_offset
            r, g, b, _ = palette[index]
            color = QColor(r, g, b)

            pixmap = QPixmap(size, size)
            pixmap.fill(color)
            icon = QIcon(pixmap)

            self.addItem(icon, f'Label {label}', label)

class PatientSelector(QComboBox):
    def setup(self, patients, current_index_changed):
        self.addItems(patients)
        self.setCurrentIndex(-1)
        self.currentIndexChanged.connect(current_index_changed)

class MainWindowUI(QMainWindow):
    def __init__(self, *Layouts, vertical=True):
        super().__init__()
        self.move(0, 0)

        widget = QWidget()
        main_layout = (QVBoxLayout if vertical else QHBoxLayout)(widget)
        self.setCentralWidget(widget)

        self.layouts = []
        for Layout in Layouts:
            if isinstance(Layout, (tuple, list)):
                Layout, kwargs = Layout
                layout = Layout(**kwargs)
            else:
                layout = Layout()
            self.layouts.append(layout)
            main_layout.addLayout(layout)

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
    def __init__(self, path, DataHandler=DataHandler):
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
            return {}

        with open(path) as file:
            return json.load(file)

    @staticmethod
    def save(path, data):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        data = dict(data)
        with open(path, 'w') as file:
            json.dump(data, file, indent=4)
