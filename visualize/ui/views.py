from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget
from pyvista import ImageData
from pyvistaqt import QtInteractor
from scipy import ndimage

from constants.color import GLASBEY_PALETTE, Color
from constants.label import Label


class SyncView(QtInteractor):
    picked = Signal(object)

    def __init__(self, size=(640, 640)):
        super().__init__()
        self.setFixedSize(*size)

        self.iren.picker = 'cell'

    @staticmethod
    def color_setting(*, color, **kwargs):
        return {
            'color': tuple(c / 255 for c in color),
            **kwargs
        }

    @staticmethod
    def get_settings(include_labels=False, num_colors=None):
        bone_setting = SyncView.color_setting(name='bone', color=Color.BONE, opacity=0.5)

        if not include_labels:
            return {
                Label.BONE: bone_setting,
                Label.TOOTH: SyncView.color_setting(name='tooth', color=Color.TOOTH)
            }

        return {
            1: bone_setting,
            **{
                label: SyncView.color_setting(name=f'label {label}', color=GLASBEY_PALETTE[label])
                for label in range(2, num_colors + 1)
            }
        }

    def update_volume(self, volume, include_labels=False, reset=True, render_settings=None):
        self.clear_actors()
        if render_settings is None:
            render_settings = SyncView.get_settings(include_labels, volume.max())

        objects = ndimage.find_objects(volume)
        for label, slices in enumerate(objects, 1):
            if slices is None:
                continue

            setting = render_settings[label]
            if setting.get('opacity') == 0:
                continue

            slices = tuple(
                slice(max(0, s.start - 1), min(shape, s.stop + 1))
                for s, shape in zip(slices, volume.shape)
            )
            roi = (volume[slices] == label).astype(numpy.uint8)

            data = ImageData(dimensions=roi.shape, origin=(s.start for s in slices))
            data.point_data['values'] = roi.flatten(order='F')
            mesh = data.contour([0.5])

            actor = self.add_mesh(mesh, smooth_shading=True, **setting)
            actor._label = label

        if 'tooth' in self.actors:
            self.actors['tooth'].SetPickable(False)
        if 'bone' in self.actors:
            self.actors['bone'].SetPickable(False)

        if reset:
            self.view_xz()
            self.reset_camera()

    def mousePressEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.AltModifier and event.button() == Qt.MouseButton.LeftButton:
            path = Path('screenshots')
            path.mkdir(exist_ok=True)

            timestamp = datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y%m%d_%H%M%S')
            self.screenshot(path / f'{timestamp}.png')

        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)

        if event.button() == Qt.MouseButton.LeftButton:
            x, y = self.iren.get_event_position()
            renderer = self.iren.get_poked_renderer(x, y)
            self.iren.picker.Pick(x, y, 0, renderer)
            actor = self.iren.picker.GetActor()
            if actor is not None:
                self.picked.emit(actor)

    def contextMenuEvent(self, event):
        self.view_xz()
        self.reset_camera()
        self.render()

class SyncViewBox(QGroupBox):
    def __init__(self, size=(640, 640)):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
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
        self.views = []
        for box in self.boxes:
            layout.addWidget(box)
            self.views.append(box.view)

        camera = self.views[0].camera
        for view in self.views[1:]:
            view.camera = camera

        for view in self.views:
            callback = self._create_callback(view)
            for event in ('InteractionEvent', 'MouseWheelForwardEvent', 'MouseWheelBackwardEvent'):
                view.iren.add_observer(event, callback)

    def _create_callback(self, source):
        def callback(caller, event):
            for view in self.views:
                if view is not source:
                    view.render()

        return callback

    def set_titles(self, *titles):
        for box, title in zip(self.boxes, titles):
            box.setTitle(title)

    def closeEvent(self, event):
        super().closeEvent(event)
        for view in self.views:
            view.close()
