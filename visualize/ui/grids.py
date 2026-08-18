from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from visualize.ui.images import PointImage


def _match_keypoints(segmentation_label, keypoint_label):
    if not segmentation_label:
        return []

    start_index = min(range(len(segmentation_label)), key=lambda i: segmentation_label[i] % 10)
    label = segmentation_label[start_index]

    shift = 0
    for index, candidates in enumerate(keypoint_label):
        if label in candidates:
            shift = index - start_index
            break

    result = []
    for index, label in enumerate(segmentation_label):
        index += shift
        fdi = keypoint_label[index] if 0 <= index < len(keypoint_label) else None
        result.append((label, None if fdi is None or len(fdi) > 1 else str(fdi[0])))

    return result

def _equalize(first, second, front):
    diff = len(second) - len(first)

    if diff > 0:
        padding = [(None, None)] * diff
        first = padding + first if front else first + padding
    elif diff < 0:
        padding = [(None, None)] * -diff
        second = padding + second if front else second + padding

    return first, second


class LabelGrid(QWidget):
    def __init__(self):
        super().__init__()

        self.left_grid = QGridLayout()
        self.right_grid = QGridLayout()

        layout = QHBoxLayout()
        layout.addLayout(self.left_grid)
        layout.addItem(QSpacerItem(30, 15))
        layout.addLayout(self.right_grid)
        self.setLayout(layout)

    def update_labels(self, segmentation_labels, keypoint_labels=None, saved_data=None):
        self._clear()

        if saved_data is not None:
            results = [
                [
                    (
                        label,
                        None if str(label) not in saved_data else '-' if saved_data[str(label)] is None else str(saved_data[str(label)])
                    )
                    for label in segmentation_label
                ]
                for segmentation_label in segmentation_labels
            ]
        else:
            results = [
                _match_keypoints(segmentation_label, keypoint_label)
                for segmentation_label, keypoint_label in zip(segmentation_labels, keypoint_labels)
            ]

        results[0], results[3] = _equalize(results[0], results[3], front=True)
        results[1], results[2] = _equalize(results[1], results[2], front=False)

        return self._populate(results)

    def _clear(self):
        for grid in (self.left_grid, self.right_grid):
            while grid.count():
                item = grid.takeAt(0)
                if item is None:
                    continue

                if isinstance(item, QSpacerItem):
                    del item
                    continue

                item.widget().deleteLater()

    def _populate(self, results):
        label_editors = {}
        for i, result in enumerate(results):
            row = 0 if i in {0, 1} else 3
            grid = self.left_grid if i in {0, 3} else self.right_grid
            for index, (label, fdi) in enumerate(result):
                if label is None:
                    label_text = QLabel()
                    fdi_editor = QLineEdit()
                    fdi_editor.setVisible(False)
                else:
                    label_text = QLabel(str(label))
                    fdi_editor = QLineEdit(fdi)
                    label_editors[label] = fdi_editor

                grid.addWidget(label_text, row, index)
                grid.addWidget(fdi_editor, row + 1, index)

        self.left_grid.addItem(QSpacerItem(30, 15), 2, 0)
        self.right_grid.addItem(QSpacerItem(30, 15), 2, 0)

        return label_editors

class ImageCell:
    def __init__(self, title, row, column):
        self.title = title
        self.row = row
        self.column = column

class PointImageGrid(QWidget):
    def __init__(self, cells, size=(128, 128), clickable=False):
        super().__init__()
        self.cells = cells

        layout = QGridLayout()
        self.setLayout(layout)

        self.images = []
        self.boxes = []
        for cell in cells:
            box = QGroupBox(cell.title)
            box_layout = QVBoxLayout()
            image = PointImage(size, clickable)
            box_layout.addWidget(image)
            box.setLayout(box_layout)

            layout.addWidget(box, cell.row, cell.column)
            self.images.append(image)
            self.boxes.append(box)

        self.setFixedSize(self.sizeHint())

    def update_images(self, slices, points=None):
        if points is None:
            points = [None] * len(self.images)

        for image, slice_image, point in zip(self.images, slices, points):
            image.update_image(slice_image)

            if point is None:
                continue

            width, height = image.size
            origin_height, origin_width = slice_image.shape[:2]
            left_x, left_y, right_x, right_y = point

            image.left_point = (left_x * width / origin_width, left_y * height / origin_height)
            image.right_point = (right_x * width / origin_width, right_y * height / origin_height)
            image.update()

    def update_titles(self, suffixes):
        for box, cell, suffix in zip(self.boxes, self.cells, suffixes):
            box.setTitle(f'{cell.title} {suffix}')

    def mousePressEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.AltModifier and event.button() == Qt.MouseButton.LeftButton:
            path = Path('screenshots')
            path.mkdir(exist_ok=True)

            timestamp = datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y%m%d_%H%M%S')
            self.grab().save(str(path / f'{timestamp}.png'))

        super().mousePressEvent(event)
