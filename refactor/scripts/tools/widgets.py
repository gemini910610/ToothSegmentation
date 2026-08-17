import os
from datetime import datetime

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QVBoxLayout, QWidget


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
            painter.setPen(QPen(Qt.GlobalColor.red, 3))
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
