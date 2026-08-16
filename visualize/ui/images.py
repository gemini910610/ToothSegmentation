import cv2
from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QColor, QDrag, QIcon, QImage, QPixmap
from PySide6.QtWidgets import QLabel, QMessageBox


class Icon(QIcon):
    def __init__(self, size, r, g, b):
        super().__init__()

        color = QColor(r, g, b)
        pixmap = QPixmap(size, size)
        pixmap.fill(color)
        self.addPixmap(pixmap)

def bgr_to_pixmap(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_RGB888)

    return QPixmap.fromImage(image)

class AutoFitImage(QLabel):
    def __init__(self, pixmap=None):
        super().__init__(scaledContents=True)
        self.origin_pixmap = pixmap
        if pixmap is not None:
            self.setPixmap(pixmap)

    def resizeEvent(self, event):
        if self.origin_pixmap is None:
            return

        pixmap = self.origin_pixmap.scaledToWidth(event.size().width())
        self.setPixmap(pixmap)

    def update_image(self, pixmap):
        self.origin_pixmap = pixmap
        pixmap = pixmap.scaledToWidth(self.width())
        self.setPixmap(pixmap)

class SourceImage(AutoFitImage):
    def __init__(self, pixmap=None):
        super().__init__(pixmap)
        self.origin_mask = None

    def update_image(self, pixmap, mask):
        super().update_image(pixmap)
        self.origin_mask = mask

    def mousePressEvent(self, event):
        if self.origin_mask is None:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            width = self.pixmap().width()
            height = self.pixmap().height()
            origin_height, origin_width = self.origin_mask.shape
            position = event.position()
            x = round(position.x() / width * origin_width)
            y = round(position.y() / height * origin_height)
            label = self.origin_mask[y, x]
            if label == 0:
                return

            drag = QDrag(self)
            data = QMimeData()
            data.setText(str(label))
            drag.setMimeData(data)
            drag.exec(Qt.DropAction.MoveAction)

class DestinationImage(AutoFitImage):
    fdi_updated = Signal(int, str)

    def __init__(self, pixmap=None):
        super().__init__(pixmap)
        self.setAcceptDrops(True)
        self.tooth_data = None

    def update_image(self, pixmap, tooth_data):
        super().update_image(pixmap)
        self.tooth_data = tooth_data

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if self.tooth_data is None:
            return

        width = self.pixmap().width()
        height = self.pixmap().height()
        origin_width = self.origin_pixmap.width()
        origin_height = self.origin_pixmap.height()
        position = event.position()
        x = round(position.x() / width * origin_width)
        y = round(position.y() / height * origin_height)
        label = int(event.mimeData().text())

        tooth_data = sorted(self.tooth_data, key=lambda tooth: (tooth['center'][0] - x) ** 2 + (tooth['center'][1] - y) ** 2)
        fdi = None
        for tooth in tooth_data:
            if abs(x - tooth['center'][0]) < tooth['box_size'][0] / 2 and abs(y - tooth['center'][1]) < tooth['box_size'][1] / 2:
                fdi = tooth['FDI']
                break

        if fdi is None:
            return

        if len(fdi) > 1:
            message_box = QMessageBox(self, text=f'Select FDI of Label {label}')
            message_box.setWindowTitle('Multiple FDI')
            message_box.setStandardButtons(QMessageBox.StandardButton.Cancel)
            for value in fdi:
                button = message_box.addButton(value, QMessageBox.ButtonRole.AcceptRole)
                button.setProperty('value', value)
            message_box.exec()

            button = message_box.clickedButton()
            fdi = button.property('value')
            if fdi is None:
                return
        else:
            fdi = fdi[0]

        self.fdi_updated.emit(label, fdi)
