from PySide6.QtGui import QColor, QIcon, QPixmap


class Icon(QIcon):
    def __init__(self, size, r, g, b):
        super().__init__()

        color = QColor(r, g, b)
        pixmap = QPixmap(size, size)
        pixmap.fill(color)
        self.addPixmap(pixmap)
