from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QVBoxLayout, QWidget


class BaseWindow(QMainWindow):
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
            main_layout.addLayout(layout)
            self.layouts.append(layout)

    def closeEvent(self, event):
        super().closeEvent(event)
        for layout in self.layouts:
            for widget in layout.widgets:
                widget.close()
