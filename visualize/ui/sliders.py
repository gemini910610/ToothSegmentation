from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget


class PercentSlider(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.slider = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=100, value=50)
        self.label = QLabel(f'{self.slider.value()}%')
        layout.addWidget(self.slider)
        layout.addWidget(self.label)

        self.slider.valueChanged.connect(lambda value: self.label.setText(f'{value}%'))

    def value(self):
        return self.slider.value() / 100

    def on_release(self, callback):
        self.slider.sliderReleased.connect(lambda: callback(self.value()))
