from PySide6.QtWidgets import QCheckBox, QGridLayout, QWidget

from constants.color import GLASBEY_PALETTE
from visualize.ui.images import Icon


class IconCheckboxTable(QWidget):
    def __init__(self):
        super().__init__()
        self.grid = QGridLayout()
        self.setLayout(self.grid)
        self.checkboxes = {}

    def _clear(self):
        while self.grid.count() > 0:
            self.grid.takeAt(0).widget().deleteLater()
        self.checkboxes.clear()

    def update_items(self, labels):
        self._clear()

        for i, label in enumerate(labels):
            label = int(label)
            checkbox = QCheckBox(f'Label {label}')
            checkbox.setProperty('value', label)
            size = checkbox.font().pointSize()

            r, g, b = GLASBEY_PALETTE[label]
            icon = Icon(size, r, g, b)
            checkbox.setIcon(icon)

            row = i // 2
            column = i % 2
            self.grid.addWidget(checkbox, row, column)
            self.checkboxes[label] = checkbox

    def selected_labels(self):
        return [
            checkbox.property('value')
            for checkbox in self.checkboxes.values()
            if checkbox.isChecked()
        ]

    def clear_selected(self):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)
