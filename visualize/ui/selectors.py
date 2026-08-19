from PySide6.QtWidgets import QComboBox

from constants.color import GLASBEY_PALETTE
from visualize.ui.images import Icon


class PatientSelector(QComboBox):
    def __init__(self):
        super().__init__(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)

    def setup(self, patients, on_patient_changed):
        self.addItems(patients)
        self.setCurrentIndex(-1)
        self.currentIndexChanged.connect(on_patient_changed)

class LabelSelector(QComboBox):
    def __init__(self):
        super().__init__(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)

    def update_items(self, labels):
        self.blockSignals(True)
        self.clear()

        self._add_items(labels)

        self.setCurrentIndex(-1)
        self.blockSignals(False)

    def _add_items(self, labels):
        for label in labels:
            self.addItem(f'Label {label}', int(label))

class IconLabelSelector(LabelSelector):
    def _add_items(self, labels):
        size = self.font().pointSize()
        for label in labels:
            r, g, b = GLASBEY_PALETTE[label]
            icon = Icon(size, r, g, b)
            self.addItem(icon, f'Label {label}', int(label))
