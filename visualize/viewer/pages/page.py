from pathlib import Path

from PySide6.QtWidgets import QWidget

NOISE_LABEL = 150
FILTERED_LABEL = 170


def is_noise_label(label):
    return NOISE_LABEL <= label < FILTERED_LABEL


class Page(QWidget):
    def __init__(self, working_dir=Path('working')):
        super().__init__()
        self.working_dir = working_dir
        self.patient = None
        self.dirty = False

    def finalize(self):
        pass

class VolumeViewerPage(Page):
    def closeEvent(self, event):
        super().closeEvent(event)
        self.viewer.close()

    def _on_picked(self, actor):
        label = actor._label
        index = self.label_selector.findData(label)
        self.label_selector.setCurrentIndex(index)
