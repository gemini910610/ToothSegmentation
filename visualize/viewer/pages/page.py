from pathlib import Path

from PySide6.QtWidgets import QWidget


class Page(QWidget):
    def __init__(self, working_dir=Path('working')):
        super().__init__()
        self.working_dir = working_dir
        self.patient = None
        self.dirty = False

    def finalize(self, recompute=True):
        pass

class VolumeViewerPage(Page):
    def closeEvent(self, event):
        super().closeEvent(event)
        self.viewer.close()

    def _on_picked(self, actor):
        label = actor._label
        index = self.label_selector.findData(label)
        self.label_selector.setCurrentIndex(index)
