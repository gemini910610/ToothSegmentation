import numpy
from PySide6.QtWidgets import QLabel

from constants.color import GLASBEY_PALETTE
from constants.mode import Mode
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.selectors import PatientSelector
from visualize.ui.views import VolumeViewer
from visualize.ui.windows import BaseWindow


class TopLayout(HorizontalLayout):
    def _init_widgets(self):
        return [
            PatientSelector(),
            None,
            QLabel('Marked min: -'),
            QLabel('Unmarked max: -')
        ]

class BottomLayout(HorizontalLayout):
    def _init_widgets(self):
        return [
            VolumeViewer(1)
        ]

class MainWindow(BaseWindow):
    def __init__(self, data_manager, mode):
        super().__init__(TopLayout, BottomLayout)
        self.data_manager = data_manager
        self.mode = mode

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector, self.marked_min_label, self.unmarked_max_label = self.layouts[0].widgets
        self.volume_viewer, = self.layouts[1].widgets

        self.patient_selector.setup(data_manager.patients, self._on_patient_changed)

        self.volume_viewer.set_titles(Mode.get_title(mode))
        self.volume_viewer.views[0].picked.connect(self._on_picked)

        self.label_sizes = {}
        self.marked_labels = set()

    def _on_patient_changed(self, index):
        volumes = self.data_manager.load_data(index)
        volume = volumes[self.mode]

        labels, counts = numpy.unique(volume, return_counts=True)
        self.label_sizes = {int(label): int(count) for label, count in zip(labels, counts) if label > 0}
        self.marked_labels = set()

        self.volume_viewer.views[0].update_volume(volume, include_labels=True)

        self._update_stats()

    def _on_picked(self, actor):
        label = actor._label
        if label not in self.label_sizes:
            return

        if label in self.marked_labels:
            self.marked_labels.remove(label)
            color = tuple(c / 255 for c in GLASBEY_PALETTE[label])
        else:
            self.marked_labels.add(label)
            color = (0, 0, 0)

        self.volume_viewer.views[0].actors[f'label {label}'].prop.color = color
        self.volume_viewer.views[0].render()

        self._update_stats()

    def _update_stats(self):
        marked_sizes = [self.label_sizes[label] for label in self.marked_labels]
        unmarked_sizes = [size for label, size in self.label_sizes.items() if label not in self.marked_labels]

        marked_min = min(marked_sizes) if marked_sizes else '-'
        unmarked_max = max(unmarked_sizes) if unmarked_sizes else '-'

        self.marked_min_label.setText(f'Marked min: {marked_min}')
        self.unmarked_max_label.setText(f'Unmarked max: {unmarked_max}')

if __name__ == '__main__':
    import sys

    from post_processing.utils import iterate_datasets
    from tui.apps import ThresholdApp
    from visualize.utils import DataManager, run_app

    args = ThresholdApp(required_mode=Mode.TOOTH_CONNECTED_COMPONENT).run()
    if args is None:
        sys.exit()

    segmentation_experiment_name = args['exp']
    datasets = args['datasets']
    mode = args['mode']

    patients = [f'{dataset}/{patient}' for dataset, patient in iterate_datasets(segmentation_experiment_name, datasets)]
    data_manager = DataManager(segmentation_experiment_name, patients, [mode])
    run_app(MainWindow, data_manager, mode)
