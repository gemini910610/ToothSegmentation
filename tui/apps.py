from pathlib import Path

from textual.app import App

from tui.buttons import ActionButtons
from tui.containers import BaseContainer, BaseGrid, BaseHorizontal
from tui.decorators import HorizontalRule
from tui.inputs import BaseInput
from tui.selectors import DirectoryMultiSelector, DirectorySelector
from tui.switches import LabelSwitch


class BaseApp(App):
    def compose(self):
        with BaseContainer(width='80%'):
            widgets = self.selectors()
            with BaseGrid(rows=1, columns=len(widgets)):
                yield from widgets

            yield from self.extra_widgets()

            yield HorizontalRule()

            yield ActionButtons(
                ('cancel', self._on_cancel),
                ('continue', self._on_continue, 'success')
            )

    def selectors(self):
        self.experiment_selector = DirectorySelector(Path('outputs'), 'experiment', self._on_experiment_change)
        self.dataset_selector = DirectoryMultiSelector(title='datasets')
        return [self.experiment_selector, self.dataset_selector]

    def extra_widgets(self):
        yield from ()

    def collect_data(self):
        return {
            'experiment_name': self.experiment_selector.selected,
            'datasets': self.dataset_selector.selected
        }

    def on_mount(self):
        self.screen.styles.align = ('center', 'middle')

    def _on_experiment_change(self, experiment_name):
        self.dataset_selector.update(Path('outputs') / experiment_name)

    def _on_cancel(self):
        self.exit(None)

    def _on_continue(self):
        data = self.collect_data()
        self.exit(data)

class BoneLossApp(BaseApp):
    def selectors(self):
        self.segmentation_experiment_selector = DirectorySelector(Path('outputs'), 'segmentation experiment', self._on_experiment_change)
        self.keypoint_experiment_selector = DirectorySelector(Path('logs'), 'keypoint experiment')
        self.dataset_selector = DirectoryMultiSelector(title='datasets')
        return [self.segmentation_experiment_selector, self.keypoint_experiment_selector, self.dataset_selector]

    def collect_data(self):
        return {
            'segmentation_experiment_name': self.segmentation_experiment_selector.selected,
            'keypoint_experiment_name': self.keypoint_experiment_selector.selected,
            'datasets': self.dataset_selector.selected
        }

class ConnectedComponentApp(BaseApp):
    def extra_widgets(self):
        yield HorizontalRule()

        with BaseHorizontal():
            self.label_switch = LabelSwitch('bone', 'tooth')
            self.label_switch.styles.margin = (0, 10, 0, 0)
            self.threshold_input = BaseInput('threshold')
            self.threshold_input.styles.width = '1fr'
            yield self.label_switch
            yield self.threshold_input

    def collect_data(self):
        return {
            'experiment_name': self.experiment_selector.selected,
            'datasets': self.dataset_selector.selected,
            'target': self.label_switch.selected,
            'threshold': self.threshold_input.get_value()
        }
