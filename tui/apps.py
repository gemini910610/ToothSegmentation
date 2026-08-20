from pathlib import Path

from textual import work
from textual.app import App

from tui.buttons import ActionButtons, BaseButton
from tui.containers import BaseContainer, BaseGrid, BaseHorizontal
from tui.decorators import HorizontalRule
from tui.inputs import BaseInput
from tui.screens import DirectoryBrowserScreen
from tui.selectors import DirectoryMultiSelector, DirectorySelector, FileSelector, OptionSelector
from tui.switches import LabelSwitch
from utils.config import load_config


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
        self.keypoint_experiment_selector = DirectorySelector(Path('logs'), 'keypoint experiment', task='keypoints')
        self.dataset_selector = DirectoryMultiSelector(title='datasets')
        return [self.segmentation_experiment_selector, self.keypoint_experiment_selector, self.dataset_selector]

    def collect_data(self):
        return {
            'segmentation_experiment_name': self.segmentation_experiment_selector.selected,
            'keypoint_experiment_name': self.keypoint_experiment_selector.selected,
            'datasets': self.dataset_selector.selected
        }

class LauncherApp(BaseApp):
    def selectors(self):
        self.segmentation_experiment_selector = DirectorySelector(Path('logs'), 'segmentation dataset', task='segmentation')
        self.keypoint_experiment_selector = DirectorySelector(Path('logs'), 'keypoint dataset', task='keypoints')
        return [self.segmentation_experiment_selector, self.keypoint_experiment_selector]

    def extra_widgets(self):
        yield HorizontalRule()

        with BaseHorizontal():
            self.segmentation_fold_input = BaseInput('segmentation fold')
            self.segmentation_fold_input.styles.width = '1fr'
            self.segmentation_fold_input.value = '4'
            yield self.segmentation_fold_input

            self.keypoint_fold_input = BaseInput('keypoint fold')
            self.keypoint_fold_input.styles.width = '1fr'
            self.keypoint_fold_input.value = '1'
            yield self.keypoint_fold_input

    def collect_data(self):
        return {
            'segmentation_experiment_name': self.segmentation_experiment_selector.selected,
            'segmentation_fold': self.segmentation_fold_input.get_value(),
            'keypoint_experiment_name': self.keypoint_experiment_selector.selected,
            'keypoint_fold': self.keypoint_fold_input.get_value()
        }

class CompareBoneLossApp(BoneLossApp):
    def extra_widgets(self):
        yield HorizontalRule()

        with BaseHorizontal():
            self.keypoint_fold_input = BaseInput('keypoint fold')
            self.keypoint_fold_input.styles.width = '1fr'
            self.keypoint_fold_input.value = '1'
            yield self.keypoint_fold_input

    def collect_data(self):
        return {
            **super().collect_data(),
            'keypoint_fold': self.keypoint_fold_input.get_value()
        }

class PanoramicPredictApp(BaseApp):
    def selectors(self):
        self.dataset_selector = DirectoryMultiSelector(Path('datasets'), 'xray datasets')
        return [self.dataset_selector]

    def collect_data(self):
        return {
            'datasets': self.dataset_selector.selected
        }

class DCM2PNGApp(BaseApp):
    def selectors(self):
        return []

    def extra_widgets(self):
        with BaseHorizontal():
            self.dcm_root_input = BaseInput('dcm root', type='text')
            self.dcm_root_input.styles.width = '1fr'
            yield self.dcm_root_input

            yield BaseButton('browse', self._on_browse_dcm_root)

        with BaseHorizontal():
            self.output_dir_input = BaseInput('output dir', type='text')
            self.output_dir_input.styles.width = '1fr'
            yield self.output_dir_input

            yield BaseButton('browse', self._on_browse_output_dir)

        yield HorizontalRule()

        with BaseHorizontal():
            self.min_input = BaseInput('min')
            self.min_input.styles.width = '1fr'
            self.min_input.value = '-1000'
            yield self.min_input

            self.max_input = BaseInput('max')
            self.max_input.styles.width = '1fr'
            self.max_input.value = '4500'
            yield self.max_input

            self.spacing_input = BaseInput('spacing', type='number')
            self.spacing_input.styles.width = '1fr'
            self.spacing_input.value = '0.25'
            yield self.spacing_input

    @work
    async def _on_browse_dcm_root(self):
        path = await self.push_screen_wait(DirectoryBrowserScreen(self.dcm_root_input.value))
        if path is not None:
            self.dcm_root_input.value = str(path)

    @work
    async def _on_browse_output_dir(self):
        path = await self.push_screen_wait(DirectoryBrowserScreen(self.output_dir_input.value, allow_new_folder=True))
        if path is not None:
            self.output_dir_input.value = str(path)

    def collect_data(self):
        return {
            'dcm_root': self.dcm_root_input.get_value(),
            'output_dir': self.output_dir_input.get_value(),
            'min': self.min_input.get_value(),
            'max': self.max_input.get_value(),
            'spacing': self.spacing_input.get_value()
        }

class RunExperimentApp(BaseApp):
    def selectors(self):
        return []

    def extra_widgets(self):
        with BaseHorizontal():
            self.task_switch = LabelSwitch('segmentation', 'keypoints')
            yield self.task_switch

    def collect_data(self):
        return {
            'task': self.task_switch.selected
        }

class TrainApp(BaseApp):
    def selectors(self):
        return []

    def extra_widgets(self):
        with BaseHorizontal():
            self.task_switch = LabelSwitch('segmentation', 'keypoints', self._on_task_changed)
            yield self.task_switch

        yield HorizontalRule()

        self.fold_selector = OptionSelector(TrainApp._folds('segmentation'), 'fold')
        yield self.fold_selector

    async def _on_task_changed(self, task):
        await self.fold_selector.update(TrainApp._folds(task))

    @staticmethod
    def _folds(task):
        config = load_config(Path('configs') / f'{task}.toml')

        return [str(fold) for fold in range(1, config.num_folds + 1)]

    def collect_data(self):
        return {
            'task': self.task_switch.selected,
            'fold': int(self.fold_selector.selected)
        }

class CorrelationApp(BaseApp):
    def selectors(self):
        self.experiment_selector = FileSelector(Path('bone_loss'), 'experiment', suffix='.xlsx')
        return [self.experiment_selector]

    def extra_widgets(self):
        yield HorizontalRule()

        with BaseHorizontal():
            self.exclude_selector = OptionSelector(['none', 'slice', 'tooth'], 'exclude')
            self.exclude_selector.styles.margin = (0, 10, 0, 0)
            yield self.exclude_selector

            self.single_only_switch = LabelSwitch('all teeth', 'single only')
            yield self.single_only_switch

    def collect_data(self):
        exclude = self.exclude_selector.selected

        return {
            'experiment_name': self.experiment_selector.selected,
            'exclude': None if exclude == 'none' else exclude,
            'single_only': self.single_only_switch.selected == 'single only'
        }

class EvaluateApp(BaseApp):
    def selectors(self):
        return []

    def extra_widgets(self):
        with BaseHorizontal():
            self.task_switch = LabelSwitch('segmentation', 'keypoints', self._on_task_changed)
            yield self.task_switch

        yield HorizontalRule()

        self.experiment_selector = DirectorySelector(Path('logs'), 'experiment', task='segmentation')
        yield self.experiment_selector

    async def _on_task_changed(self, task):
        await self.experiment_selector.set_task(task)

    def collect_data(self):
        return {
            'task': self.task_switch.selected,
            'experiment_name': self.experiment_selector.selected
        }

class SegmentApp(BaseApp):
    def selectors(self):
        self.experiment_selector = DirectorySelector(Path('logs'), 'segmentation experiment name', task='segmentation')
        self.dataset_selector = DirectoryMultiSelector(Path('datasets'), 'datasets')
        return [self.experiment_selector, self.dataset_selector]

    def extra_widgets(self):
        yield HorizontalRule()

        with BaseHorizontal():
            self.fold_input = BaseInput('segmentation fold')
            self.fold_input.styles.width = '1fr'
            self.fold_input.value = '4'
            yield self.fold_input

    def collect_data(self):
        return {
            'segmentation_experiment_name': self.experiment_selector.selected,
            'segmentation_fold': self.fold_input.get_value(),
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
