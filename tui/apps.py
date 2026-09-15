import sys
from argparse import ArgumentParser
from pathlib import Path

from textual import work
from textual.app import App

from constants.mode import Mode
from tui.buttons import ActionButtons, BaseButton
from tui.containers import BaseContainer, BaseGrid, BaseHorizontal
from tui.decorators import HorizontalRule
from tui.inputs import BaseInput
from tui.screens import DirectoryBrowserScreen
from tui.selectors import (
    DirectoryMultiSelector,
    DirectorySelector,
    OptionSelector,
)
from tui.switches import LabelSwitch, ToggleSwitch
from utils.config import load_config


def parse_args(app_class, configure_parser, rename=None, app_kwargs=None):
    if app_kwargs is None:
        app_kwargs = {}

    if len(sys.argv) == 1:
        args = app_class(**app_kwargs).run()
        if args is None:
            sys.exit()
    else:
        parser = ArgumentParser()
        configure_parser(parser)
        args = vars(parser.parse_args())

    if rename:
        args = {rename.get(key, key): value for key, value in args.items()}

    return args

def parse_segmentation_args(required_mode=None):
    def configure_parser(parser):
        parser.add_argument('segmentation_exp', type=str)
        parser.add_argument('datasets', nargs='+', type=str)

    return parse_args(BaseApp, configure_parser, rename={'exp': 'segmentation_exp'}, app_kwargs={'required_mode': required_mode})

def _folds(task):
    config = load_config(Path('configs') / f'{task}.toml')

    return [str(fold) for fold in range(1, config.num_folds + 1)]

def has_toml(task):
    def predicate(folder):
        return (folder / f'{task}.toml').is_file()

    return predicate

def has_npy(mode, depth):
    def predicate(folder):
        pattern = '/'.join(['*'] * depth + [f'{mode}.npy'])
        return next(folder.glob(pattern), None) is not None

    return predicate

class BaseApp(App):
    def __init__(self, required_mode=None):
        super().__init__()
        self.required_mode = required_mode

    def compose(self):
        with BaseContainer(width='80%'):
            widgets = self._selectors()
            with BaseGrid(rows=1, columns=len(widgets)):
                yield from widgets

            yield from self._extra_widgets()

            yield HorizontalRule()

            yield ActionButtons(
                ('cancel', self._on_cancel),
                ('continue', self._on_continue, 'success')
            )

    def _selectors(self):
        predicate = None if self.required_mode is None else has_npy(self.required_mode, depth=2)
        self.experiment_selector = DirectorySelector(Path('outputs'), 'experiment', self._on_experiment_change, predicate=predicate)
        self.dataset_selector = DirectoryMultiSelector(title='datasets')
        return [self.experiment_selector, self.dataset_selector]

    def _extra_widgets(self):
        yield from ()

    def _collect_data(self):
        return {
            'exp': self.experiment_selector.selected,
            'datasets': self.dataset_selector.selected
        }

    def on_mount(self):
        self.screen.styles.align = ('center', 'middle')

    def _on_experiment_change(self, experiment_name):
        predicate = None if self.required_mode is None else has_npy(self.required_mode, depth=1)
        self.dataset_selector.update(Path('outputs') / experiment_name, predicate=predicate)

    def _on_cancel(self):
        self.exit(None)

    def _on_continue(self):
        data = self._collect_data()
        self.exit(data)

class SliceApp(BaseApp):
    def _selectors(self):
        self.experiment_selector = DirectorySelector(Path('outputs') / 'Slice', 'experiment', self._on_experiment_change)
        self.dataset_selector = DirectoryMultiSelector(title='datasets')
        return [self.experiment_selector, self.dataset_selector]

    def _on_experiment_change(self, experiment_name):
        self.dataset_selector.update(Path('outputs') / 'Slice' / experiment_name)

class DualExperimentFoldApp(BaseApp):
    def _selectors(self):
        predicate = None if self.required_mode is None else has_npy(self.required_mode, depth=2)
        self.segmentation_experiment_selector = DirectorySelector(Path('outputs'), 'segmentation experiment', self._on_experiment_change, predicate=predicate)
        self.keypoints_experiment_selector = DirectorySelector(Path('logs'), 'keypoints experiment', predicate=has_toml('keypoints'))
        self.dataset_selector = DirectoryMultiSelector(title='datasets')
        return [self.segmentation_experiment_selector, self.keypoints_experiment_selector, self.dataset_selector]

    def _extra_widgets(self):
        yield HorizontalRule()

        self.keypoints_fold_selector = OptionSelector(_folds('keypoints'), 'keypoints fold', default='3')
        yield self.keypoints_fold_selector

    def _collect_data(self):
        return {
            'segmentation_exp': self.segmentation_experiment_selector.selected,
            'keypoints_exp': self.keypoints_experiment_selector.selected,
            'datasets': self.dataset_selector.selected,
            'keypoints_fold': int(self.keypoints_fold_selector.selected)
        }

class DualExperimentDualFoldApp(BaseApp):
    def _selectors(self):
        self.segmentation_experiment_selector = DirectorySelector(Path('logs'), 'segmentation experiment', predicate=has_toml('segmentation'))
        self.keypoints_experiment_selector = DirectorySelector(Path('logs'), 'keypoints experiment', predicate=has_toml('keypoints'))
        return [self.segmentation_experiment_selector, self.keypoints_experiment_selector]

    def _extra_widgets(self):
        yield HorizontalRule()

        with BaseHorizontal():
            segmentation_folds = _folds('segmentation')
            self.segmentation_fold_selector = OptionSelector(segmentation_folds, 'segmentation fold', default='4')
            yield self.segmentation_fold_selector

            self.keypoints_fold_selector = OptionSelector(_folds('keypoints'), 'keypoints fold', default='3')
            yield self.keypoints_fold_selector

    def _collect_data(self):
        return {
            'segmentation_exp': self.segmentation_experiment_selector.selected,
            'segmentation_fold': int(self.segmentation_fold_selector.selected),
            'keypoints_exp': self.keypoints_experiment_selector.selected,
            'keypoints_fold': int(self.keypoints_fold_selector.selected)
        }

class DatasetsApp(BaseApp):
    def __init__(self, predicate=None):
        super().__init__()
        self.predicate = predicate

    def _selectors(self):
        self.dataset_selector = DirectoryMultiSelector(Path('datasets'), 'xray datasets', predicate=self.predicate)
        return [self.dataset_selector]

    def _collect_data(self):
        return {
            'datasets': self.dataset_selector.selected
        }

class PathsAndNumbersApp(BaseApp):
    def _selectors(self):
        return []

    def _extra_widgets(self):
        with BaseHorizontal():
            self.dcm_root_input = BaseInput('dcm root', type='text')
            self.dcm_root_input.styles.width = '1fr'
            yield self.dcm_root_input

            yield BaseButton('browse', self._on_browse_dcm_root)

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

    def _collect_data(self):
        return {
            'dcm_root': self.dcm_root_input.get_value(),
            'min': self.min_input.get_value(),
            'max': self.max_input.get_value(),
            'spacing': self.spacing_input.get_value()
        }

class DirectoryApp(BaseApp):
    def _selectors(self):
        return []

    def _extra_widgets(self):
        with BaseHorizontal():
            self.directory_input = BaseInput('directory', type='text')
            self.directory_input.styles.width = '1fr'
            yield self.directory_input

            yield BaseButton('browse', self._on_browse)

    @work
    async def _on_browse(self):
        path = await self.push_screen_wait(DirectoryBrowserScreen(self.directory_input.value))
        if path is not None:
            self.directory_input.value = str(path)

    def _collect_data(self):
        return {
            'directory': self.directory_input.get_value()
        }

class TaskApp(BaseApp):
    def _selectors(self):
        return []

    def _extra_widgets(self):
        with BaseHorizontal():
            self.task_switch = LabelSwitch('segmentation', 'keypoints')
            yield self.task_switch

    def _collect_data(self):
        return {
            'task': self.task_switch.selected
        }

class TaskMaximizeApp(BaseApp):
    def _selectors(self):
        return []

    def _extra_widgets(self):
        with BaseHorizontal():
            self.task_switch = LabelSwitch('segmentation', 'keypoints')
            self.task_switch.styles.margin = (0, 10, 0, 0)
            yield self.task_switch

            self.mode_switch = LabelSwitch('train', 'valid')
            self.mode_switch.styles.margin = (0, 10, 0, 0)
            yield self.mode_switch

            self.maximize_switch = ToggleSwitch('maximize')
            yield self.maximize_switch

    def _collect_data(self):
        return {
            'task': self.task_switch.selected,
            'is_train': self.mode_switch.selected == 'train',
            'maximize': self.maximize_switch.selected
        }

class TaskFoldApp(BaseApp):
    def _selectors(self):
        return []

    def _extra_widgets(self):
        with BaseHorizontal():
            self.task_switch = LabelSwitch('segmentation', 'keypoints', self._on_task_changed)
            yield self.task_switch

        yield HorizontalRule()

        self.fold_selector = OptionSelector(_folds('segmentation'), 'fold')
        yield self.fold_selector

    async def _on_task_changed(self, task):
        await self.fold_selector.update(_folds(task))

    def _collect_data(self):
        return {
            'task': self.task_switch.selected,
            'fold': int(self.fold_selector.selected)
        }

class DualExperimentOptionSwitchApp(BaseApp):
    def _selectors(self):
        predicate = None if self.required_mode is None else has_npy(self.required_mode, depth=2)
        self.segmentation_experiment_selector = DirectorySelector(Path('outputs'), 'segmentation experiment', predicate=predicate)
        self.keypoints_experiment_selector = DirectorySelector(Path('logs'), 'keypoints experiment', predicate=has_toml('keypoints'))
        return [self.segmentation_experiment_selector, self.keypoints_experiment_selector]

    def _extra_widgets(self):
        yield HorizontalRule()

        with BaseHorizontal():
            self.exclude_selector = OptionSelector(['none', 'slice', 'tooth'], 'exclude')
            self.exclude_selector.styles.margin = (0, 10, 0, 0)
            yield self.exclude_selector

            self.single_only_switch = LabelSwitch('all teeth', 'single only')
            yield self.single_only_switch

    def _collect_data(self):
        exclude = self.exclude_selector.selected

        return {
            'segmentation_exp': self.segmentation_experiment_selector.selected,
            'keypoints_exp': self.keypoints_experiment_selector.selected,
            'exclude': None if exclude == 'none' else exclude,
            'single_only': self.single_only_switch.selected == 'single only'
        }

class TaskExperimentApp(BaseApp):
    def _selectors(self):
        return []

    def _extra_widgets(self):
        with BaseHorizontal():
            self.task_switch = LabelSwitch('segmentation', 'keypoints', self._on_task_changed)
            yield self.task_switch

        yield HorizontalRule()

        self.experiment_selector = DirectorySelector(Path('logs'), 'experiment', predicate=has_toml('segmentation'))
        yield self.experiment_selector

    async def _on_task_changed(self, task):
        await self.experiment_selector.set_predicate(has_toml(task))

    def _collect_data(self):
        return {
            'task': self.task_switch.selected,
            'exp': self.experiment_selector.selected
        }

class ExperimentFoldDatasetsApp(BaseApp):
    def _selectors(self):
        self.experiment_selector = DirectorySelector(Path('logs'), 'segmentation experiment name', predicate=has_toml('segmentation'))
        self.dataset_selector = DirectoryMultiSelector(Path('datasets'), 'datasets')
        return [self.experiment_selector, self.dataset_selector]

    def _extra_widgets(self):
        yield HorizontalRule()

        folds = _folds('segmentation')
        self.fold_selector = OptionSelector(folds, 'segmentation fold', default='4')
        yield self.fold_selector

    def _collect_data(self):
        return {
            'segmentation_exp': self.experiment_selector.selected,
            'segmentation_fold': int(self.fold_selector.selected),
            'datasets': self.dataset_selector.selected
        }

class SwitchThresholdApp(BaseApp):
    def _extra_widgets(self):
        yield HorizontalRule()

        with BaseHorizontal():
            self.label_switch = LabelSwitch('bone', 'tooth')
            self.label_switch.styles.margin = (0, 10, 0, 0)
            self.threshold_input = BaseInput('threshold')
            self.threshold_input.styles.width = '1fr'
            yield self.label_switch
            yield self.threshold_input

    def _collect_data(self):
        return {
            'exp': self.experiment_selector.selected,
            'datasets': self.dataset_selector.selected,
            'target': self.label_switch.selected,
            'threshold': self.threshold_input.get_value()
        }

class ThresholdApp(BaseApp):
    def _extra_widgets(self):
        yield HorizontalRule()

        modes = [Mode.TOOTH_CONNECTED_COMPONENT, Mode.BONE_CONNECTED_COMPONENT, Mode.REFINE_MERGE]
        outputs_dir = Path('outputs')
        self.mode_titles = {Mode.get_title(mode): mode for mode in modes if has_npy(mode, depth=3)(outputs_dir)}

        self.mode_selector = OptionSelector(list(self.mode_titles), 'mode')
        yield self.mode_selector

    def _collect_data(self):
        data = super()._collect_data()
        title = self.mode_selector.selected
        data['mode'] = None if title is None else self.mode_titles[title]
        return data
