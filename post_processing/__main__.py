import subprocess
import sys
from pathlib import Path

from constants.mode import Mode
from post_processing.panoramic import find_panoramic_paths
from tui.apps import BaseApp, has_npy
from tui.selectors import OptionSelector

STAGES = [
    ('connected_component', Mode.PREDICT),
    ('watershed', Mode.TOOTH_CONNECTED_COMPONENT),
    ('refine_split', Mode.WATERSHED),
    ('refine_merge', Mode.REFINE_SPLIT),
    ('exclude_truncated', Mode.REFINE_MERGE),
    ('exclude_tooth', Mode.UNTRUNCATED),
    ('fill_holes', Mode.CLEANED),
    ('relabel', Mode.TOOTH_FILLED),
    ('relabel_fdi', Mode.RELABELED),
    ('tooth_slice', Mode.FDI),
]


def _has_panoramic_prediction():
    datasets_dir = Path('datasets')
    if not datasets_dir.is_dir():
        return False

    for dataset_dir in datasets_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        for patient_dir in dataset_dir.iterdir():
            paths = find_panoramic_paths(dataset_dir.name, patient_dir.name)
            if paths is not None and paths[1].is_file():
                return True

    return False

def _available_stages():
    outputs_dir = Path('outputs')
    stages = [name for name, mode in STAGES if has_npy(mode, depth=3)(outputs_dir)]
    if _has_panoramic_prediction():
        stages.append('panoramic')

    return stages

class PostProcessingApp(BaseApp):
    def __init__(self, available):
        super().__init__()
        self.available = available

    def _selectors(self):
        return []

    def _extra_widgets(self):
        self.stage_selector = OptionSelector(self.available, 'post-processing stage')
        yield self.stage_selector

    def _collect_data(self):
        return {
            'stage': self.stage_selector.selected
        }

available = _available_stages()
if not available:
    print('No post-processing stage is ready to run yet.')
    sys.exit()

args = PostProcessingApp(available).run()
if args is None:
    sys.exit()

stage = args['stage']
subprocess.run([sys.executable, '-m', f'post_processing.{stage}'], check=True)
