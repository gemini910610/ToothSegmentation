import subprocess
import sys
from pathlib import Path

from constants.mode import Mode
from tui.apps import BaseApp, mode_predicate
from tui.selectors import OptionSelector

EDITORS = [
    ('threshold', Mode.TOOTH_CONNECTED_COMPONENT),
    ('split', Mode.WATERSHED),
    ('merge', Mode.REFINE_SPLIT),
    ('exclude', Mode.UNTRUNCATED),
    ('fdi', Mode.RELABELED),
]


def _available_editors():
    outputs_dir = Path('outputs')
    return [name for name, mode in EDITORS if mode_predicate(mode, depth=3)(outputs_dir)]

class EditorApp(BaseApp):
    def __init__(self, available):
        super().__init__()
        self.available = available

    def _selectors(self):
        return []

    def _extra_widgets(self):
        self.editor_selector = OptionSelector(self.available, 'editor')
        yield self.editor_selector

    def _collect_data(self):
        return {
            'editor': self.editor_selector.selected
        }

available = _available_editors()
if not available:
    print('No editor is ready to run yet.')
    sys.exit()

args = EditorApp(available).run()
if args is None:
    sys.exit()

editor = args['editor']
subprocess.run([sys.executable, '-m', f'visualize.editor.{editor}'], check=True)
