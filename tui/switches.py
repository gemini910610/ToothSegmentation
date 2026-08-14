from textual.widgets import Switch

from tui.containers import BaseHorizontal
from tui.labels import BaseLabel


class BaseSwitch(Switch):
    def __init__(self):
        super().__init__()
        self.can_focus = False
        self.styles.border = ('solid', 'transparent')
        self.styles.background = 'transparent'

class LabelSwitch(BaseHorizontal):
    def __init__(self, left_text, right_text):
        super().__init__()
        self.left_text = left_text
        self.right_text = right_text

        self.styles.width = 'auto'

    def compose(self):
        yield BaseLabel(self.left_text)
        self.switch = BaseSwitch()
        yield self.switch
        yield BaseLabel(self.right_text)

    @property
    def selected(self):
        return self.right_text if self.switch.value else self.left_text
