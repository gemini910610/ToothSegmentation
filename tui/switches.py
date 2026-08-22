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
    def __init__(self, left_text, right_text, callback=None):
        super().__init__()
        self.left_text = left_text
        self.right_text = right_text
        self.callback = callback

        self.styles.width = 'auto'

    def compose(self):
        yield BaseLabel(self.left_text)
        self.switch = BaseSwitch()
        yield self.switch
        yield BaseLabel(self.right_text)

    async def on_switch_changed(self, event):
        if self.callback is not None:
            await self.callback(self.selected)

    @property
    def selected(self):
        return self.right_text if self.switch.value else self.left_text

class ToggleSwitch(BaseHorizontal):
    def __init__(self, text, callback=None):
        super().__init__()
        self.text = text
        self.callback = callback

        self.styles.width = 'auto'

    def compose(self):
        self.switch = BaseSwitch()
        yield self.switch
        yield BaseLabel(self.text)

    async def on_switch_changed(self, event):
        if self.callback is not None:
            await self.callback(self.selected)

    @property
    def selected(self):
        return self.switch.value
