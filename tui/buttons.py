from textual.widgets import Button

from tui.containers import BaseHorizontal


class BaseButton(Button):
    def __init__(self, label, callback=None, variant='default'):
        super().__init__(label, variant)
        self.callback = callback

    def on_button_pressed(self):
        if self.callback is not None:
            self.callback()

class ActionButtons(BaseHorizontal):
    def __init__(self, *button_parameters):
        super().__init__(align='right')
        self.button_parameters = button_parameters

    def compose(self):
        for parameters in self.button_parameters:
            button = BaseButton(*parameters)
            button.styles.margin = (0, 0, 0, 1)
            yield button
