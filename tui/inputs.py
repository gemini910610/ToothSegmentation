from textual.widgets import Input


class BaseInput(Input):
    def __init__(self, label):
        super().__init__(type='integer')
        self.border_title = label

        color = self.app.get_css_variables()['primary']
        self.styles.border = ('solid', color)
        self.styles.background = 'transparent'

    def get_value(self):
        value = super().value
        return None if value == '' else int(super().value)
