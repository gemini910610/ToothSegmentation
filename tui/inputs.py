from textual.widgets import Input


class BaseInput(Input):
    def __init__(self, label, type='integer'):
        super().__init__(type=type)
        self.border_title = label

        color = self.app.get_css_variables()['primary']
        self.styles.border = ('solid', color)
        self.styles.background = 'transparent'

    def get_value(self):
        value = super().value
        if value == '':
            return None

        match self.type:
            case 'integer':
                return int(value)
            case 'number':
                return float(value)
            case _:
                return value
