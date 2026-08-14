from textual.widgets import Rule


class HorizontalRule(Rule):
    def __init__(self):
        super().__init__()
        self.styles.height = 'auto'
        self.styles.margin = 0
