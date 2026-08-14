from textual.widgets import Label


class BaseLabel(Label):
    def __init__(self, label):
        super().__init__(label)
        self.styles.content_align_vertical = 'middle'
        self.styles.height = '100%'
