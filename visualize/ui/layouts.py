from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout


class BaseLayout:
    def __init__(self):
        super().__init__()
        widgets = self._init_widgets()
        self.widgets = []
        for widget in widgets:
            if widget is None:
                self.addStretch()
            else:
                self.addWidget(widget)
                self.widgets.append(widget)

    def _init_widgets(self):
        return []

class HorizontalLayout(BaseLayout, QHBoxLayout):
    pass

class VerticalLayout(BaseLayout, QVBoxLayout):
    pass
