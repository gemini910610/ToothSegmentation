from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QRadioButton, QWidget


class RadioGroup(QWidget):
    def __init__(self, titles):
        super().__init__()
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.group = QButtonGroup()
        for title in titles:
            radio = QRadioButton(title)
            layout.addWidget(radio)
            self.group.addButton(radio)

    def checked_title(self):
        button = self.group.checkedButton()

        return None if button is None else button.text()

    def select_first(self):
        self.group.buttons()[0].setChecked(True)

    def select_title(self, title):
        for button in self.group.buttons():
            if button.text() == title:
                button.setChecked(True)
                return

        checked_button = self.group.checkedButton()
        if checked_button is not None:
            self.group.setExclusive(False)
            checked_button.setChecked(False)
            self.group.setExclusive(True)

    def on_change(self, callback):
        self.group.buttonClicked.connect(lambda button: callback(button.text()))
