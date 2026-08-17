from textual.widgets import RadioButton, RadioSet, SelectionList


class DirectorySelector(RadioSet):
    def __init__(self, directory=None, title=None, callback=None):
        super().__init__()
        self.directory = directory
        self.border_title = title
        self.callback = callback

        color = self.app.get_css_variables()['primary']
        self.styles.border = ('solid', color)
        self.styles.background = 'transparent'

    async def on_mount(self):
        if self.directory is not None:
            await self.update(self.directory)

    def on_radio_set_changed(self, event):
        if self.callback is not None:
            self.callback(self.selected)

    @property
    def selected(self):
        button = self.pressed_button

        return None if button is None else str(button.label)

    async def update(self, directory):
        await self.remove_children()
        await self.mount(*[
            RadioButton(folder.name)
            for folder in directory.iterdir()
        ])
        self.directory = directory

        self.children[0].value = True

class DirectoryMultiSelector(SelectionList):
    def __init__(self, directory=None, title=None):
        super().__init__(id='test')
        self.directory = directory
        self.border_title = title

        color = self.app.get_css_variables()['primary']
        self.styles.border = ('solid', color)
        self.styles.background = 'transparent'

    def on_mount(self):
        if self.directory is not None:
            self.update(self.directory)

    def update(self, directory):
        self.clear_options()
        self.add_options([
            (folder.name, folder.name)
            for folder in directory.iterdir()
        ])
        self.select_all()
        self.directory = directory
