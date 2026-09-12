from textual.widgets import RadioButton, RadioSet, SelectionList


class DirectorySelector(RadioSet):
    def __init__(self, directory=None, title=None, callback=None, predicate=None):
        super().__init__()
        self.directory = directory
        self.border_title = title
        self.callback = callback
        self.predicate = predicate

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

        folders = [
            folder for folder in directory.iterdir()
            if self.predicate is None or self.predicate(folder)
        ]

        buttons = [RadioButton(folder.name) for folder in folders]
        for button in buttons:
            button.can_focus = False

        await self.mount(*buttons)
        self.directory = directory

        self._selected = 0
        self.children[0].value = True

    async def set_predicate(self, predicate):
        self.predicate = predicate
        await self.update(self.directory)

        self._selected = 0
        self.children[0].value = True

class OptionSelector(RadioSet):
    def __init__(self, options, title=None, default=None):
        super().__init__()
        self.options = options
        self.border_title = title
        self.default = default

        color = self.app.get_css_variables()['primary']
        self.styles.border = ('solid', color)
        self.styles.background = 'transparent'

    async def on_mount(self):
        await self.update(self.options, self.default)

    async def update(self, options, default=None):
        await self.remove_children()

        buttons = [RadioButton(option) for option in options]
        for button in buttons:
            button.can_focus = False

        await self.mount(*buttons)

        index = options.index(default) if default in options else 0
        self._selected = index
        self.children[index].value = True

        self.options = options
        self.default = default

    @property
    def selected(self):
        button = self.pressed_button

        return None if button is None else str(button.label)

class DirectoryMultiSelector(SelectionList):
    def __init__(self, directory=None, title=None, predicate=None):
        super().__init__(id='test')
        self.directory = directory
        self.border_title = title
        self.predicate = predicate

        color = self.app.get_css_variables()['primary']
        self.styles.border = ('solid', color)
        self.styles.background = 'transparent'

    def on_mount(self):
        if self.directory is not None:
            self.update(self.directory)

    def update(self, directory, predicate=None):
        self.predicate = predicate

        self.clear_options()
        self.add_options([
            (folder.name, folder.name)
            for folder in directory.iterdir()
            if self.predicate is None or self.predicate(folder)
        ])
        self.select_all()
        self.directory = directory
