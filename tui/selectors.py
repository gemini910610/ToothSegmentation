from textual.widgets import RadioButton, RadioSet, SelectionList


class DirectorySelector(RadioSet):
    def __init__(self, directory=None, title=None, callback=None, task=None):
        super().__init__()
        self.directory = directory
        self.border_title = title
        self.callback = callback
        self._task_name = task

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
            if self._task_name is None or (folder / f'{self._task_name}.toml').is_file()
        ]

        buttons = [RadioButton(folder.name) for folder in folders]
        for button in buttons:
            button.can_focus = False

        await self.mount(*buttons)
        self.directory = directory

        self._selected = 0
        self.children[0].value = True

    async def set_task(self, task):
        self._task_name = task
        await self.update(self.directory)

class FileSelector(RadioSet):
    def __init__(self, directory=None, title=None, suffix=''):
        super().__init__()
        self.directory = directory
        self.border_title = title
        self.suffix = suffix

        color = self.app.get_css_variables()['primary']
        self.styles.border = ('solid', color)
        self.styles.background = 'transparent'

    async def on_mount(self):
        if self.directory is not None:
            await self.update(self.directory)

    @property
    def selected(self):
        button = self.pressed_button

        return None if button is None else str(button.label)

    async def update(self, directory):
        await self.remove_children()

        files = sorted(path.stem for path in directory.iterdir() if path.suffix == self.suffix)

        buttons = [RadioButton(file) for file in files]
        for button in buttons:
            button.can_focus = False

        await self.mount(*buttons)
        self.directory = directory

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
