import string
import sys
from pathlib import Path

from rich.text import Text
from textual.screen import ModalScreen
from textual.widgets import DirectoryTree
from textual.widgets._tree import TOGGLE_STYLE

from tui.buttons import ActionButtons
from tui.containers import BaseContainer
from tui.decorators import HorizontalRule
from tui.inputs import BaseInput


def _list_drives():
    if sys.platform != 'win32':
        return []

    return [
        drive
        for letter in string.ascii_uppercase
        if (drive := Path(f'{letter}:/')).is_dir()
    ]

class UpDirectoryTree(DirectoryTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.navigation_nodes = set()
        self.show_drive_picker = False

    def _populate_node(self, node, content):
        if node is self.root and self.show_drive_picker:
            self._populate_drive_picker(node)
            return

        super()._populate_node(node, content)

        if node is not self.root:
            return

        self.show_root = True
        self.navigation_nodes.clear()

        parent = node.data.path.parent
        if parent == node.data.path and len(_list_drives()) <= 1:
            return

        up_node = node.add('..', data=type(node.data)(parent), allow_expand=False, before=0)
        self.navigation_nodes.add(up_node)

    def _populate_drive_picker(self, node):
        node.remove_children()
        self.show_root = False
        self.navigation_nodes.clear()

        for drive in _list_drives():
            drive_node = node.add(str(drive), data=type(node.data)(drive), allow_expand=False)
            self.navigation_nodes.add(drive_node)

        node.expand()

    def navigate(self, node, path):
        if str(node.label) == '..' and self.path.parent == self.path:
            self.show_drive_picker = True
        else:
            self.show_drive_picker = False

        self.path = path

    def render_label(self, node, base_style, style):
        if node not in self.navigation_nodes:
            return super().render_label(node, base_style, style)

        node_label = node._label.copy()
        node_label.stylize(style)
        node_label.stylize_before(
            self.get_component_rich_style('directory-tree--folder', partial=True)
        )

        return Text.assemble((self.ICON_NODE, base_style + TOGGLE_STYLE), node_label)

class DirectoryBrowserScreen(ModalScreen):
    def __init__(self, root=None, allow_new_folder=False):
        super().__init__()
        self.root = Path(root).resolve() if root else Path.cwd()
        if not self.root.is_dir():
            self.root = Path.cwd()

        self.allow_new_folder = allow_new_folder
        self.new_folder_input = None
        self.chosen_path = self.root

    def compose(self):
        with BaseContainer(width='80%', height='80%'):
            self.directory_tree = UpDirectoryTree(self.root)
            self.directory_tree.styles.height = '1fr'
            yield self.directory_tree

            if self.allow_new_folder:
                self.new_folder_input = BaseInput('new folder name (optional)', type='text')
                yield self.new_folder_input

            yield HorizontalRule()

            yield ActionButtons(
                ('cancel', self._on_cancel),
                ('select', self._on_select, 'success')
            )

    def on_mount(self):
        self.styles.align = ('center', 'middle')

    def on_directory_tree_directory_selected(self, event):
        self.chosen_path = event.path

        if event.node in self.directory_tree.navigation_nodes:
            self.directory_tree.navigate(event.node, event.path)

    def _on_cancel(self):
        self.dismiss(None)

    def _on_select(self):
        new_folder_name = None if self.new_folder_input is None else self.new_folder_input.get_value()
        path = self.chosen_path if not new_folder_name else self.chosen_path / new_folder_name

        self.dismiss(path)
