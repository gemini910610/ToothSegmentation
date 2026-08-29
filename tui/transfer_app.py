import stat
from pathlib import Path, PurePosixPath

from paramiko import SSHException
from textual import work
from textual.app import App
from textual.containers import Horizontal, Vertical
from textual.widgets import DirectoryTree, Label
from textual.worker import get_current_worker

from tui.buttons import BaseButton
from tui.containers import BaseHorizontal
from tui.inputs import BaseInput
from tui.screens import UpDirectoryTree
from tui.switches import LabelSwitch
from utils.transfer import SFTPTransfer, get_env


class SFTPDirectoryTree(DirectoryTree):
    PATH = PurePosixPath

    def __init__(self, sftp, root):
        self.sftp = sftp
        super().__init__(root)

    def _safe_is_dir(self, path):
        try:
            return stat.S_ISDIR(self.sftp.stat(str(path)).st_mode)
        except OSError:
            return False

    def _directory_content(self, location, worker):
        try:
            for attr in self.sftp.listdir_attr(str(location)):
                if worker.is_cancelled:
                    break
                yield location / attr.filename
        except OSError:
            pass

    @work(thread=True, exit_on_error=False)
    def _load_directory(self, node):
        path = node.data.path
        return sorted(
            self.filter_paths(self._directory_content(path, get_current_worker())),
            key=lambda path: (not self._safe_is_dir(path), path.name.lower())
        )

class TransferBrowserApp(App):
    def compose(self):
        self.status_label = Label('Connecting to server...')
        yield self.status_label

        self.panes = Horizontal()
        yield self.panes

        with BaseHorizontal():
            self.direction_switch = LabelSwitch('download', 'upload', self._on_direction_changed)
            yield self.direction_switch

            self.ignore_input = BaseInput('ignore files', type='text')
            self.ignore_input.styles.width = '1fr'
            self.ignore_input.styles.margin = (0, 2, 0, 2)
            yield self.ignore_input

            yield BaseButton('transfer', self._on_transfer)
            yield BaseButton('quit', self.exit)

        self.remote_tree = None
        self.local_tree = None
        self.remote_new_folder_input = None
        self.local_new_folder_input = None
        self.remote_path = None
        self.local_path = None
        self.transfer = None

    def on_mount(self):
        self._connect()

    @work(thread=True)
    def _connect(self):
        try:
            env = get_env()
        except FileNotFoundError:
            self.call_from_thread(self._set_status, '.env file not found. Copy .env.example to .env and fill in your SFTP credentials.')
            return

        self.transfer = SFTPTransfer(env['hostname'], env['port'], env['username'], env['password'])
        try:
            self.transfer.__enter__()
        except SSHException as exception:
            message = f'{exception.__class__.__name__}: {exception}. Try connecting manually first: ssh {env["username"]}@{env["hostname"]} -p {env["port"]}'
            self.call_from_thread(self._set_status, message)
            return

        browse_sftp = self.transfer.client.open_sftp()
        root = PurePosixPath(browse_sftp.normalize('.'))

        self.call_from_thread(self._on_connected, browse_sftp, root)

    def _on_connected(self, browse_sftp, root):
        self.remote_tree = SFTPDirectoryTree(browse_sftp, root)
        self.remote_tree.styles.width = '1fr'
        self.remote_tree.border_title = 'Remote'

        self.remote_new_folder_input = BaseInput('new folder name (optional)', type='text')
        self.remote_new_folder_input.display = False

        local_root = Path.cwd()
        self.local_tree = UpDirectoryTree(local_root)
        self.local_tree.styles.width = '1fr'
        self.local_tree.border_title = 'Local'

        self.local_new_folder_input = BaseInput('new folder name (optional)', type='text')

        self.remote_path = root
        self.local_path = local_root.resolve()

        remote_pane = Vertical(self.remote_tree, self.remote_new_folder_input)
        remote_pane.styles.width = '1fr'
        local_pane = Vertical(self.local_tree, self.local_new_folder_input)
        local_pane.styles.width = '1fr'

        self.panes.mount(remote_pane, local_pane)
        self._set_status('Connected.')

    def on_directory_tree_directory_selected(self, event):
        if event.control is self.local_tree and event.node in self.local_tree.navigation_nodes:
            self.local_tree.navigate(event.node, event.path)
            self.local_path = event.path
            return

        self._set_selected(event.control, event.path)

    def on_directory_tree_file_selected(self, event):
        self._set_selected(event.control, event.path)

    def _set_selected(self, tree, path):
        if tree is self.remote_tree:
            self.remote_path = path
        elif tree is self.local_tree:
            self.local_path = path

    def _set_status(self, text):
        self.status_label.update(text)

    async def _on_direction_changed(self, direction):
        if self.local_new_folder_input is None or self.remote_new_folder_input is None:
            return

        is_download = direction == 'download'
        self.local_new_folder_input.display = is_download
        self.remote_new_folder_input.display = not is_download

    def _get_ignore_files(self):
        value = self.ignore_input.get_value() or ''
        return [pattern.strip() for pattern in value.split(',') if pattern.strip()]

    def _on_transfer(self):
        if self.direction_switch.selected == 'download':
            self._on_download()
        else:
            self._on_upload()

    @work(thread=True)
    def _on_download(self):
        if self.transfer is None or self.remote_path is None or self.local_path is None:
            return

        destination_dir = self.local_path if self.local_path.is_dir() else self.local_path.parent
        new_folder = self.local_new_folder_input.get_value()
        if new_folder:
            destination_dir = destination_dir / new_folder

        name = self.remote_path.name
        self.call_from_thread(self._set_status, f'Downloading {name}...')

        try:
            transferred = self.transfer.download(
                self.remote_path,
                destination_dir / name,
                ignore_files=self._get_ignore_files(),
                callback=lambda relative: self.call_from_thread(self._set_status, f'Downloading {name}... ({relative})')
            )
        except Exception as exception:
            self.call_from_thread(self._set_status, f'Download failed: {exception}')
            return

        if not transferred:
            self.call_from_thread(self._set_status, f'Skipped {name} (matches ignore pattern).')
            return

        self.call_from_thread(self._finish_transfer, self.local_tree, f'Downloaded {name}.')

    @work(thread=True)
    def _on_upload(self):
        if self.transfer is None or self.remote_path is None or self.local_path is None:
            return

        destination_dir = self.remote_path if self.transfer.is_remote_dir(self.remote_path) else self.remote_path.parent
        new_folder = self.remote_new_folder_input.get_value()
        if new_folder:
            destination_dir = destination_dir / new_folder

        name = self.local_path.name
        self.call_from_thread(self._set_status, f'Uploading {name}...')

        try:
            transferred = self.transfer.upload(
                self.local_path,
                destination_dir / name,
                ignore_files=self._get_ignore_files(),
                callback=lambda relative: self.call_from_thread(self._set_status, f'Uploading {name}... ({relative})')
            )
        except Exception as exception:
            self.call_from_thread(self._set_status, f'Upload failed: {exception}')
            return

        if not transferred:
            self.call_from_thread(self._set_status, f'Skipped {name} (matches ignore pattern).')
            return

        self.call_from_thread(self._finish_transfer, self.remote_tree, f'Uploaded {name}.')

    def _finish_transfer(self, tree, message):
        self._set_status(message)
        tree.reload()

    def on_unmount(self):
        if self.transfer is not None:
            self.transfer.__exit__(None, None, None)
