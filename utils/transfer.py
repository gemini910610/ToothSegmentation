import fnmatch
import os
import stat
from pathlib import Path, PurePosixPath

from dotenv import load_dotenv
from paramiko import SSHClient


class SFTPTransfer:
    def __init__(self, hostname, port, username, password):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password

        self.client = None
        self.sftp = None

    def _walk_remote(self, remote_dir=None, relative_dir=None, ignore_files=None):
        if remote_dir is None:
            remote_dir = PurePosixPath('.')
        if relative_dir is None:
            relative_dir = PurePosixPath('.')
        if ignore_files is None:
            ignore_files = []

        total_files = []
        total_size = 0
        for item in self.sftp.listdir_attr(str(remote_dir)):
            path = remote_dir / item.filename
            relative_path = relative_dir / item.filename

            if stat.S_ISLNK(item.st_mode):
                print(f'\033[33m[WARNING]\033[0m {path} is a symbolic link, will be ignored.')
                continue
            if any(fnmatch.fnmatch(item.filename, pattern) for pattern in ignore_files):
                continue

            if stat.S_ISDIR(item.st_mode):
                files, size = self._walk_remote(path, relative_path, ignore_files)
                total_files.extend(files)
                total_size += size
            else:
                size = item.st_size
                if size is None:
                    size = 0
                total_files.append((path, relative_path, size))
                total_size += size

        return total_files, total_size

    def _walk_local(self, local_dir, relative_dir=None, ignore_files=None):
        if relative_dir is None:
            relative_dir = Path('.')
        if ignore_files is None:
            ignore_files = []

        total_files = []
        total_size = 0
        for item in local_dir.iterdir():
            relative_path = relative_dir / item.name

            if item.is_symlink():
                print(f'\033[33m[WARNING]\033[0m {item} is a symbolic link, will be ignored.')
                continue
            if any(fnmatch.fnmatch(item.name, pattern) for pattern in ignore_files):
                continue

            if item.is_dir():
                files, size = self._walk_local(item, relative_path, ignore_files)
                total_files.extend(files)
                total_size += size
            else:
                size = item.stat().st_size
                total_files.append((item, relative_path, size))
                total_size += size

        return total_files, total_size

    def is_remote_dir(self, path):
        try:
            return stat.S_ISDIR(self.sftp.stat(str(path)).st_mode)
        except OSError:
            return False

    def _mkdir_remote_p(self, remote_dir):
        remote_dir = PurePosixPath(remote_dir)
        parts = remote_dir.parts

        current = PurePosixPath(parts[0])
        for part in parts[1:]:
            current = current / part
            try:
                self.sftp.mkdir(str(current))
            except OSError:
                pass

    def download(self, remote_path, local_path, ignore_files=None, callback=None):
        remote_path = PurePosixPath(remote_path)
        local_path = Path(local_path)
        ignore_files = ignore_files or []

        if self.is_remote_dir(remote_path):
            files, total_size = self._walk_remote(remote_path, ignore_files=ignore_files)
            transferred_size = 0
            for source_path, relative_path, size in files:
                destination = local_path / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)

                def file_callback(file_transferred, file_size, relative_path=relative_path, base=transferred_size):
                    if callback is not None:
                        callback(relative_path, base + file_transferred, total_size)

                self.sftp.get(str(source_path), destination, callback=file_callback)
                transferred_size += size
            return True

        if any(fnmatch.fnmatch(remote_path.name, pattern) for pattern in ignore_files):
            return False

        local_path.parent.mkdir(parents=True, exist_ok=True)
        total_size = self.sftp.stat(str(remote_path)).st_size
        file_callback = None if callback is None else (lambda transferred, size: callback(local_path.name, transferred, total_size))
        self.sftp.get(str(remote_path), local_path, callback=file_callback)
        return True

    def upload(self, local_path, remote_path, ignore_files=None, callback=None):
        local_path = Path(local_path)
        remote_path = PurePosixPath(remote_path)
        ignore_files = ignore_files or []

        if local_path.is_dir():
            files, total_size = self._walk_local(local_path, ignore_files=ignore_files)
            self._mkdir_remote_p(remote_path)
            transferred_size = 0
            for source_path, relative_path, size in files:
                destination = remote_path.joinpath(*relative_path.parts)
                self._mkdir_remote_p(destination.parent)

                def file_callback(file_transferred, file_size, relative_path=relative_path, base=transferred_size):
                    if callback is not None:
                        callback(relative_path, base + file_transferred, total_size)

                self.sftp.put(source_path, str(destination), callback=file_callback)
                transferred_size += size
            return True

        if any(fnmatch.fnmatch(local_path.name, pattern) for pattern in ignore_files):
            return False

        self._mkdir_remote_p(remote_path.parent)
        total_size = local_path.stat().st_size
        file_callback = None if callback is None else (lambda transferred, size: callback(local_path.name, transferred, total_size))
        self.sftp.put(local_path, str(remote_path), callback=file_callback)
        return True

    def __enter__(self):
        self.client = SSHClient()
        self.client.load_system_host_keys()
        self.client.connect(self.hostname, self.port, self.username, self.password)
        self.sftp = self.client.open_sftp()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.sftp is not None:
            self.sftp.close()
        if self.client is not None:
            self.client.close()
        return False

def get_env():
    if not Path('.env').is_file():
        raise FileNotFoundError(
            '.env file not found.\033[0m\n'
            'This file is required for SFTP connection.\n\n'
            'Copy ".env.example" to ".env" and fill in your credentials.\n'
            'Example:\n'
            '\033[90mSFTP_HOSTNAME = your.server.address\n'
            'SFTP_PORT = 22\n'
            'SFTP_USERNAME = your_username\n'
            'SFTP_PASSWORD = your_password\033[0m\n'
            'Please refer to README.md for more details.'
        )

    load_dotenv()
    return {
        'hostname': os.getenv('SFTP_HOSTNAME'),
        'port': int(os.getenv('SFTP_PORT')),
        'username': os.getenv('SFTP_USERNAME'),
        'password': os.getenv('SFTP_PASSWORD')
    }
