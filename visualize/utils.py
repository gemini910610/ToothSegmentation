import json
from pathlib import Path

import numpy
from PySide6.QtWidgets import QApplication


class DataHandler:
    def __init__(self, data=None):
        if data is None:
            data = {}
        self.data = data

    def find(self, *args, **kwargs):
        raise NotImplementedError

    def set_data(self, *args, **kwargs):
        raise NotImplementedError

    def remove_data(self, *args, **kwargs):
        raise NotImplementedError

class JsonHandler:
    def __init__(self, path, Handler=DataHandler):
        self.path = path
        data = JsonHandler.load(path)
        self.handler = Handler(data)

    def __enter__(self):
        return self.handler

    def __exit__(self, exc_type, exc_value, traceback):
        JsonHandler.save(self.path, self.handler.data)

    @staticmethod
    def load(path):
        if not path.exists():
            return {}

        with path.open() as file:
            return json.load(file)

    @staticmethod
    def save(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)

        data = dict(data)
        with path.open('w') as file:
            json.dump(data, file, indent=4)

class DataManager:
    def __init__(self, experiment_name, patients, modes):
        self.experiment_name = experiment_name
        self.patients = patients
        self.modes = modes
        self.data_dir = Path('outputs')

    def load_data(self, index):
        patient = self.patients[index]
        data_dir = self.data_dir / self.experiment_name / patient

        return {
            mode: numpy.load(data_dir / f'{mode}.npy')
            for mode in self.modes
        }

def run_app(Window, *args, **kwargs):
    app = QApplication([])

    window = Window(*args, **kwargs)

    window.show()

    app.exec()
