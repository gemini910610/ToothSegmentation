import numpy
import os

from PySide6.QtWidgets import QHBoxLayout, QComboBox, QButtonGroup, QRadioButton, QPushButton
from .widgets import VolumeViewer, Mode, VolumeColorizer, IconLabelSelector, VolumeLoader, PatientSelector, MainWindowUI, Color, JsonHandler, DataHandler, Label
from scipy import ndimage

class RemoveDataHandler(DataHandler):
    def find(self, patient, label):
        if patient not in self.data:
            return None
        for remove_type in self.data[patient]:
            if label in self.data[patient][remove_type]:
                return remove_type
        return None
    def set_data(self, patient, remove_type, label):
        origin_remove_type = self.find(patient, label)
        if origin_remove_type is not None:
            if origin_remove_type == remove_type:
                return
            self.remove_data(patient, label)

        data = self.data
        if patient not in data:
            data[patient] = {}

        data = data[patient]
        if remove_type not in data:
            data[remove_type] = []

        data = data[remove_type]
        if label not in data:
            data.append(label)
    def remove_data(self, patient, label):
        remove_type = self.find(patient, label)
        if remove_type is None:
            return

        self.data[patient][remove_type].remove(label)

        if len(self.data[patient][remove_type]) == 0:
            del self.data[patient][remove_type]

        if len(self.data[patient]) == 0:
            del self.data[patient]

class TopLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.patient_selector = PatientSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.label_selector = IconLabelSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        for widget in [self.patient_selector, self.label_selector]:
            self.addWidget(widget)
        self.addStretch()

        self.remove_selector = QButtonGroup()
        group_layout = QHBoxLayout()
        for index, title in enumerate(['Normal Tooth', 'Unerupted Tooth', 'Residual Root', 'Fake Tooth']):
            radio = QRadioButton(title)
            self.remove_selector.addButton(radio, index)
            group_layout.addWidget(radio)
        self.save_button = QPushButton('Save')
        group_layout.addWidget(self.save_button)
        group_layout.addStretch()
        self.addLayout(group_layout)
    def get_widgets(self):
        return self.patient_selector, self.label_selector, self.remove_selector, self.save_button

class BottomLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.volume_viewer = VolumeViewer()
        self.addWidget(self.volume_viewer)
    def get_widgets(self):
        return self.volume_viewer

class MainWindow(MainWindowUI):
    def __init__(self, data_manager):
        super().__init__(TopLayout, BottomLayout)
        self.loader = VolumeLoader(data_manager, self._handle_volumes, keep_origin=True)

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector, self.label_selector, self.remove_selector, self.save_button = self.top_layout.get_widgets()
        self.volume_viewer = self.bottom_layout.get_widgets()

        self.patient_selector.setup(data_manager.patients, self.loader.load_patient)
        self.remove_selector.buttons()[0].setChecked(True)

        self.volume_viewer.set_titles(Mode.get_title(Mode.CLEANED), 'Instance')

        self.loader.setup(self.patient_selector, self.label_selector, *self.remove_selector.buttons(), self.save_button)

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.save_button.clicked.connect(self._on_save)

        self.volume = None

    def _handle_volumes(self, volumes):
        self.volume = volumes['origin'][Mode.CLEANED]
        volume, tooth_count = volumes[Mode.CLEANED]
        bone_volume = volumes['origin'][Mode.BONE_CONNECTED_COMPONENT]

        erosion = ndimage.binary_erosion(bone_volume > 0)
        bone_surface = (bone_volume > 0) & (~erosion)
        volume[bone_surface] = Color.BONE

        self.volume_viewer.views[0].view.update_volume(volume)

        labels = numpy.unique(self.volume)
        labels = labels[(labels > 0) & (labels < Label.CROPPED)]

        self.label_selector.update_items(labels, tooth_count, -1)
        self._on_label_changed(0, reset=True)

    def _on_label_changed(self, index, reset=False):
        if self.volume is None:
            return

        label = self.label_selector.current_label()

        volume = self.volume == label
        volume = volume * label
        volume = volume.astype(numpy.int32, copy=False)
        volume, _ = VolumeColorizer.color_components(volume)
        self.volume_viewer.views[1].view.update_volume(volume, reset)

        self.remove_selector.buttons()[0].setChecked(True)

    def _on_save(self):
        if self.volume is None:
            return

        patient = self.patient_selector.currentText()
        label = int(self.label_selector.current_label())

        path = os.path.join('remove', f'{self.loader.data_manager.experiment_name}.json')
        with JsonHandler(path, RemoveDataHandler) as handler:
            if self.remove_selector.checkedId() == 0: # Normal Tooth
                handler.remove_data(patient, label)
            else: # Unerupted Tooth, Residual Root, Fake Tooth
                remove_type = self.remove_selector.checkedButton().text()
                handler.set_data(patient, remove_type, label)

if __name__ == '__main__':
    from argparse import ArgumentParser
    from PySide6.QtWidgets import QApplication
    from src.config import load_config
    from .widgets import get_patient_fold_mapping, DataManager

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    patient_fold_map = get_patient_fold_mapping(config)

    app = QApplication([])

    data_manager = DataManager(experiment_name, patient_fold_map, [Mode.CLEANED, Mode.BONE_CONNECTED_COMPONENT])
    window = MainWindow(data_manager)

    window.show()

    app.exec()
