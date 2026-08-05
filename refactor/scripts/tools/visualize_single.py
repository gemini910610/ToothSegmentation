import numpy
import os

from PySide6.QtWidgets import QButtonGroup, QComboBox, QHBoxLayout, QPushButton, QRadioButton

from refactor.scripts.tools.widgets import DataHandler, IconLabelSelector, JsonHandler, Label, MainWindowUI, Mode, PatientSelector, VolumeViewer

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
    def __init__(self, data_manager, default_patient=None, default_label=None):
        super().__init__(TopLayout, BottomLayout)
        self.data_manager = data_manager
        self.default_label = default_label

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector, self.label_selector, self.remove_selector, self.save_button = self.layouts[0].get_widgets()
        self.volume_viewer = self.layouts[1].get_widgets()

        self.patient_selector.setup(data_manager.patients, self._on_patient_changed)
        self.remove_selector.buttons()[0].setChecked(True)

        self.volume_viewer.set_titles(Mode.get_title(Mode.CLEANED), 'Instance')

        self.label_selector.currentIndexChanged.connect(self._on_label_changed)
        self.save_button.clicked.connect(self._on_save)
        self.volume_viewer.boxes[0].view.picked.connect(self._on_picked)

        self.volume = None

        if default_patient is not None:
            index = self.data_manager.patients.index(default_patient)
            self.patient_selector.setCurrentIndex(index)

    def _on_picked(self, actor):
        label = actor._label
        index = self.label_selector.findData(label)
        self.label_selector.setCurrentIndex(index)

    def _on_patient_changed(self, index):
        volumes = self.data_manager.load_data(index)
        volume = volumes[Mode.CLEANED]
        bone_volume = volumes[Mode.BONE_CONNECTED_COMPONENT]
        self.volume = volume.copy()

        volume[bone_volume > 0] = 1

        self.volume_viewer.boxes[0].view.update_volume(volume, Mode.POST_PROCESSING)

        labels = numpy.unique(self.volume)
        labels = labels[(labels > 0) & (labels < Label.CROPPED)]

        self.label_selector.update_items(labels, self.volume.max(), -1)

        if self.default_label is not None and self.default_label in labels:
            index = labels.tolist().index(self.default_label)
            self.label_selector.setCurrentIndex(index)
        else:
            self._on_label_changed(0)
        self.default_label = None

    def _on_label_changed(self, index):
        if self.volume is None:
            return

        label = self.label_selector.current_label()

        volume = self.volume == label
        volume = volume * label
        self.volume_viewer.boxes[1].view.update_volume(volume, Mode.CLEANED, reset=False)

        self.remove_selector.buttons()[0].setChecked(True)

    def _on_save(self):
        if self.volume is None:
            return

        patient = self.patient_selector.currentText()
        label = int(self.label_selector.current_label())

        path = os.path.join('remove', f'{self.data_manager.experiment_name}.json')
        with JsonHandler(path, RemoveDataHandler) as handler:
            if self.remove_selector.checkedId() == 0: # Normal Tooth
                handler.remove_data(patient, label)
            else: # Unerupted Tooth, Residual Root, Fake Tooth
                remove_type = self.remove_selector.checkedButton().text()
                handler.set_data(patient, remove_type, label)

    def closeEvent(self, event):
        super().closeEvent(event)
        for box in self.volume_viewer.boxes:
            box.view.close()

if __name__ == '__main__':
    from argparse import ArgumentParser
    from PySide6.QtWidgets import QApplication

    from refactor.scripts.post_processing.iterators import iterate_datasets
    from refactor.scripts.tools.widgets import DataManager

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('--patient', type=str)
    parser.add_argument('--label', type=int)
    args = parser.parse_args()

    experiment_name = args.exp
    default_patient = args.patient
    default_label = args.label

    datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
    patients = [f'{dataset}/{patient}' for dataset, patient in iterate_datasets(experiment_name, datasets)]

    if default_patient is not None and default_patient not in patients:
        raise ValueError(f'Patient {default_patient} not found')

    app = QApplication([])

    data_manager = DataManager(experiment_name, patients, [Mode.CLEANED, Mode.BONE_CONNECTED_COMPONENT])
    window = MainWindow(data_manager, default_patient, default_label)

    window.show()

    app.exec()
