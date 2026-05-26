import numpy
import os

from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QPushButton

from scripts.tools.widgets import Color, DataHandler, JsonHandler, MainWindowUI, Mode, PatientSelector, VolumeColorizer, VolumeLoader, VolumeViewer

class MergeDataHandler(DataHandler):
    def find(self, patient, labels):
        if patient not in self.data:
            return None
        exists = set()
        for group in self.data[patient]:
            exists |= set(group)
        labels = set(labels)
        duplicated = exists & labels
        if len(duplicated) == 0:
            return None
        return sorted(duplicated)

    def set_data(self, patient, labels):
        duplicated = self.find(patient, labels)
        if duplicated is not None:
            raise ValueError(f'Label {", ".join([str(label) for label in duplicated])} already exists.')

        if patient not in self.data:
            self.data[patient] = []
        self.data[patient].append(labels)

    def remove_data(self, patient):
        if patient not in self.data:
            return
        del self.data[patient]

class TopLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.patient_selector = PatientSelector(sizeAdjustPolicy=QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.addWidget(self.patient_selector)
        self.addStretch()
        self.preview_button = QPushButton('Preview')
        self.clear_button = QPushButton('Clear')
        self.save_button = QPushButton('Save')
        for widget in [self.preview_button, self.clear_button, self.save_button]:
            self.addWidget(widget)

    def get_widgets(self):
        return self.patient_selector, self.preview_button, self.clear_button, self.save_button

class IconCheckboxTable(QGridLayout):
    def _clear(self):
        while self.count() > 0:
            self.takeAt(0).widget().deleteLater()

    def update_items(self, labels, index_offset):
        self._clear()

        max_label = max(labels)
        palette = VolumeColorizer.glasbey_palette(max_label)
        for i, label in enumerate(labels):
            checkbox = QCheckBox(f'Label {label}')
            checkbox.setProperty('value', int(label))
            size = checkbox.font().pointSize()

            index = label.astype(numpy.int32) + index_offset
            r, g, b, _ = palette[index]
            color = QColor(r, g, b)

            pixmap = QPixmap(size, size)
            pixmap.fill(color)
            icon = QIcon(pixmap)
            checkbox.setIcon(icon)

            row = i // 2
            column = i % 2
            self.addWidget(checkbox, row, column)

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        for i in range(self.count()):
            self.itemAt(i).widget().setEnabled(enabled)

    def get_selected_label(self):
        labels = []
        for i in range(self.count()):
            checkbox = self.itemAt(i).widget()
            if checkbox.isChecked():
                label = checkbox.property('value')
                labels.append(label)
        return labels

    def clear_selected(self):
        for i in range(self.count()):
            checkbox = self.itemAt(i).widget()
            checkbox.setChecked(False)

class BottomLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.volume_viewer = VolumeViewer(2, (512, 512))
        self.label_table = IconCheckboxTable()
        self.addWidget(self.volume_viewer)
        self.addLayout(self.label_table)

    def get_widgets(self):
        return self.volume_viewer, self.label_table

class MainWindow(MainWindowUI):
    def __init__(self, data_manager):
        super().__init__(TopLayout, BottomLayout)
        self.loader = VolumeLoader(data_manager, self._handle_volumes, keep_origin=True)

        self.setWindowTitle(data_manager.experiment_name)

        self.patient_selector, self.preview_button, self.clear_button, self.save_button = self.layouts[0].get_widgets()
        self.volume_viewer, self.label_table = self.layouts[1].get_widgets()

        self.patient_selector.setup(data_manager.patients, self.loader.load_patient)

        self.volume_viewer.set_titles(Mode.get_title(Mode.REFINE_SPLIT), 'Preview')

        self.loader.setup(self.patient_selector, self.preview_button, self.clear_button, self.save_button, self.label_table)

        self.preview_button.clicked.connect(self._on_preview)
        self.clear_button.clicked.connect(self._on_clear)
        self.save_button.clicked.connect(self._on_save)

        self.volume = None
        self.display_volume = None
        self.label_masks = None

    def _handle_volumes(self, volumes):
        rgb_volume, _ = volumes[Mode.REFINE_SPLIT]
        self.volume = volumes['origin'][Mode.REFINE_SPLIT]

        self.volume_viewer.views[0].view.update_volume(rgb_volume)

        self.display_volume = numpy.zeros_like(rgb_volume)
        self.volume_viewer.views[1].view.update_volume(self.display_volume)

        labels = numpy.unique(self.volume)
        labels = labels[labels > 0]

        self.label_table.update_items(labels, -1)

        coordinates = numpy.argwhere(self.volume > 0)
        values = self.volume[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]
        self.label_masks = {
            label: coordinates[values == label]
            for label in labels
        }

    def _on_preview(self):
        if self.volume is None or self.display_volume is None or self.label_masks is None:
            return

        self.display_volume.fill(0)
        labels = self.label_table.get_selected_label()
        if len(labels) > 0:
            coordinates = numpy.concatenate([self.label_masks[label] for label in labels])
            self.display_volume[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = Color.TOOTH
        self.volume_viewer.views[1].view.update_volume(self.display_volume, reset=False)

    def _on_clear(self):
        if self.volume is None or self.display_volume is None or self.label_masks is None:
            return

        self.label_table.clear_selected()

        self.display_volume.fill(0)
        self.volume_viewer.views[1].view.update_volume(self.display_volume, reset=False)

        patient = self.patient_selector.currentText()
        path = os.path.join('merge', f'{self.loader.data_manager.experiment_name}.json')
        with JsonHandler(path, MergeDataHandler) as handler:
            handler.remove_data(patient)

    def _on_save(self):
        if self.volume is None or self.display_volume is None or self.label_masks is None:
            return

        patient = self.patient_selector.currentText()
        labels = self.label_table.get_selected_label()
        if len(labels) < 2:
            return

        path = os.path.join('merge', f'{self.loader.data_manager.experiment_name}.json')
        with JsonHandler(path, MergeDataHandler) as handler:
            handler.set_data(patient, [label for label in labels])

if __name__ == '__main__':
    from argparse import ArgumentParser
    from PySide6.QtWidgets import QApplication

    from scripts.post_processing.iterators import iterate_datasets
    from scripts.tools.widgets import DataManager

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
    patients = [f'{dataset}/{patient}' for dataset, patient in iterate_datasets(experiment_name, datasets)]

    app = QApplication([])

    data_manager = DataManager(experiment_name, patients, [Mode.REFINE_SPLIT])
    window = MainWindow(data_manager)

    window.show()

    app.exec()
