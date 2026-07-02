import copy
import cv2
import numpy

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QDrag, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpacerItem

from scripts.post_processing.panoramic import draw_label, get_panoramic
from scripts.tools.widgets import DataHandler, glasbey_palette, JsonHandler, MainWindowUI, Mode, PatientSelector

class FDIHandler(DataHandler):
    def set_data(self, patient, data):
        self.data[patient] = data

def IoU(data_a, data_b):
    a_center_x, a_center_y = data_a['center']
    a_width, a_height = data_a['box_size']
    a_area = a_width * a_height
    a_left = a_center_x - a_width / 2
    a_right = a_center_x + a_width / 2
    a_top = a_center_y - a_height / 2
    a_bottom = a_center_y + a_height / 2

    b_center_x, b_center_y = data_b['center']
    b_width, b_height = data_b['box_size']
    b_area = b_width * b_height
    b_left = b_center_x - b_width / 2
    b_right = b_center_x + b_width / 2
    b_top = b_center_y - b_height / 2
    b_bottom = b_center_y + b_height / 2

    left = max(a_left, b_left)
    right = min(a_right, b_right)
    top = max(a_top, b_top)
    bottom = min(a_bottom, b_bottom)

    intersection = max(0, right - left) * max(0, bottom - top)
    union = a_area + b_area - intersection
    return intersection / union

def merge_box(tooth_data, threshold=0.9):
    box_count = len(tooth_data)
    overlaps = {
        index: []
        for index in range(box_count)
    }
    for i in range(box_count - 1):
        for j in range(i + 1, box_count):
            iou = IoU(tooth_data[i], tooth_data[j])
            if iou > threshold:
                overlaps[i].append(j)
                overlaps[j].append(i)

    visited = [False] * box_count
    merged_data = []
    for index in range(box_count):
        if visited[index]:
            continue

        group = []
        queue = [index]
        visited[index] = True
        while queue:
            current = queue.pop(0)
            group.append(current)
            for neighbor in overlaps[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        best_index = max(group, key=lambda index: tooth_data[index]['bone_loss'])
        data = copy.deepcopy(tooth_data[best_index])
        for index in group:
            data['FDI'].extend(tooth_data[index]['FDI'])
        data['FDI'] = sorted(set(data['FDI']))
        merged_data.append(data)

    return merged_data

def load_panoramic(dataset, patient):
    xray_folder = os.path.join('datasets', dataset.replace('bone_tooth_mask', 'xray'), patient)
    for folder in os.listdir(xray_folder):
        if folder.endswith('Pano'):
            folder = os.path.join(xray_folder, folder)
            filename = os.listdir(folder)[0]
            image = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            filename = filename[:-4]
            break
    with open(f'pano_predict/{filename}.txt') as file:
        data = file.read().strip()
    tooth_data = []
    for line in data.split('\n'):
        fdi, center_x, center_y, width, height, left_crone_x, left_crone_y, left_bone_x, left_bone_y, left_root_x, left_root_y, right_crone_x, right_crone_y, right_bone_x, right_bone_y, right_root_x, right_root_y, bone_loss = line.split(',')
        tooth_data.append({
            'FDI': [fdi],
            'center': (float(center_x), float(center_y)),
            'box_size': (int(width), int(height)),
            'left_crone': (float(left_crone_x), float(left_crone_y)),
            'left_bone': (float(left_bone_x), float(left_bone_y)),
            'left_root': (float(left_root_x), float(left_root_y)),
            'right_crone': (float(right_crone_x), float(right_crone_y)),
            'right_bone': (float(right_bone_x), float(right_bone_y)),
            'right_root': (float(right_root_x), float(right_root_y)),
            'bone_loss': int(bone_loss)
        })
    tooth_data = merge_box(tooth_data)
    return image, tooth_data

def draw_panoramic(image, tooth_data):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    palette = glasbey_palette(48)
    for tooth in tooth_data:
        center_x, center_y = tooth['center']
        (width, height), baseline = cv2.getTextSize(tooth['FDI'][0], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        x = int(center_x - width / 2)
        y = int(center_y + (height - baseline) / 2)
        color = palette[int(tooth['FDI'][0])-1][:3]
        for fdi in tooth['FDI']:
            cv2.putText(image, fdi, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 10, cv2.LINE_AA)
            cv2.putText(image, fdi, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            y += height + 15
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def rgb2pixmap(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(image)
    return pixmap

class AutoFitImage(QLabel):
    def __init__(self, pixmap=None):
        super().__init__(scaledContents=True)
        self.origin_pixmap = pixmap
        if pixmap is not None:
            self.setPixmap(pixmap)

    def resizeEvent(self, event):
        if self.origin_pixmap is None:
            return
        pixmap = self.origin_pixmap
        pixmap = pixmap.scaledToWidth(event.size().width())
        self.setPixmap(pixmap)

    def update_image(self, pixmap):
        self.origin_pixmap = pixmap
        width = self.width()
        pixmap = pixmap.scaledToWidth(width)
        self.setPixmap(pixmap)

class SourceImage(AutoFitImage):
    def __init__(self, pixmap=None):
        super().__init__(pixmap)
        self.origin_mask = None

    def update_image(self, pixmap, mask):
        super().update_image(pixmap)
        self.origin_mask = mask

    def mousePressEvent(self, event):
        if self.origin_mask is None:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            width = self.pixmap().width()
            height = self.pixmap().height()
            origin_height, origin_width = self.origin_mask.shape
            position = event.position()
            x = round(position.x() / width * origin_width)
            y = round(position.y() / height * origin_height)
            label = self.origin_mask[y, x]
            if label == 0:
                return

            drag = QDrag(self)
            data = QMimeData()
            data.setText(str(label))
            drag.setMimeData(data)
            drag.exec(Qt.DropAction.MoveAction)

class DestinationImage(AutoFitImage):
    fdi_updated = Signal(int, str)

    def __init__(self, pixmap=None):
        super().__init__(pixmap)
        self.setAcceptDrops(True)
        self.tooth_data = None

    def update_image(self, pixmap, tooth_data):
        super().update_image(pixmap)
        self.tooth_data = tooth_data

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if self.tooth_data is None:
            return

        width = self.pixmap().width()
        height = self.pixmap().height()
        origin_width = self.origin_pixmap.width()
        origin_height = self.origin_pixmap.height()
        position = event.position()
        x = round(position.x() / width * origin_width)
        y = round(position.y() / height * origin_height)
        label = int(event.mimeData().text())

        tooth_data = sorted(self.tooth_data, key=lambda tooth: (tooth['center'][0] - x) ** 2 + (tooth['center'][1] - y) ** 2)
        fdi = None
        for tooth in tooth_data:
            if abs(x - tooth['center'][0]) < tooth['box_size'][0] / 2 and abs(y - tooth['center'][1]) < tooth['box_size'][1] / 2:
                fdi = tooth['FDI']
                break
        if fdi is None:
            return

        if len(fdi) > 1:
            message_box = QMessageBox(self, text=f'Select FDI of Label {label}')
            message_box.setWindowTitle('Multiple FDI')
            message_box.setStandardButtons(QMessageBox.StandardButton.Cancel)
            for value in fdi:
                button = message_box.addButton(value, QMessageBox.ButtonRole.AcceptRole)
                button.setProperty('value', value)
            message_box.exec()

            button = message_box.clickedButton()
            fdi = button.property('value')
            if fdi is None:
                return
        else:
            fdi = fdi[0]

        self.fdi_updated.emit(label, fdi)

class TopLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.patient_selector = PatientSelector()
        self.addWidget(self.patient_selector)
        self.addStretch()

    def get_widgets(self):
        return self.patient_selector

class CentralLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.left_image = SourceImage()
        self.right_image = DestinationImage()
        for widget in [self.left_image, self.right_image]:
            self.addWidget(widget)

    def get_widgets(self):
        return self.left_image, self.right_image

class LabelGrid(QHBoxLayout):
    def __init__(self):
        super().__init__()

        self.left_grid = QGridLayout()
        self.right_grid = QGridLayout()

        self.addLayout(self.left_grid)
        self.addItem(QSpacerItem(30, 15))
        self.addLayout(self.right_grid)

    def update_labels(self, segmentation_labels, keypoint_labels):
        for grid in [self.left_grid, self.right_grid]:
            while grid.count():
                item = grid.takeAt(0)
                if item is None:
                    continue
                if isinstance(item, QSpacerItem):
                    del item
                    continue
                item.widget().deleteLater()

        results = [[], [], [], []]
        for i, (segmentation_label, keypoint_label) in enumerate(zip(segmentation_labels, keypoint_labels), 1):
            if len(segmentation_label) < len(keypoint_label):
                if i in {1, 4}:
                    keypoint_label = keypoint_label[-len(segmentation_label):]
                else:
                    keypoint_label = keypoint_label[:len(segmentation_label)]
            if len(segmentation_label) > len(keypoint_label):
                if i in {1, 4}:
                    keypoint_label = [None] * (len(segmentation_label) - len(keypoint_label)) + keypoint_label
                else:
                    keypoint_label = keypoint_label + [None] * (len(segmentation_label) - len(keypoint_label))
            for label, fdi in zip(segmentation_label, keypoint_label):
                results[i-1].append((label, fdi))

        if len(results[0]) < len(results[3]):
            results[0] = [(None, None) for _ in range(len(results[3]) - len(results[0]))] + results[0]
        if len(results[0]) > len(results[3]):
            results[3] = [(None, None) for _ in range(len(results[0]) - len(results[3]))] + results[3]
        if len(results[1]) < len(results[2]):
            results[1] = results[1] + [(None, None) for _ in range(len(results[2]) - len(results[1]))]
        if len(results[1]) > len(results[2]):
            results[2] = results[2] + [(None, None) for _ in range(len(results[1]) - len(results[2]))]

        label_editors = {}
        for i, result in enumerate(results):
            row = 0 if i in {0, 1} else 3
            grid = self.left_grid if i in {0, 3} else self.right_grid
            for index, (label, fdi) in enumerate(result):
                if label is None:
                    continue

                label_text = QLabel(str(label))
                fdi_editor = QLineEdit(None if fdi is None or len(fdi) > 1 else str(fdi[0]))
                grid.addWidget(label_text, row, index)
                grid.addWidget(fdi_editor, row + 1, index)
                label_editors[label] = fdi_editor

        self.left_grid.addItem(QSpacerItem(30, 15), 2, 0)
        self.right_grid.addItem(QSpacerItem(30, 15), 2, 0)

        return label_editors

class BottomLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.grid = LabelGrid()
        self.addLayout(self.grid)
        self.addStretch()
        self.save_button = QPushButton('Save')
        self.addWidget(self.save_button)

    def get_widgets(self):
        return self.grid, self.save_button

class MainWindow(MainWindowUI):
    def __init__(self, experiment_name, patients):
        super().__init__(TopLayout, CentralLayout, BottomLayout)
        self.experiment_name = experiment_name
        self.patients = patients

        self.setWindowTitle(experiment_name)

        self.patient_selector = self.layouts[0].get_widgets()
        self.left_image, self.right_image = self.layouts[1].get_widgets()
        self.grid, self.save_button = self.layouts[2].get_widgets()

        self.patient_selector.setup([f'{dataset}/{patient}' for dataset, patient in patients], self.on_patient_changed)

        self.right_image.fdi_updated.connect(self.on_fdi_update)
        self.save_button.clicked.connect(self.on_save)

        self.label_editors = None

    def showEvent(self, event):
        width = QApplication.primaryScreen().availableGeometry().width()
        self.setFixedWidth(width)
        self.centralWidget().layout().activate()
        self.adjustSize()

    def on_patient_changed(self, index):
        dataset, patient = self.patients[index]
        keypoint_panoramic, tooth_data = load_panoramic(dataset, patient)
        keypoint_panoramic = draw_panoramic(keypoint_panoramic, tooth_data)

        segmentation_volume_path = os.path.join('outputs', self.experiment_name, dataset, patient, f'{Mode.POST_PROCESSING}.npy')
        segmentation_volume = numpy.load(segmentation_volume_path)
        image_volume_path = os.path.join('outputs', self.experiment_name, dataset, patient, f'{Mode.IMAGE}.npy')
        image_volume = numpy.load(image_volume_path)

        jaw_labels, image_panoramic, segmentation_panoramic_mask = get_panoramic(image_volume, segmentation_volume)
        rate = 4
        image_panoramic = cv2.resize(image_panoramic, (0, 0), fx=rate, fy=rate)
        segmentation_panoramic_mask = cv2.resize(segmentation_panoramic_mask, (0, 0), fx=rate, fy=rate, interpolation=cv2.INTER_NEAREST)
        segmentation_panoramic = draw_label(image_panoramic, segmentation_panoramic_mask)

        target_height, target_width, _ = keypoint_panoramic.shape
        height, width, _ = segmentation_panoramic.shape
        padding_top = (target_height - height) // 2
        padding_left = (target_width - width) // 2
        padding_bottom = target_height - height - padding_top
        padding_right = target_width - width - padding_left
        segmentation_panoramic = cv2.copyMakeBorder(segmentation_panoramic, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT, value=0)
        segmentation_panoramic_mask = cv2.copyMakeBorder(segmentation_panoramic_mask, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT, value=0)

        pixmap = rgb2pixmap(segmentation_panoramic)
        self.left_image.update_image(pixmap, segmentation_panoramic_mask)

        pixmap = rgb2pixmap(keypoint_panoramic)
        self.right_image.update_image(pixmap, tooth_data)

        segmentation_labels = [[], [], [], []]
        for labels in jaw_labels:
            for label in labels:
                segmentation_labels[label//10-1].append(label)

        with JsonHandler(os.path.join('fdi', f'{self.experiment_name}.json'), FDIHandler) as handler:
            data = handler.data.get(f'{dataset}/{patient}')

        keypoint_labels = [[], [], [], []]
        if data is not None:
            for label, fdi in data.items():
                keypoint_labels[int(label)//10-1].append([int(fdi)])
        else:
            for tooth in tooth_data:
                fdi = int(tooth['FDI'][0])
                keypoint_labels[fdi//10-1].append([int(fdi) for fdi in tooth['FDI']])

        self.label_editors = self.grid.update_labels(segmentation_labels, keypoint_labels)

    def on_fdi_update(self, label, fdi):
        if self.label_editors is None:
            return

        self.label_editors[label].setText(fdi)

    def on_save(self):
        if self.label_editors is None:
            return

        data = {}
        exists = set()
        empty_labels = []
        duplicated_fdis = []
        for label, fdi_editor in self.label_editors.items():
            fdi = fdi_editor.text()
            if fdi == '-':
                continue

            if fdi == '':
                empty_labels.append(label)
            elif fdi in exists:
                duplicated_fdis.append(fdi)
            data[label] = fdi
            exists.add(fdi)

        messages = []
        if empty_labels:
            messages.append(f'Empty Labels: {", ".join([str(label) for label in empty_labels])}')
        if duplicated_fdis:
            messages.append(f'Duplicated FDI: {", ".join([str(fdi) for fdi in duplicated_fdis])}')

        if messages:
            QMessageBox.warning(self, 'Warning', '\n'.join(messages))
            return

        index = self.patient_selector.currentIndex()
        dataset, patient = self.patients[index]
        with JsonHandler(os.path.join('fdi', f'{self.experiment_name}.json'), FDIHandler) as handler:
            handler.set_data(f'{dataset}/{patient}', data)

if __name__ == '__main__':
    import os

    from argparse import ArgumentParser
    from scripts.post_processing.iterators import iterate_datasets

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
    patients = list(iterate_datasets(experiment_name, datasets))

    app = QApplication([])

    window = MainWindow(experiment_name, patients)

    window.show()

    app.exec()
