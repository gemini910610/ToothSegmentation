from pathlib import Path

import cc3d
import numpy
from PySide6.QtCore import QSize, QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from constants.label import Label
from dicom.importer import DuplicatePatientError, import_dicom, import_panoramic
from post_processing.connected_component import filter_connected_component
from post_processing.fill_holes import fill_holes
from post_processing.panoramic import find_panoramic_image
from post_processing.relabel import relabel_volume
from post_processing.watershed import split_component
from segmentation.predict import predict_patient
from utils.console import track
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.selectors import PatientSelector
from visualize.viewer.pages.bone_loss_page import BoneLossPage
from visualize.viewer.pages.exclude_page import ExcludePage
from visualize.viewer.pages.fdi_page import FDIPage
from visualize.viewer.pages.merge_page import MergePage
from visualize.viewer.pages.split_page import SplitPage
from visualize.viewer.pipeline import (
    BONE_FILENAME,
    IMAGE_FILENAME,
    PREDICT_FILENAME,
    TOOTH_FILENAME,
    auto_exclude,
)
from visualize.viewer.tooth_data import save_tooth_data

BONE_THRESHOLD = 45000
TOOTH_THRESHOLD = 8200

class AutoPipeline:
    def __init__(self, model, config, patient_dir):
        self.model = model
        self.config = config
        self.patient_dir = patient_dir

    def __len__(self):
        return 8

    def __iter__(self):
        save_tooth_data(self.patient_dir, {})

        image_dir = self.patient_dir / 'image'
        image_volume, predict_volume = predict_patient(self.model, image_dir, self.config)
        numpy.save(self.patient_dir / IMAGE_FILENAME, image_volume)
        numpy.save(self.patient_dir / PREDICT_FILENAME, predict_volume)
        yield

        bone_cc_volume = filter_connected_component(predict_volume == Label.BONE, is_bone=True, voxel_threshold=BONE_THRESHOLD)
        yield

        tooth_cc_volume = filter_connected_component(predict_volume == Label.TOOTH, is_bone=False, voxel_threshold=TOOTH_THRESHOLD)
        yield

        watershed_volume = split_component(tooth_cc_volume)
        yield

        truncated_labels = numpy.union1d(watershed_volume[:, :, :3], watershed_volume[:, :, -3:])
        truncated_labels = [int(label) for label in truncated_labels if label > 0]
        watershed_volume = cc3d.dust(watershed_volume, 1000, connectivity=6, precomputed_ccl=True)
        yield

        remaining_labels = {int(label) for label in numpy.unique(watershed_volume)}
        tooth_data = {label: 'Truncated' for label in truncated_labels if label in remaining_labels}
        tooth_data.update(auto_exclude(watershed_volume, bone_cc_volume, excluded_ids=set(tooth_data)))
        yield

        tooth_filled, bone_filled = fill_holes(watershed_volume, bone_cc_volume)
        numpy.save(self.patient_dir / BONE_FILENAME, bone_filled)
        yield

        _, positions = relabel_volume(tooth_filled, bone_filled, excluded_ids=set(tooth_data))
        tooth_data.update(positions)
        save_tooth_data(self.patient_dir, tooth_data)
        numpy.save(self.patient_dir / TOOTH_FILENAME, tooth_filled)
        yield

class HeaderLayout(HorizontalLayout):
    def _init_widgets(self):
        return [
            PatientSelector(),
            QPushButton('Import'),
            QPushButton('Predict'),
            None,
        ]

class TabWidget(QTabWidget):
    def sizeHint(self):
        current = self.currentWidget()
        if current is None:
            return super().sizeHint()

        tab_bar_size = self.tabBar().sizeHint()
        content_size = current.sizeHint()

        return QSize(
            max(tab_bar_size.width(), content_size.width()),
            tab_bar_size.height() + content_size.height()
        )

    def minimumSizeHint(self):
        return self.sizeHint()

class MainWindow(QMainWindow):
    def __init__(self, model, config, cej_finder, panoramic_models, working_dir=Path('working')):
        super().__init__()
        self.move(0, 0)
        self.setWindowTitle('Automated Segmentation Pipeline')
        self.model = model
        self.config = config
        self.working_dir = working_dir
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.header = HeaderLayout()
        self.patient_selector, self.import_button, self.predict_button = self.header.widgets

        self.split_page = SplitPage(working_dir)
        self.merge_page = MergePage(working_dir)
        self.exclude_page = ExcludePage(working_dir)
        self.fdi_page = FDIPage(panoramic_models, working_dir)
        self.bone_loss_page = BoneLossPage(cej_finder, panoramic_models, working_dir)
        self.pages = (self.split_page, self.merge_page, self.exclude_page, self.fdi_page, self.bone_loss_page)

        self.tabs = TabWidget()
        self.tabs.addTab(self.bone_loss_page, 'Bone Loss')
        self.tabs.addTab(self.exclude_page, 'Exclude')
        self.tabs.addTab(self.split_page, 'Split')
        self.tabs.addTab(self.merge_page, 'Merge')
        self.tabs.addTab(self.fdi_page, 'FDI')

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.addLayout(self.header)
        central_layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        patients = sorted(patient.name for patient in working_dir.iterdir() if (patient / 'image').is_dir())
        self.patient_selector.setup(patients, self._on_patient_changed)

        self.import_button.clicked.connect(self._on_import)
        self.predict_button.clicked.connect(self._on_predict)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.current_patient = None
        self._previous_page = self.tabs.currentWidget()

    def closeEvent(self, event):
        self._previous_page.finalize()

        super().closeEvent(event)
        for page in self.pages:
            page.close()

    def _on_import(self):
        dicom_path = QFileDialog.getExistingDirectory(self, 'Select DICOM Folder')
        if not dicom_path:
            return

        dicom_dir = Path(dicom_path)

        try:
            name = import_dicom(dicom_dir, self.working_dir)
        except DuplicatePatientError as error:
            QMessageBox.warning(self, 'Import Failed', f'Patient "{error.patient_name}" already exists, cannot import a duplicate\n({error.patient_dir})')
            return

        if name is None:
            QMessageBox.warning(self, 'Import Failed', 'No valid DICOM files found in the selected folder')
            return

        patient_dir = self.working_dir / name

        current_index = self.patient_selector.currentIndex()
        self.patient_selector.blockSignals(True)
        self.patient_selector.addItem(name)
        self.patient_selector.setCurrentIndex(current_index)
        self.patient_selector.blockSignals(False)

        QMessageBox.information(self, 'Import Complete', f'DICOM converted to PNG: {patient_dir / "image"}')

    def _refresh_all_pages(self):
        for page in self.pages:
            page.set_patient(self.current_patient)
        self.tabs.setCurrentWidget(self.bone_loss_page)

    def _on_patient_changed(self, index):
        self._previous_page.finalize()

        self.current_patient = self.patient_selector.currentText()
        self._refresh_all_pages()

    def _on_predict(self):
        if not self.current_patient:
            return

        patient_dir = self.working_dir / self.current_patient
        try:
            pipeline = AutoPipeline(self.model, self.config, patient_dir)
            for _ in track(pipeline, desc=self.current_patient):
                pass
        except Exception as exception:
            QMessageBox.warning(self, 'Predict Failed', str(exception))
            return

        self._refresh_all_pages()

    def _on_tab_changed(self, index):
        current = self.tabs.currentWidget()

        if current is self.fdi_page and self.current_patient and find_panoramic_image(self.working_dir, self.current_patient) is None and not self._prompt_import_panoramic():
            self.tabs.blockSignals(True)
            self.tabs.setCurrentWidget(self._previous_page)
            self.tabs.blockSignals(False)

            self._resize_to_fit_content()
            return

        was_dirty = self._previous_page.dirty
        recompute = current not in (self.split_page, self.merge_page)
        self._previous_page.finalize(recompute=recompute)
        if was_dirty:
            for page in self.pages:
                page.patient = None

        if current.patient != self.current_patient:
            current.set_patient(self.current_patient)
        self._previous_page = current

        self._resize_to_fit_content()

    def _resize_to_fit_content(self):
        self.tabs.updateGeometry()
        self.centralWidget().layout().activate()
        QTimer.singleShot(0, lambda: QTimer.singleShot(0, lambda: self.resize(self.sizeHint())))

    def _prompt_import_panoramic(self):
        patient_id = self.current_patient.split('_')[0]
        title = f'Select Panoramic Images ({patient_id})'
        image_paths, _ = QFileDialog.getOpenFileNames(self, title, filter='Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if not image_paths:
            return False

        image_paths = [Path(path) for path in image_paths]
        patient_dir = self.working_dir / self.current_patient
        import_panoramic(image_paths, patient_dir)

        return True

if __name__ == '__main__':
    import sys

    from keypoints.models import load_model as load_keypoints_model
    from keypoints.predict import CEJFinder
    from panoramic.predict import PanoramicModels
    from scripts.optimize_batch_size import get_optimal_batch_size
    from segmentation.models import load_model as load_segmentation_model
    from tui.apps import DualExperimentDualFoldApp
    from utils.config import load_config
    from visualize.utils import run_app

    args = DualExperimentDualFoldApp().run()
    if args is None:
        sys.exit()

    segmentation_experiment_name = args['segmentation_exp']
    segmentation_fold = args['segmentation_fold']
    keypoints_experiment_name = args['keypoints_exp']
    keypoints_fold = args['keypoints_fold']

    config = load_config(Path('logs') / segmentation_experiment_name / 'segmentation.toml', fold=segmentation_fold)
    config.batch_size = get_optimal_batch_size('segmentation', is_train=False)
    segmentation_model = load_segmentation_model(config)

    keypoints_config = load_config(Path('logs') / keypoints_experiment_name / 'keypoints.toml', fold=keypoints_fold)
    keypoints_model = load_keypoints_model(keypoints_config)
    cej_finder = CEJFinder(keypoints_model, keypoints_config.device)

    panoramic_weights_dir = Path('panoramic') / 'weights'
    panoramic_models = PanoramicModels(
        panoramic_weights_dir / 'pano_best32type_nolr.pt',
        panoramic_weights_dir / 'pano_transpose_v2_5p.pkl',
        panoramic_weights_dir / 'pano_transpose_v2_6p.pkl',
    )

    working_dir = Path('working') / f'{segmentation_experiment_name}_{keypoints_experiment_name}'

    run_app(MainWindow, segmentation_model, config, cej_finder, panoramic_models, working_dir)
