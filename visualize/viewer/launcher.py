from pathlib import Path

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

from dicom.importer import DuplicatePatientError, import_dicom, import_panoramic
from dicom.loader import read_patient_id
from visualize.ui.layouts import HorizontalLayout
from visualize.ui.selectors import PatientSelector
from visualize.viewer.pages.bone_loss_page import BoneLossPage
from visualize.viewer.pages.bone_page import BonePage
from visualize.viewer.pages.exclude_page import ExcludePage
from visualize.viewer.pages.fdi_page import FDIPage
from visualize.viewer.pages.merge_page import MergePage
from visualize.viewer.pages.predict_page import PredictPage
from visualize.viewer.pages.split_page import SplitPage
from visualize.viewer.pages.tooth_page import ToothPage


class HeaderLayout(HorizontalLayout):
    def _init_widgets(self):
        return [
            PatientSelector(),
            QPushButton('Import'),
            None
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
        self.setWindowTitle('Segmentation Predict')
        self.working_dir = working_dir
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.header = HeaderLayout()
        self.patient_selector, self.import_button = self.header.widgets

        self.predict_page = PredictPage(model, config, working_dir)
        self.tooth_page = ToothPage(working_dir)
        self.split_page = SplitPage(working_dir)
        self.merge_page = MergePage(working_dir)
        self.exclude_page = ExcludePage(working_dir)
        self.fdi_page = FDIPage(panoramic_models, working_dir)
        self.bone_loss_page = BoneLossPage(cej_finder, panoramic_models, working_dir)
        self.bone_page = BonePage(working_dir)

        self.pages = [
            self.predict_page,
            self.bone_page,
            self.tooth_page,
            self.split_page,
            self.merge_page,
            self.exclude_page,
            self.fdi_page,
            self.bone_loss_page,
        ]

        self.tabs = TabWidget()
        self.tabs.addTab(self.predict_page, 'Predict')
        self.tabs.addTab(self.bone_page, 'Bone')
        self.tabs.addTab(self.tooth_page, 'Tooth')
        self.tabs.addTab(self.split_page, 'Split')
        self.tabs.addTab(self.merge_page, 'Merge')
        self.tabs.addTab(self.exclude_page, 'Exclude')
        self.tabs.addTab(self.fdi_page, 'FDI')
        self.tabs.addTab(self.bone_loss_page, 'Bone Loss')

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.addLayout(self.header)
        central_layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        patients = sorted(patient.name for patient in working_dir.iterdir() if (patient / 'image').is_dir())
        self.patient_selector.setup(patients, self._on_patient_changed)

        self.import_button.clicked.connect(self._on_import)

        self.pipeline_next = {
            self.predict_page: self.bone_page,
            self.bone_page: self.tooth_page,
            self.tooth_page: self.split_page,
            self.split_page: self.merge_page,
            self.merge_page: self.exclude_page,
            self.exclude_page: self.fdi_page,
            self.fdi_page: self.bone_loss_page,
        }
        for current_page, next_page in self.pipeline_next.items():
            current_page.next_requested.connect(lambda patient, next_page=next_page: self._on_page_next(patient, next_page))

        self.pipeline_fallback = {
            self.bone_page: [self.predict_page],
            self.tooth_page: [self.predict_page],
            self.split_page: [self.tooth_page],
            self.merge_page: [self.split_page],
            self.exclude_page: [self.merge_page],
            self.fdi_page: [self.exclude_page],
            self.bone_loss_page: [self.fdi_page],
        }

        self.current_patient = None

        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._on_tab_changed(self.tabs.currentIndex())

    def closeEvent(self, event):
        super().closeEvent(event)
        for page in self.pages:
            page.close()

    def _on_import(self):
        dicom_path = QFileDialog.getExistingDirectory(self, 'Select DICOM Folder')
        if not dicom_path:
            return

        dicom_dir = Path(dicom_path)

        patient_id = read_patient_id(dicom_dir)
        title = 'Select Panoramic Images' if patient_id is None else f'Select Panoramic Images ({patient_id})'
        image_paths, _ = QFileDialog.getOpenFileNames(self, title, filter='Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        image_paths = [Path(path) for path in image_paths]

        try:
            name = import_dicom(dicom_dir, self.working_dir)
        except DuplicatePatientError as error:
            QMessageBox.warning(self, 'Import Failed', f'Patient "{error.patient_name}" already exists, cannot import a duplicate\n({error.patient_dir})')
            return

        if name is None:
            QMessageBox.warning(self, 'Import Failed', 'No valid DICOM files found in the selected folder')
            return

        patient_dir = self.working_dir / name
        if image_paths:
            import_panoramic(image_paths, patient_dir)

        current_index = self.patient_selector.currentIndex()
        self.patient_selector.blockSignals(True)
        self.patient_selector.addItem(name)
        self.patient_selector.setCurrentIndex(current_index)
        self.patient_selector.blockSignals(False)

        message = f'DICOM converted to PNG: {patient_dir / "image"}'
        if image_paths:
            message += f'\nCopied {len(image_paths)} panoramic image(s): {patient_dir / "panoramic"}'
        QMessageBox.information(self, 'Import Complete', message)

    def _prompt_import_panoramic(self):
        message_box = QMessageBox(self)
        message_box.setWindowTitle('FDI')
        message_box.setText('No panoramic photo found for this patient. Import one before continuing to FDI.')
        select_button = message_box.addButton('Select Panoramic Photo', QMessageBox.ButtonRole.ActionRole)
        message_box.addButton(QMessageBox.StandardButton.Ok)
        message_box.exec()

        if message_box.clickedButton() is not select_button:
            return False

        title = f'Select Panoramic Images ({self.current_patient})'
        image_paths, _ = QFileDialog.getOpenFileNames(self, title, filter='Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if not image_paths:
            return False

        image_paths = [Path(path) for path in image_paths]
        patient_dir = self.working_dir / self.current_patient
        import_panoramic(image_paths, patient_dir)

        return True

    def _on_patient_changed(self, index):
        self.current_patient = self.patient_selector.currentText()
        self._refresh_current_page()

    def _refresh_current_page(self):
        page = self.tabs.currentWidget()
        self._sync_page(page)

        if not self.current_patient:
            return

        if page.is_ready:
            return

        if page is self.fdi_page and self.fdi_page.missing_panoramic:
            if self._prompt_import_panoramic() and self._sync_page(page):
                return

        fallback_pages = self.pipeline_fallback.get(page)
        if not fallback_pages:
            return

        target = fallback_pages[-1]
        for fallback_page in fallback_pages:
            if not self._sync_page(fallback_page):
                target = fallback_page
                break

        self._sync_page(target)
        self.tabs.setCurrentWidget(target)

    def _sync_page(self, page):
        if page.patient != self.current_patient or not page.is_ready:
            page.set_patient(self.current_patient)

        return page.is_ready

    def _on_page_next(self, patient, next_page):
        self.current_patient = patient
        next_page.set_patient(patient)
        self.tabs.setCurrentWidget(next_page)

    def _on_tab_changed(self, index):
        self._refresh_current_page()

        self.tabs.updateGeometry()
        self.centralWidget().layout().activate()
        QTimer.singleShot(0, lambda: QTimer.singleShot(0, lambda: self.resize(self.sizeHint())))

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
