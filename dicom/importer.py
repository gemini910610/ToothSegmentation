import shutil

from dicom.exporter import save_volume
from dicom.loader import load_volume, read_patient_name
from dicom.processor import process_volume


class DuplicatePatientError(Exception):
    def __init__(self, patient_name, patient_dir):
        super().__init__(f'patient "{patient_name}" already exists at "{patient_dir}"')
        self.patient_name = patient_name
        self.patient_dir = patient_dir

def import_dicom(dicom_dir, working_dir, min_value=-1000, max_value=4500, new_spacing=None):
    patient_name = read_patient_name(dicom_dir)
    if patient_name is None:
        return None

    patient_dir = working_dir / patient_name
    if patient_dir.is_dir():
        raise DuplicatePatientError(patient_name, patient_dir)

    volume = load_volume(dicom_dir)
    volume = process_volume(volume, min_value, max_value, new_spacing)
    save_volume(volume, patient_dir / 'image')

    return patient_name

def import_panoramic(image_paths, patient_dir):
    panoramic_dir = patient_dir / 'panoramic'
    panoramic_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        shutil.copy2(path, panoramic_dir / path.name)
