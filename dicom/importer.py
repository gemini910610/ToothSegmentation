import shutil

from dicom.exporter import save_volume
from dicom.loader import load_volume, read_patient_id
from dicom.processor import process_volume


class DuplicatePatientError(Exception):
    def __init__(self, patient_id, patient_dir):
        super().__init__(f'patient "{patient_id}" already exists at "{patient_dir}"')
        self.patient_id = patient_id
        self.patient_dir = patient_dir

def import_dicom(dicom_dir, working_dir, min_value=-1000, max_value=4500, new_spacing=None):
    patient_id = read_patient_id(dicom_dir)
    if patient_id is None:
        return None

    patient_dir = working_dir / patient_id
    if patient_dir.exists():
        raise DuplicatePatientError(patient_id, patient_dir)

    volume = load_volume(dicom_dir)
    volume = process_volume(volume, min_value, max_value, new_spacing)
    save_volume(volume, patient_dir / 'image')

    return patient_id

def import_panoramic(image_paths, patient_dir):
    panoramic_dir = patient_dir / 'panoramic'
    panoramic_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        shutil.copy2(path, panoramic_dir / path.name)
