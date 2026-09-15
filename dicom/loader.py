import pydicom
import SimpleITK


def _read_dicom_dataset(dicom_dir):
    for file in dicom_dir.glob('*.dcm'):
        try:
            return pydicom.dcmread(file, stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:
            continue

    return None

def read_patient_id(dicom_dir):
    dcm = _read_dicom_dataset(dicom_dir)
    return None if dcm is None else dcm.PatientID

def read_patient_name(dicom_dir):
    dcm = _read_dicom_dataset(dicom_dir)
    return None if dcm is None else f'{dcm.PatientID}_{dcm.AcquisitionDate}'

def load_volume(volume_dir):
    reader = SimpleITK.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(volume_dir)
    filenames = [
        [
            filename
            for filename in reader.GetGDCMSeriesFileNames(volume_dir, series_id)
            if filename.endswith('.dcm')
        ]
        for series_id in series_ids
    ]
    filenames.sort(key=len, reverse=True)
    reader.SetFileNames(filenames[0])
    volume = reader.Execute()

    return volume
