import pydicom
import SimpleITK


def read_patient_id(dicom_dir):
    for file in dicom_dir.glob('*.dcm'):
        try:
            dcm = pydicom.dcmread(file, stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:
            continue

        return dcm.PatientID

    return None

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
