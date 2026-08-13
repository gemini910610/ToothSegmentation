import pydicom
import SimpleITK


def collect_patients(root_dir):
    patients = {}
    for file in root_dir.rglob('*.dcm'):
        parent = file.parent
        if parent in patients:
            continue

        try:
            dcm = pydicom.dcmread(file, stop_before_pixels=True)
            patient_id = dcm.PatientID
        except pydicom.errors.InvalidDicomError:
            continue

        patients[parent] = patient_id

    return patients

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
