from collections import defaultdict
from pathlib import Path

import pydicom

from tui.apps import DirectoryApp, parse_args
from utils.console import Table, track


class PatientMismatchError(Exception):
    def __init__(self, image, value):
        self.image = image
        self.value = value

class DateMismatchError(Exception):
    def __init__(self, image, value):
        self.image = image
        self.value = value

class NeedRenameError(Exception):
    def __init__(self, folder, value):
        self.folder = folder
        self.value = value

def check_xray(patient_folder, image_folder, image, is_pano):
    patient, date = image.stem.split('_')[:2]
    if patient_folder.name != patient:
        raise PatientMismatchError(image, patient)

    if date not in image_folder.name:
        raise DateMismatchError(image, date)

    folder_name = f'{date} {"Pano" if is_pano else "PA"}'
    if image_folder.name != folder_name:
        raise NeedRenameError(image_folder, folder_name)

def check_cbct(patient_folder, image_folder, image):
    dcm = pydicom.dcmread(image, stop_before_pixels=True)
    patient = dcm.PatientID
    date = dcm.AcquisitionDate
    if patient_folder.name != patient:
        raise PatientMismatchError(image, patient)

    if date not in image_folder.name:
        raise DateMismatchError(image, date)

    folder_name = f'{date} CBCT'
    if image_folder.name != folder_name:
        raise NeedRenameError(image_folder, folder_name)

def configure_parser(parser):
    parser.add_argument('data_dir', type=str)

args = parse_args(DirectoryApp, configure_parser, rename={'directory': 'data_dir'})

data_dir = Path(args['data_dir'])

patient_mismatches = set()
date_mismatches = set()
need_renames = set()

for folder in data_dir.iterdir():
    if not folder.is_dir():
        continue

    for patient_folder in track(list(folder.iterdir()), desc=folder.name):
        for image_folder in patient_folder.iterdir():
            image_types = defaultdict(set)
            for image in image_folder.iterdir():
                stem = image.stem
                if image.suffix == '.dcm':
                    image_types['.dcm'].add(image)
                elif 'IO' in stem:
                    image_types['IO'].add(image)
                elif 'PX' in stem:
                    image_types['PX'].add(image)

            try:
                if image_types['IO']:
                    for image in image_types['IO']:
                        check_xray(patient_folder, image_folder, image, is_pano=False)
                elif image_types['PX']:
                    for image in image_types['PX']:
                        check_xray(patient_folder, image_folder, image, is_pano=True)
                elif len(image_types['.dcm']) > 8:
                    for image in image_types['.dcm']:
                        break
                    check_cbct(patient_folder, image_folder, image)
            except PatientMismatchError as exception:
                patient_mismatches.add((exception.image, exception.value))
            except DateMismatchError as exception:
                date_mismatches.add((exception.image, exception.value))
            except NeedRenameError as exception:
                need_renames.add((exception.folder, exception.value))

if patient_mismatches:
    Table(
        ['Patient Mismatch', 'Got'],
        *[
            (image.relative_to(data_dir), value)
            for image, value in sorted(patient_mismatches)
        ]
    ).display()

if date_mismatches:
    Table(
        ['Date Mismatch', 'Got'],
        *[
            (image.relative_to(data_dir), value)
            for image, value in sorted(date_mismatches)
        ]
    ).display()

if need_renames:
    with Table(['Folder', 'Rename']) as table:
        for folder, value in track(sorted(need_renames), desc='rename'):
            rename = folder.with_name(value)
            folder.rename(rename)
            table.add_row([folder.relative_to(data_dir), rename.relative_to(data_dir)])

if not patient_mismatches and not date_mismatches and not need_renames:
    print('\033[92mOK\033[0m')
