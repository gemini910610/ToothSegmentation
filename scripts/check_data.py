from collections import defaultdict
from pathlib import Path

import pandas
import pydicom
from pandas import DataFrame, ExcelWriter

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

sheets = defaultdict(list)

for folder in data_dir.iterdir():
    if not folder.is_dir():
        continue

    for patient_folder in track(folder.iterdir(), desc=folder.name):
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

        cbct_folders = patient_folder.glob('* CBCT', case_sensitive=True)
        pano_folders = patient_folder.glob('* Pano', case_sensitive=True)
        pa_folders = patient_folder.glob('* PA', case_sensitive=True)

        pano_date = ', '.join(pano_folder.name[:8] for pano_folder in pano_folders)
        if not pano_date:
            pano_date = None
        elif ', ' not in pano_date:
            pano_date = int(pano_date)

        pa_date = ', '.join(pa_folder.name[:8] for pa_folder in pa_folders)
        if not pa_date:
            pa_date = None
        elif ', ' not in pa_date:
            pa_date = int(pa_date)

        sheets[folder.name].extend(
            {
                'Patient': int(patient_folder.name),
                'CBCT Date': int(cbct_folder.name[:8]),
                'Pano Date': pano_date,
                'PA Date': pa_date
            }
            for cbct_folder in cbct_folders
        )

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

excel_path = Path('datasets') / 'data.xlsx'
excel_path.parent.mkdir(parents=True, exist_ok=True)
if not patient_mismatches and not date_mismatches and not need_renames:
    for sheet_name, sheet in sheets.items():
        sheet = DataFrame(sheet)

        try:
            origin_sheet = pandas.read_excel(excel_path, sheet_name=sheet_name, engine='openpyxl')
            sheet = pandas.concat([origin_sheet, sheet], ignore_index=True)
            sheet = sheet.drop_duplicates(subset=['Patient', 'CBCT Date'])
        except (FileNotFoundError, ValueError):
            pass

        file_exists = excel_path.is_file()
        with ExcelWriter(excel_path, engine='openpyxl', mode='a' if file_exists else 'w', if_sheet_exists='replace' if file_exists else None) as writer:
            sheet.to_excel(writer, sheet_name=sheet_name, index=False)
