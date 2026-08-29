from pathlib import Path

import pandas
from pandas import ExcelWriter

from dicom.exporter import save_volume
from dicom.loader import load_volume
from dicom.metadata import get_info
from dicom.processor import process_volume
from tui.apps import PathsAndNumbersApp, parse_args
from utils.console import Table, track
from utils.image import imread


def configure_parser(parser):
    parser.add_argument('dcm_root', type=str)
    parser.add_argument('--min', type=int, default=-1000)
    parser.add_argument('--max', type=int, default=4500)
    parser.add_argument('--spacing', type=float, default=0.25)

args = parse_args(PathsAndNumbersApp, configure_parser)

dcm_root = Path(args['dcm_root'])
min_value = args['min']
max_value = args['max']
new_spacing = args['spacing']

output_dir = Path('test_datasets')
excel_path = output_dir / 'data.xlsx'

with Table(['Dataset', 'Patient', 'CBCT Date', 'DCM Count', 'Image Count', 'Slice Thickness', 'Pixel Spacing', 'Origin Image Size', 'Image Size']) as table:
    for folder in dcm_root.iterdir():
        if not folder.is_dir():
            continue

        excel = pandas.read_excel(excel_path, sheet_name=folder.name, engine='openpyxl')

        for patient_folder in track(folder.iterdir(), desc=folder.name):
            cbct_folders = patient_folder.glob('* CBCT', case_sensitive=True)
            for cbct_folder in cbct_folders:
                origin_volume = load_volume(cbct_folder)
                volume = process_volume(origin_volume, min_value, max_value, new_spacing)
                patient_date = f'{patient_folder.name}_{cbct_folder.name[:8]}'

                origin_info = get_info(origin_volume)
                info = get_info(volume)

                output_image_dir = output_dir / folder.name / patient_date / 'image'
                dir_exists = output_image_dir.is_dir()
                if dir_exists:
                    images = list(output_image_dir.iterdir())
                    info['count'] = len(images)
                    info['size'] = imread(images[0]).shape[0]
                else:
                    save_volume(volume, output_image_dir)

                table.add_row([folder.name, patient_folder.name, cbct_folder.name[:8], origin_info['count'], info['count'], origin_info['slice_thickness'], origin_info['pixel_spacing'], origin_info['size'], info['size']])

                mask = (excel['Patient'] == int(patient_folder.name)) & (excel['CBCT Date'] == int(cbct_folder.name[:8]))
                data = {
                    'DCM Count': origin_info['count'],
                    'Image Count': info['count'],
                    'Slice Thickness': origin_info['slice_thickness'],
                    'Pixel Spacing': origin_info['pixel_spacing'],
                    'Origin Image Size': origin_info['size'],
                    'Image Size': info['size']
                }
                excel.loc[mask, data.keys()] = data.values()

        with ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            excel.to_excel(writer, sheet_name=folder.name, index=False)
