import shutil
from pathlib import Path

import pandas

from tui.apps import DirectoryApp, parse_args
from utils.console import Table, track


def configure_parser(parser):
    parser.add_argument('raw_root', type=str)

args = parse_args(DirectoryApp, configure_parser, rename={'directory': 'raw_root'})

raw_root = Path(args['raw_root'])

output_dir = Path('datasets')
excel_path = output_dir / 'data.xlsx'

with Table(['Dataset', 'Patient', 'Pano Date']) as table:
    for folder in raw_root.iterdir():
        if not folder.is_dir():
            continue

        excel = pandas.read_excel(excel_path, sheet_name=folder.name, engine='openpyxl')
        rows = excel.dropna(subset=['Pano Date']).drop_duplicates(subset=['Patient', 'Pano Date'])
        rows[['Patient', 'Pano Date']] = rows[['Patient', 'Pano Date']].astype(int).astype(str)

        output_group_dir = output_dir / folder.name

        for _, row in track(rows.iterrows(), desc=folder.name):
            patient = row['Patient']
            patient_folder = folder / patient

            date = row['Pano Date']
            pano_folder = patient_folder / f'{date} Pano'

            output_pano_dir = output_group_dir / patient / f'{date} Pano'
            for image in pano_folder.iterdir():
                break

            output_pano_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, output_pano_dir / image.name)

            table.add_row([folder.name, patient, date])
