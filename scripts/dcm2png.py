from pathlib import Path

from dicom.exporter import save_volume
from dicom.loader import collect_patients, load_volume
from dicom.metadata import get_info
from dicom.processor import process_volume
from tui.apps import PathsAndNumbersApp, parse_args
from utils.console import Table, track


def configure_parser(parser):
    parser.add_argument('dcm_root', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--min', type=int, default=-1000)
    parser.add_argument('--max', type=int, default=4500)
    parser.add_argument('--spacing', type=float, default=0.25)

args = parse_args(PathsAndNumbersApp, configure_parser)

dcm_root = Path(args['dcm_root'])
output_dir = Path(args['output_dir'])
min_value = args['min']
max_value = args['max']
new_spacing = args['spacing']

if output_dir.is_dir():
    raise FileExistsError(f'folder "{output_dir}" already exists')

with Table(['Patient', 'DCM Count', 'Image Count', 'Slice Thickness', 'Pixel Spacing', 'Origin Image Size', 'Image Size']) as table:
    patients = collect_patients(dcm_root)
    for patient, patient_id in track(patients.items()):
        origin_volume = load_volume(patient)
        volume = process_volume(origin_volume, min_value, max_value, new_spacing)
        save_volume(volume, output_dir / patient_id)

        origin_info = get_info(origin_volume)
        info = get_info(volume)
        table.add_row([patient_id, origin_info['count'], info['count'], origin_info['slice_thickness'], origin_info['pixel_spacing'], origin_info['size'], info['size']])
