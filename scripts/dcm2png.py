import cv2
import numpy
import os
import re
import SimpleITK

from pathlib import Path

def nature_path_key(path, root_dir):
    relative_path = str(path.relative_to(root_dir))
    return tuple(
        int(x) if x.isdigit() else x
        for x in re.findall(r'\d+|\D+', relative_path)
    )

def collect_patients(root_dir, exclude_rule=None):
    root_dir = Path(root_dir)
    patients = set()
    for file in root_dir.rglob('*.dcm'):
        parent = file.parent
        if exclude_rule is not None and exclude_rule(parent):
            continue
        patients.add(parent)
    return sorted(patients, key=lambda x: nature_path_key(x, root_dir))

def load_volume(volume_dir):
    reader = SimpleITK.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(volume_dir)
    if len(series_ids) == 0:
        raise ValueError(f'No DICOM series found in "{volume_dir}"')
    series_id = series_ids[0]
    filenames = reader.GetGDCMSeriesFileNames(volume_dir, series_id)
    filenames = [filename for filename in filenames if filename.endswith('.dcm')]
    reader.SetFileNames(filenames)
    volume = reader.Execute()
    return volume

def preprocess_volume(volume, min_value=-1000, max_value=4500):
    volume = SimpleITK.IntensityWindowing(volume, min_value, max_value) # [-1000, 4500] -> [0, 255]
    volume = SimpleITK.Cast(volume, SimpleITK.sitkUInt8)
    return volume

def resample_volume(volume, new_spacing=[0.25, 0.25, 0.25]):
    size = numpy.array(volume.GetSize())
    spacing = numpy.array(volume.GetSpacing())
    new_spacing = numpy.array(new_spacing)
    new_size = size * spacing / new_spacing

    new_size = [int(size) for size in new_size]
    new_spacing = [float(spacing) for spacing in new_spacing]

    volume = SimpleITK.Flip(volume, [False, False, True])

    resample = SimpleITK.ResampleImageFilter()
    resample.SetReferenceImage(volume)
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(SimpleITK.sitkLinear)

    volume = resample.Execute(volume)
    volume = SimpleITK.Flip(volume, [False, False, True])
    return volume

def save_volume(volume, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    volume = SimpleITK.GetArrayFromImage(volume)
    for index, image in enumerate(volume, 1):
        image_path = os.path.join(output_dir, f'{index}.png')
        cv2.imwrite(image_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def get_info(origin_volume, volume):
    _, origin_image_size, dcm_count = origin_volume.GetSize()
    _, pixel_spacing, slice_thickness = origin_volume.GetSpacing()
    _, image_size, image_count = volume.GetSize()
    return dcm_count, image_count, slice_thickness, pixel_spacing, origin_image_size, image_size

if __name__ == '__main__':
    from argparse import ArgumentParser

    from src.console import Table, track

    parser = ArgumentParser()
    parser.add_argument('dcm', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--min', type=int, default=-1000)
    parser.add_argument('--max', type=int, default=4500)
    parser.add_argument('--spacing', type=float, default=0.25)
    args = parser.parse_args()

    dcm_root = args.dcm
    output_dir = args.output
    min_value = args.min
    max_value = args.max
    new_spacing = [args.spacing, args.spacing, args.spacing]

    if os.path.exists(output_dir):
        raise FileExistsError(f'folder "{output_dir}" already exists')

    with Table(['Data ID', 'Source Path', 'DCM Count', 'Image Count', 'Slice Thickness', 'Pixel Spacing', 'Origin Image Size', 'Image Size']) as table:
        patients = collect_patients(dcm_root, lambda x: 'IOS' in str(x))
        for index, patient in enumerate(track(patients), 1):
            data = re.search(r'data_(\d+)', str(patient))
            if data is not None:
                index = data.group(1)
            origin_volume = load_volume(patient)
            volume = preprocess_volume(origin_volume, min_value, max_value)
            volume = resample_volume(volume, new_spacing)
            info = get_info(origin_volume, volume)
            save_volume(volume, f'{output_dir}/data_{index}')
            table.add_row([f'data_{index}', patient.relative_to(dcm_root), *info])
