import numpy

from scipy import ndimage
from scripts.post_processing.connected_component import get_bounding_box

def fill_holes(tooth_volume, bone_volume):
    tooth_volume = tooth_volume.copy()
    bone_volume = bone_volume.copy()

    objects = ndimage.find_objects(tooth_volume)

    for label, slices in enumerate(objects, 1):
        if slices is None:
            continue

        roi = tooth_volume[slices]
        mask = roi == label
        mask = ndimage.binary_fill_holes(mask)
        roi[mask] = label

    volume = (bone_volume > 0) | (tooth_volume > 0)
    x0, x1, y0, y1, z0, z1 = get_bounding_box(volume)
    mask = volume[x0:x1, y0:y1, z0:z1]
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask
    roi = bone_volume[x0:x1, y0:y1, z0:z1]
    roi[holes] = 1

    return tooth_volume, bone_volume

if __name__ == '__main__':
    import os

    from argparse import ArgumentParser
    from src.config import load_config
    from src.console import track
    from src.dataset import get_fold
    from scripts.tools.widgets import Mode

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    config.split_file_path = os.path.join('logs', experiment_name, f'{config.split_filename}.json')

    for fold in range(1, config.num_folds + 1):
        _, valid_dataset_patients = get_fold(config.split_file_path, fold)
        for dataset, patients in valid_dataset_patients.items():
            for patient in track(patients, desc=f'Fold {fold} {dataset}'):
                output_dir = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient)
                tooth_volume_path = os.path.join(output_dir, f'{Mode.REMOVED}.npy')
                bone_volume_path = os.path.join(output_dir, f'{Mode.BONE_CONNECTED_COMPONENT}.npy')
                tooth_volume = numpy.load(tooth_volume_path)
                bone_volume = numpy.load(bone_volume_path)

                tooth_volume, bone_volume = fill_holes(tooth_volume, bone_volume)

                tooth_filled_path = os.path.join(output_dir, f'{Mode.TOOTH_FILLED}.npy')
                bone_filled_path = os.path.join(output_dir, f'{Mode.BONE_FILLED}.npy')
                numpy.save(tooth_filled_path, tooth_volume)
                numpy.save(bone_filled_path, bone_volume)
