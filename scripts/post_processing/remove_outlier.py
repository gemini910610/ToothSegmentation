import numpy
import cc3d

from scipy import ndimage

def remove_outlier(volume, voxel_threshold=0):
    volume = volume.copy()
    component_count = volume.max()

    objects = ndimage.find_objects(volume, max_label=component_count)

    valid_labels = []
    z_centroids = []

    for label, slices in enumerate(objects, 1):
        if slices is None:
            continue

        roi = volume[slices] == label

        z0 = slices[2].start
        z1 = slices[2].stop

        zs = numpy.arange(z0, z1, dtype=numpy.float32)
        sum_z = roi.sum(axis=(0, 1))

        z_centroid = (zs * sum_z).sum() / roi.sum()

        valid_labels.append(label)
        z_centroids.append(z_centroid)

    valid_labels = numpy.array(valid_labels, dtype=numpy.uint8)
    z_centroids = numpy.array(z_centroids, dtype=numpy.float32)

    median = numpy.median(z_centroids)
    distance_z = numpy.abs(z_centroids - median)
    outlier_labels = valid_labels[distance_z > 150]

    lookup_table = numpy.zeros(component_count + 1, dtype=bool)
    lookup_table[outlier_labels] = True

    volume[lookup_table[volume]] = 0

    volume = cc3d.dust(volume, voxel_threshold, connectivity=6)

    return volume

def remove_cropped(volume):
    volume = volume.copy()
    component_count = volume.max()

    remove_labels = numpy.union1d(volume[:, :, :3], volume[:, :, -3:])
    remove_labels = remove_labels[remove_labels != 0]

    lookup_table = numpy.zeros(component_count + 1, dtype=bool)
    lookup_table[remove_labels] = True

    volume[lookup_table[volume]] = 0

    return volume

if __name__ == '__main__':
    import os

    from argparse import ArgumentParser
    from src.config import load_config
    from src.console import track
    from src.dataset import get_fold
    from scripts.tools.widgets import Mode

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('--threshold', type=int, default=7500)
    args = parser.parse_args()

    experiment_name = args.exp
    voxel_threshold = args.threshold

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    config.split_file_path = os.path.join('logs', experiment_name, f'{config.split_filename}.json')

    for fold in range(1, config.num_folds + 1):
        _, valid_dataset_patients = get_fold(config.split_file_path, fold)
        for dataset, patients in valid_dataset_patients.items():
            for patient in track(patients, desc=f'Fold {fold} {dataset}'):
                volume_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.REFINE}.npy')
                volume = numpy.load(volume_path)
                volume = remove_outlier(volume, voxel_threshold)
                volume = remove_cropped(volume)

                cleaned_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.CLEANED}.npy')
                numpy.save(cleaned_path, volume)
