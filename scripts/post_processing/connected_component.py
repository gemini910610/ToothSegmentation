import cc3d
import numpy

from scipy import ndimage
from scripts.tools.widgets import Label

def get_bounding_box(mask):
    xs = numpy.where(mask.any(axis=(1, 2)))[0]
    ys = numpy.where(mask.any(axis=(0, 2)))[0]
    zs = numpy.where(mask.any(axis=(0, 1)))[0]

    return xs[0], xs[-1] + 1, ys[0], ys[-1] + 1, zs[0], zs[-1] + 1

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

def filter_connected_component(volume, target, voxel_threshold=0):
    origin_shape = volume.shape
    x0, x1, y0, y1, z0, z1 = get_bounding_box(volume)

    x0 = max(x0 - 10, 0)
    y0 = max(y0 - 10, 0)
    z0 = max(z0 - 10, 0)

    x1 = min(x1 + 10, volume.shape[0])
    y1 = min(y1 + 10, volume.shape[1])
    z1 = min(z1 + 10, volume.shape[2])

    volume = volume[x0:x1, y0:y1, z0:z1]

    components = cc3d.connected_components(volume, connectivity=6, binary_image=True)
    components = cc3d.dust(components, voxel_threshold, connectivity=6, precomputed_ccl=True)
    if target == Label.TOOTH:
        volume = relabel(components)
    else:
        volume = cc3d.largest_k(components, k=3, precomputed_ccl=True)
        volume = (volume > 0).astype(numpy.uint8)
    origin_volume = numpy.zeros(origin_shape, dtype=numpy.uint8)
    origin_volume[x0:x1, y0:y1, z0:z1] = volume
    return origin_volume

def relabel(volume):
    labels = numpy.unique(volume)
    labels = labels[labels != 0]

    lookup_table = numpy.zeros(labels.max() + 1, dtype=numpy.uint8)
    lookup_table[labels] = numpy.arange(2, len(labels) + 2, dtype=numpy.uint8)

    volume = lookup_table[volume]
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
    parser.add_argument('target', type=int)
    parser.add_argument('--threshold', type=int, default=7500)
    args = parser.parse_args()

    experiment_name = args.exp
    target = args.target
    voxel_threshold = args.threshold

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    config.split_file_path = os.path.join('logs', experiment_name, f'{config.split_filename}.json')

    for fold in range(1, config.num_folds + 1):
        _, valid_dataset_patients = get_fold(config.split_file_path, fold)
        for dataset, patients in valid_dataset_patients.items():
            for patient in track(patients, desc=f'Fold {fold} {dataset}'):
                predict_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.PREDICT}.npy')
                volume = numpy.load(predict_path)
                volume = volume == target
                volume = filter_connected_component(volume, target, voxel_threshold)
                if target == Label.TOOTH:
                    volume = remove_outlier(volume, voxel_threshold)

                cc_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.TOOTH_CONNECTED_COMPONENT if target == Label.TOOTH else Mode.BONE_CONNECTED_COMPONENT}.npy')
                numpy.save(cc_path, volume)
