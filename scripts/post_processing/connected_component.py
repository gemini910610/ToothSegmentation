import cc3d
import numpy

def filter_connected_component(volume, voxel_threshold, binary=False, keep=False):
    origin_shape = volume.shape
    coordinates = numpy.flatnonzero(volume)
    coordinates = numpy.column_stack(numpy.unravel_index(coordinates, volume.shape))
    x0, y0, z0 = coordinates.min(axis=0)
    x1, y1, z1 = coordinates.max(axis=0) + 1

    x0 = max(x0 - 10, 0)
    y0 = max(y0 - 10, 0)
    z0 = max(z0 - 10, 0)

    x1 = min(x1 + 10, volume.shape[0])
    y1 = min(y1 + 10, volume.shape[1])
    z1 = min(z1 + 10, volume.shape[2])

    volume = volume[x0:x1, y0:y1, z0:z1]

    components = cc3d.connected_components(volume, connectivity=6)
    if binary:
        labels, counts = numpy.unique(components, return_counts=True)
        valid_labels = labels[(labels != 0) & (counts >= voxel_threshold)]
        volume = numpy.isin(components, valid_labels).astype(numpy.uint8)
    else:
        counts = numpy.bincount(components.ravel())
        valid = counts >= voxel_threshold
        valid[0] = False # background
        lookup_table = numpy.full_like(counts, -1 if keep else 0, dtype=numpy.int32)
        lookup_table[0] = 0 # background
        lookup_table[valid] = numpy.arange(1, valid.sum() + 1, dtype=numpy.int32)
        volume = lookup_table[components]

    origin_volume = numpy.zeros(origin_shape, dtype=numpy.int32)
    origin_volume[x0:x1, y0:y1, z0:z1] = volume
    return origin_volume

if __name__ == '__main__':
    import os

    from argparse import ArgumentParser
    from src.config import load_config
    from src.console import track
    from src.dataset import get_fold

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('target', type=int)
    parser.add_argument('--threshold', type=int, default=7500)
    parser.add_argument('--keep', action='store_true')
    args = parser.parse_args()

    experiment_name = args.exp
    target = args.target
    voxel_threshold = args.threshold
    keep = args.keep

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    config.split_file_path = os.path.join('logs', experiment_name, f'{config.split_filename}.json')

    for fold in range(1, config.num_folds + 1):
        _, valid_dataset_patients = get_fold(config.split_file_path, fold)
        for dataset, patients in valid_dataset_patients.items():
            for patient in track(patients, desc=f'Fold {fold} {dataset}'):
                predict_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, 'volume.npy')
                volume = numpy.load(predict_path)
                volume = volume == target
                volume = filter_connected_component(volume, voxel_threshold, keep=keep)

                cc_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'cc_volume_{target}.npy')
                numpy.save(cc_path, volume)
