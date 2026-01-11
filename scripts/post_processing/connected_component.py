import cc3d
import numpy

def filter_connected_component(volume, voxel_threshold, binary=False, keep=False):
    components = cc3d.connected_components(volume, connectivity=6)
    if binary:
        labels, counts = numpy.unique(components, return_counts=True)
        valid_labels = labels[(labels != 0) & (counts >= voxel_threshold)]
        return numpy.isin(components, valid_labels).astype(numpy.uint8)
    else:
        counts = numpy.bincount(components.ravel())
        valid = counts >= voxel_threshold
        valid[0] = False # background
        lookup_table = numpy.full_like(counts, -1 if keep else 0, dtype=numpy.int32)
        lookup_table[0] = 0 # background
        lookup_table[valid] = numpy.arange(1, valid.sum() + 1, dtype=numpy.int32)
        return lookup_table[components]

if __name__ == '__main__':
    import os

    from argparse import ArgumentParser
    from src.config import load_config
    from src.console import track
    from src.dataset import get_fold

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('target', type=int)
    parser.add_argument('--threshold', type=int, default=3500)
    parser.add_argument('--keep', action='store_true')
    args = parser.parse_args()

    experiment_name = args.exp
    target = args.target
    voxel_threshold = args.threshold
    keep = args.keep

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))

    for fold in range(1, config.num_folds + 1):
        _, valid_dataset_patients = get_fold(config.split_filename, fold)
        for dataset, patients in valid_dataset_patients.items():
            for patient in track(patients, desc=f'Fold {fold} {dataset}'):
                predict_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, 'volume.npy')
                volume = numpy.load(predict_path)
                volume = volume == target
                volume = filter_connected_component(volume, voxel_threshold, keep)

                cc_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'cc_volume_{target}.npy')
                numpy.save(cc_path, volume)
