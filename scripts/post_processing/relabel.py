import numpy

def relabel_volume(volume):
    tooth_mask = volume > 1
    xs = numpy.nonzero(tooth_mask)[0]
    labels = volume[tooth_mask].astype(numpy.int32)

    max_label = labels.max()
    counts = numpy.bincount(labels, minlength=max_label + 1)
    sum_x = numpy.bincount(labels, weights=xs, minlength=max_label + 1)

    present = numpy.nonzero(counts)[0]
    present = present[present > 1]

    centroid_xs = sum_x[present] / counts[present]

    order = numpy.argsort(centroid_xs)
    labels = present[order]

    lookup_table = numpy.zeros(max_label + 1, dtype=numpy.int32)
    lookup_table[1] = 1 # keep bone label
    lookup_table[labels] = numpy.arange(2, len(labels) + 2, dtype=numpy.int32)

    volume = lookup_table[volume]
    return volume

if __name__ == '__main__':
    import os

    from argparse import ArgumentParser
    from src.config import load_config
    from src.console import track
    from src.dataset import get_fold

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
                volume_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, 'removed_volume.npy')
                volume = numpy.load(volume_path)
                volume = relabel_volume(volume)

                relabeled_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, 'relabeled_volume.npy')
                numpy.save(relabeled_path, volume)
