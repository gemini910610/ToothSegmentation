import numpy

from scripts.post_processing.watershed import split_k_component

def refine_component(source_volume, destination_volume, tasks):
    new_volume = destination_volume.copy()

    for label, cluster in tasks:
        mask = source_volume == label
        volume = split_k_component(mask, h=1, k=cluster, crop=True)

        new_volume[mask] = 0
        max_label = new_volume.max()
        position = volume > 0
        new_volume[position] = volume[position] + max_label

    # labels = numpy.unique(new_volume)
    # labels = labels[labels != 0]
    # lookup_table = numpy.zeros(labels.max() + 1, dtype=numpy.int32)
    # lookup_table[labels] = numpy.arange(1, len(labels) + 1, dtype=numpy.int32)

    # new_volume = lookup_table[new_volume]

    return new_volume

if __name__ == '__main__':
    import os
    import json

    from argparse import ArgumentParser
    from src.config import load_config
    from src.dataset import get_fold
    from src.console import track

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    with open(os.path.join('refine', f'{experiment_name}.json')) as file:
        tasks = json.load(file)

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    config.split_file_path = os.path.join('logs', experiment_name, f'{config.split_filename}.json')

    for fold in range(1, config.num_folds + 1):
        _, valid_dataset_patients = get_fold(config.split_file_path, fold)
        for dataset, patients in valid_dataset_patients.items():
            for patient in track(patients, desc=f'Fold {fold} {dataset}'):
                cc_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, 'cc_1.npy')
                cc_volume = numpy.load(cc_path)
                watershed_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, 'watershed.npy')
                watershed_volume = numpy.load(watershed_path)
                volume = watershed_volume if f'{dataset}/{patient}' not in tasks else refine_component(cc_volume, watershed_volume, tasks[f'{dataset}/{patient}'])

                refine_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, 'refine.npy')
                numpy.save(refine_path, volume)
