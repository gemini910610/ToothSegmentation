import numpy

from scripts.tools.widgets import Label

def remove_tooth(volume, data):
    remove_types = {
        'Unerupted Tooth': Label.UNERUPTED,
        'Residual Root': Label.RESIDUAL,
        'Fake Tooth': Label.FAKE
    }

    lookup_table = numpy.arange(volume.max() + 1, dtype=numpy.uint8)
    for remove_type, labels in data.items():
        start_label = remove_types[remove_type]
        lookup_table[labels] = numpy.arange(start_label, start_label + len(labels), dtype=numpy.uint8)
    volume = lookup_table[volume]
    return volume

if __name__ == '__main__':
    import os
    import json

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

    with open(os.path.join('remove', f'{experiment_name}.json')) as file:
        labels = json.load(file)

    for fold in range(1, config.num_folds + 1):
        _, valid_dataset_patients = get_fold(config.split_file_path, fold)
        for dataset, patients in valid_dataset_patients.items():
            for patient in track(patients, desc=f'Fold {fold} {dataset}'):
                volume_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.CLEANED}.npy')
                volume = numpy.load(volume_path)
                data = f'{dataset}/{patient}'
                if data in labels:
                    volume = remove_tooth(volume, labels[data])

                removed_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.REMOVED}.npy')
                numpy.save(removed_path, volume)
