import numpy

from scripts.tools.widgets import Label

def remove_cropped(volume):
    volume = volume.copy()
    component_count = volume.max()

    remove_labels = numpy.union1d(volume[:, :, :3], volume[:, :, -3:])
    remove_labels = remove_labels[remove_labels != 0]

    lookup_table = numpy.arange(component_count + 1, dtype=numpy.uint8)
    lookup_table[remove_labels] = numpy.arange(Label.CROPPED, Label.CROPPED + len(remove_labels), dtype=numpy.uint8)

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
    args = parser.parse_args()

    experiment_name = args.exp

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    config.split_file_path = os.path.join('logs', experiment_name, f'{config.split_filename}.json')

    for fold in range(1, config.num_folds + 1):
        _, valid_dataset_patients = get_fold(config.split_file_path, fold)
        for dataset, patients in valid_dataset_patients.items():
            for patient in track(patients, desc=f'Fold {fold} {dataset}'):
                volume_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.REFINE}.npy')
                volume = numpy.load(volume_path)
                volume = remove_cropped(volume)

                cleaned_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.CLEANED}.npy')
                numpy.save(cleaned_path, volume)
