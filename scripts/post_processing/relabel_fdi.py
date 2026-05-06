import numpy

def relabel_fdi(volume, fdi_mapping):
    tooth_count = volume.max()
    lookup_table = numpy.arange(tooth_count + 1, dtype=numpy.uint8)
    for label, fdi in fdi_mapping.items():
        label = int(label)
        fdi = int(fdi)
        lookup_table[label] = fdi

    volume = lookup_table[volume]
    return volume

if __name__ == '__main__':
    import json
    import os

    from argparse import ArgumentParser

    from scripts.post_processing.iterators import iterate_fold_patients
    from scripts.tools.widgets import Mode
    from src.config import load_config

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    with open(os.path.join('fdi', f'{experiment_name}.json')) as file:
        fdi_mapping = json.load(file)

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    config.split_file_path = os.path.join('logs', experiment_name, f'{config.split_filename}.json')

    for fold, dataset, patient in iterate_fold_patients(config):
        if f'{dataset}/{patient}' not in fdi_mapping:
            continue

        output_dir = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient)
        volume_path = os.path.join(output_dir, f'{Mode.POST_PROCESSING}.npy')
        volume = numpy.load(volume_path)

        volume = relabel_fdi(volume, fdi_mapping[f'{dataset}/{patient}'])

        fdi_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.FDI}.npy')
        numpy.save(fdi_path, volume)
