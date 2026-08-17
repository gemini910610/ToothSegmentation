import numpy

from constants.label import Label


def relabel_fdi(volume, fdi_mapping):
    lookup_table = numpy.arange(volume.max() + 1, dtype=numpy.uint8)

    missing_label = Label.MISSING_FDI
    labels = numpy.unique(volume)
    labels = labels[(labels > 1) & (labels < Label.TRUNCATED)]
    for label in labels:
        fdi = fdi_mapping[str(label)]
        if fdi is None:
            lookup_table[label] = missing_label
            missing_label += 1
        else:
            lookup_table[label] = int(fdi)

    volume = lookup_table[volume]

    return volume

if __name__ == '__main__':
    import json
    import sys
    from argparse import ArgumentParser
    from pathlib import Path

    from constants.mode import Mode
    from post_processing.utils import iterate_datasets
    from tui.apps import BaseApp

    if len(sys.argv) == 1:
        args = BaseApp().run()
        if args is None:
            sys.exit()

        experiment_name = args['experiment_name']
        datasets = args['datasets']
    else:
        parser = ArgumentParser()
        parser.add_argument('exp', type=str)
        parser.add_argument('datasets', nargs='+', type=str)
        args = parser.parse_args()

        experiment_name = args.exp
        datasets = args.datasets

    with (Path('fdi') / f'{experiment_name}.json').open() as file:
        fdi_mapping = json.load(file)

    patients = [
        (dataset, patient)
        for dataset, patient in iterate_datasets(experiment_name, datasets)
        if f'{dataset}/{patient}' in fdi_mapping
    ]

    for dataset, patient in patients:
        volume_path = Path('outputs') / experiment_name / dataset / patient / f'{Mode.RELABELED}.npy'
        volume = numpy.load(volume_path)

        volume = relabel_fdi(volume, fdi_mapping[f'{dataset}/{patient}'])

        output_path = volume_path.parent / f'{Mode.FDI}.npy'
        numpy.save(output_path, volume)
