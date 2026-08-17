import numpy

from constants.label import Label


def exclude_tooth(volume, exclude_labels):
    exclude_types = {
        'Unerupted Tooth': Label.UNERUPTED,
        'Residual Root': Label.RESIDUAL_ROOT,
        'Fake Tooth': Label.FAKE
    }

    lookup_table = numpy.arange(volume.max() + 1, dtype=numpy.uint8)
    for exclude_type, labels in exclude_labels.items():
        start_label = exclude_types[exclude_type]
        lookup_table[labels] = numpy.arange(start_label, start_label + len(labels), dtype=numpy.uint8)

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

    with (Path('exclude') / f'{experiment_name}.json').open() as file:
        exclude_labels = json.load(file)

    for dataset, patient in iterate_datasets(experiment_name, datasets):
        volume_path = Path('outputs') / experiment_name / dataset / patient / f'{Mode.UNTRUNCATED}.npy'
        volume = numpy.load(volume_path)

        data = f'{dataset}/{patient}'
        if data in exclude_labels:
            volume = exclude_tooth(volume, exclude_labels[data])

        output_path = volume_path.parent / f'{Mode.CLEANED}.npy'
        numpy.save(output_path, volume)
