import numpy


def refine_merge(volume, merge_components):
    lookup_table = numpy.arange(volume.max() + 1, dtype=numpy.uint8)
    for labels in merge_components:
        lookup_table[labels] = labels[0]

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

    with (Path('merge') / f'{experiment_name}.json').open() as file:
        merge_components = json.load(file)

    for dataset, patient in iterate_datasets(experiment_name, datasets):
        volume_path = Path('outputs') / experiment_name / dataset / patient / f'{Mode.REFINE_SPLIT}.npy'
        volume = numpy.load(volume_path)

        data = f'{dataset}/{patient}'
        if data in merge_components:
            volume = refine_merge(volume, merge_components[data])

        output_path = volume_path.parent / f'{Mode.REFINE_MERGE}.npy'
        numpy.save(output_path, volume)
