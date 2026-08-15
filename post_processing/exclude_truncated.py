import numpy

from constants.label import Label


def exclude_truncated(volume):
    truncated_labels = numpy.union1d(volume[:, :, :3], volume[:, :, -3:])
    truncated_labels = truncated_labels[truncated_labels > 0]

    lookup_table = numpy.arange(volume.max() + 1, dtype=numpy.uint8)
    lookup_table[truncated_labels] = numpy.arange(Label.TRUNCATED, Label.TRUNCATED + len(truncated_labels), dtype=numpy.uint8)

    volume = lookup_table[volume]

    return volume

if __name__ == '__main__':
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

    for dataset, patient in iterate_datasets(experiment_name, datasets):
        volume_path = Path('outputs') / experiment_name / dataset / patient / f'{Mode.REFINE_MERGE}.npy'
        volume = numpy.load(volume_path)

        volume = exclude_truncated(volume)

        output_path = volume_path.parent / f'{Mode.UNTRUNCATED}.npy'
        numpy.save(output_path, volume)
