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
    from pathlib import Path

    from constants.mode import Mode
    from post_processing.utils import iterate_datasets
    from tui.apps import parse_segmentation_args

    args = parse_segmentation_args()
    segmentation_experiment_name = args['segmentation_exp']
    datasets = args['datasets']

    with (Path('exclude') / f'{segmentation_experiment_name}.json').open() as file:
        exclude_labels = json.load(file)

    for dataset, patient in iterate_datasets(segmentation_experiment_name, datasets):
        volume_path = Path('outputs') / segmentation_experiment_name / dataset / patient / f'{Mode.UNTRUNCATED}.npy'
        volume = numpy.load(volume_path)

        data = f'{dataset}/{patient}'
        if data in exclude_labels:
            volume = exclude_tooth(volume, exclude_labels[data])

        output_path = volume_path.parent / f'{Mode.CLEANED}.npy'
        numpy.save(output_path, volume)
