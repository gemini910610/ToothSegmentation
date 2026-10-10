import numpy

from post_processing.utils import ROIVolume, next_available_label
from post_processing.watershed import split_k_component


def refine_split(source_volume, destination_volume, split_components):
    new_volume = destination_volume.copy()
    used = {int(value) for value in numpy.unique(new_volume)}

    new_labels = {}
    for label, count in split_components.items():
        label = int(label)
        mask = source_volume == label

        roi = ROIVolume(mask)
        roi.volume = split_k_component(roi.volume, h=1, k=count)
        volume = roi.restore()

        labels = []
        for _ in range(count):
            new_label = next_available_label(used, 11)
            used.add(new_label)
            labels.append(new_label)

        lookup_table = numpy.zeros(count + 1, dtype=numpy.uint8)
        lookup_table[1:] = labels
        new_volume[mask] = lookup_table[volume[mask]]
        new_labels[label] = labels

    return new_volume, new_labels

if __name__ == '__main__':
    import json
    from pathlib import Path

    from constants.mode import Mode
    from post_processing.utils import iterate_datasets
    from tui.apps import parse_segmentation_args

    args = parse_segmentation_args(required_mode=Mode.WATERSHED)
    segmentation_experiment_name = args['segmentation_exp']
    datasets = args['datasets']

    with (Path('edit') / segmentation_experiment_name / 'split.json').open() as file:
        split_components = json.load(file)

    for dataset, patient in iterate_datasets(segmentation_experiment_name, datasets):
        cc_path = Path('outputs') / segmentation_experiment_name / dataset / patient / f'{Mode.TOOTH_CONNECTED_COMPONENT}.npy'
        cc_volume = numpy.load(cc_path)
        watershed_path = cc_path.parent / f'{Mode.WATERSHED}.npy'
        watershed_volume = numpy.load(watershed_path)

        data = f'{dataset}/{patient}'
        volume = watershed_volume if data not in split_components else refine_split(cc_volume, watershed_volume, split_components[data])[0]

        output_path = cc_path.parent / f'{Mode.REFINE_SPLIT}.npy'
        numpy.save(output_path, volume)
