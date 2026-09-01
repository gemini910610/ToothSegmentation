from post_processing.utils import ROIVolume
from post_processing.watershed import split_k_component


def refine_split(source_volume, destination_volume, split_components):
    new_volume = destination_volume.copy()

    for label, count in split_components.items():
        label = int(label)
        mask = source_volume == label

        roi = ROIVolume(mask)
        roi.volume = split_k_component(roi.volume, h=1, k=count)
        volume = roi.restore()

        max_label = new_volume.max()
        new_volume[mask] = volume[mask] + max_label

    return new_volume

if __name__ == '__main__':
    import json
    from pathlib import Path

    import numpy

    from constants.mode import Mode
    from post_processing.utils import iterate_datasets
    from tui.apps import parse_segmentation_args

    args = parse_segmentation_args()
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
        volume = watershed_volume if data not in split_components else refine_split(cc_volume, watershed_volume, split_components[data])

        output_path = cc_path.parent / f'{Mode.REFINE_SPLIT}.npy'
        numpy.save(output_path, volume)
