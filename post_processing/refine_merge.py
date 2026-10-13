import numpy


def refine_merge(volume, merge_components):
    lookup_table = numpy.arange(volume.max() + 1, dtype=numpy.uint8)
    for labels in merge_components:
        lookup_table[labels] = labels[0]

    volume = lookup_table[volume]

    return volume

if __name__ == '__main__':
    import json
    from pathlib import Path

    from constants.mode import Mode
    from post_processing.stage import advance_stage
    from post_processing.utils import iterate_datasets
    from tui.apps import parse_segmentation_args

    args = parse_segmentation_args(required_mode=Mode.REFINE_SPLIT)
    segmentation_experiment_name = args['segmentation_exp']
    datasets = args['datasets']

    with (Path('edit') / segmentation_experiment_name / 'merge.json').open() as file:
        merge_components = json.load(file)

    for dataset, patient in iterate_datasets(segmentation_experiment_name, datasets):
        patient_dir = Path('outputs') / segmentation_experiment_name / dataset / patient
        volume = numpy.load(patient_dir / f'{Mode.REFINE_SPLIT}.npy')

        data = f'{dataset}/{patient}'
        if data in merge_components:
            volume = refine_merge(volume, merge_components[data])

        numpy.save(patient_dir / f'{Mode.REFINE_MERGE}.npy', volume)

        advance_stage(patient_dir, Mode.REFINE_MERGE)
