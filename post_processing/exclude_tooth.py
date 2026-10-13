def exclude_tooth(exclude_labels):
    return {
        label: exclude_type
        for exclude_type, labels in exclude_labels.items()
        for label in labels
    }

if __name__ == '__main__':
    import json
    from pathlib import Path

    import numpy

    from constants.mode import Mode
    from post_processing.stage import advance_stage
    from post_processing.tooth_data import (
        load_tooth_data,
        prune_stale_ids,
        save_tooth_data,
    )
    from post_processing.utils import iterate_datasets
    from tui.apps import parse_segmentation_args

    args = parse_segmentation_args(required_mode=Mode.UNTRUNCATED)
    segmentation_experiment_name = args['segmentation_exp']
    datasets = args['datasets']

    with (Path('edit') / segmentation_experiment_name / 'exclude.json').open() as file:
        exclude_labels = json.load(file)

    managed_types = {'Unerupted Tooth', 'Residual Root', 'Fake Tooth', 'Noise'}

    for dataset, patient in iterate_datasets(segmentation_experiment_name, datasets):
        patient_dir = Path('outputs') / segmentation_experiment_name / dataset / patient
        volume_path = patient_dir / f'{Mode.UNTRUNCATED}.npy'

        tooth_data = load_tooth_data(patient_dir)
        tooth_data = {label: value for label, value in tooth_data.items() if value not in managed_types}

        data = f'{dataset}/{patient}'
        if data in exclude_labels:
            tooth_data.update(exclude_tooth(exclude_labels[data]))

        prune_stale_ids(tooth_data, {int(label) for label in numpy.unique(numpy.load(volume_path))})
        save_tooth_data(patient_dir, tooth_data)

        advance_stage(patient_dir, Mode.CLEANED)
