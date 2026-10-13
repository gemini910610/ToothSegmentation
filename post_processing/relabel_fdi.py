from post_processing.tooth_data import is_excluded, set_entry


def relabel_fdi(tooth_data, fdi_mapping):
    positions = {value: label for label, value in tooth_data.items() if not is_excluded(value)}

    for position, fdi in fdi_mapping.items():
        label = positions.get(position)
        if label is not None:
            set_entry(tooth_data, label, 'Missing FDI' if fdi is None else fdi)

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

    args = parse_segmentation_args(required_mode=Mode.RELABELED)
    segmentation_experiment_name = args['segmentation_exp']
    datasets = args['datasets']

    with (Path('edit') / segmentation_experiment_name / 'fdi.json').open() as file:
        fdi_mapping = json.load(file)

    patients = [
        (dataset, patient)
        for dataset, patient in iterate_datasets(segmentation_experiment_name, datasets)
        if f'{dataset}/{patient}' in fdi_mapping
    ]

    for dataset, patient in patients:
        patient_dir = Path('outputs') / segmentation_experiment_name / dataset / patient
        tooth_volume = numpy.load(patient_dir / f'{Mode.TOOTH_FILLED}.npy')

        tooth_data = load_tooth_data(patient_dir)
        relabel_fdi(tooth_data, fdi_mapping[f'{dataset}/{patient}'])
        prune_stale_ids(tooth_data, {int(label) for label in numpy.unique(tooth_volume)})
        save_tooth_data(patient_dir, tooth_data)

        advance_stage(patient_dir, Mode.FDI)
