import cc3d
import numpy


def exclude_truncated(volume):
    truncated_labels = numpy.union1d(volume[:, :, :3], volume[:, :, -3:])
    truncated_labels = [int(label) for label in truncated_labels if label > 0]

    volume = cc3d.dust(volume, 1000, connectivity=6, precomputed_ccl=True)

    remaining_labels = {int(label) for label in numpy.unique(volume)}
    truncated_labels = [label for label in truncated_labels if label in remaining_labels]

    return volume, truncated_labels

if __name__ == '__main__':
    from pathlib import Path

    from constants.mode import Mode
    from post_processing.stage import advance_stage
    from post_processing.tooth_data import (
        load_tooth_data,
        prune_stale_ids,
        save_tooth_data,
    )
    from post_processing.utils import iterate_datasets
    from tui.apps import parse_segmentation_args

    args = parse_segmentation_args(required_mode=Mode.REFINE_MERGE)
    segmentation_experiment_name = args['segmentation_exp']
    datasets = args['datasets']

    for dataset, patient in iterate_datasets(segmentation_experiment_name, datasets):
        patient_dir = Path('outputs') / segmentation_experiment_name / dataset / patient
        volume_path = patient_dir / f'{Mode.REFINE_MERGE}.npy'
        volume = numpy.load(volume_path)

        volume, truncated_labels = exclude_truncated(volume)

        output_path = patient_dir / f'{Mode.UNTRUNCATED}.npy'
        numpy.save(output_path, volume)

        tooth_data = load_tooth_data(patient_dir)
        tooth_data = {label: value for label, value in tooth_data.items() if value != 'Truncated'}
        tooth_data.update({label: 'Truncated' for label in truncated_labels})
        prune_stale_ids(tooth_data, {int(label) for label in numpy.unique(volume)})
        save_tooth_data(patient_dir, tooth_data)

        advance_stage(patient_dir, Mode.UNTRUNCATED)
