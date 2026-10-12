import numpy

from post_processing.auto_detect import detect_exclude_labels
from post_processing.relabel import relabel_volume
from visualize.viewer.tooth_data import (
    is_excluded,
    load_tooth_data,
    prune_stale_ids,
    save_tooth_data,
)

IMAGE_FILENAME = 'image.npy'
PREDICT_FILENAME = 'predict.npy'
BONE_FILENAME = 'bone.npy'
TOOTH_FILENAME = 'tooth.npy'


def auto_exclude(tooth_volume, bone_cc_volume, excluded_ids):
    bone_mask = bone_cc_volume > 0
    labels = numpy.unique(tooth_volume)
    labels = [int(label) for label in labels if label > 0 and label not in excluded_ids]

    exclude_labels = detect_exclude_labels(tooth_volume, bone_mask, labels)

    return {
        label: exclude_type
        for exclude_type, labels in exclude_labels.items()
        for label in labels
    }

def apply_manual_edit(patient_dir, bone_cc_volume, tooth_volume, recompute=True):
    tooth_data = load_tooth_data(patient_dir)

    if recompute:
        excluded_ids = {label for label, value in tooth_data.items() if is_excluded(value)}
        tooth_data.update(auto_exclude(tooth_volume, bone_cc_volume, excluded_ids))

        excluded_ids = {label for label, value in tooth_data.items() if is_excluded(value)}
        _, positions = relabel_volume(tooth_volume, bone_cc_volume, excluded_ids=excluded_ids)
        tooth_data.update(positions)

    prune_stale_ids(tooth_data, {int(label) for label in numpy.unique(tooth_volume)})

    save_tooth_data(patient_dir, tooth_data)
    numpy.save(patient_dir / TOOTH_FILENAME, tooth_volume)
