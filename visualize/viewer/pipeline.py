import numpy

from constants.label import Label
from post_processing.auto_detect import detect_exclude_labels
from post_processing.exclude_tooth import exclude_tooth
from post_processing.relabel import relabel_volume

IMAGE_FILENAME = 'image.npy'
PREDICT_FILENAME = 'predict.npy'
BONE_FILENAME = 'bone.npy'
TOOTH_FILENAME = 'tooth.npy'


def auto_exclude(watershed_volume, bone_cc_volume):
    bone_mask = bone_cc_volume > 0
    labels = numpy.unique(watershed_volume)
    labels = [int(label) for label in labels if 0 < label < Label.TRUNCATED]

    exclude_labels = detect_exclude_labels(watershed_volume, bone_mask, labels)

    return exclude_tooth(watershed_volume, exclude_labels)

def apply_manual_edit(patient_dir, bone_cc_volume, tooth_volume):
    cleaned_volume = auto_exclude(tooth_volume, bone_cc_volume)

    relabeled_volume = relabel_volume(cleaned_volume, bone_cc_volume)
    numpy.save(patient_dir / TOOTH_FILENAME, relabeled_volume)
