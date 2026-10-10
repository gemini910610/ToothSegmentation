from post_processing.utils import get_bone_surface, get_tooth_surface

UNERUPTED_BONE_RATIO_THRESHOLD = 0.7


def detect_exclude_labels(tooth_volume, bone_mask, labels):
    exclude_labels = {'Unerupted Tooth': [], 'Fake Tooth': []}
    for label in labels:
        tooth_mask = tooth_volume == label
        bone_surface = get_bone_surface(tooth_mask, bone_mask)
        if not bone_surface.any():
            exclude_labels['Fake Tooth'].append(int(label))
            continue

        tooth_surface_count = get_tooth_surface(tooth_mask).sum()
        ratio = bone_surface.sum() / tooth_surface_count
        if ratio > UNERUPTED_BONE_RATIO_THRESHOLD:
            exclude_labels['Unerupted Tooth'].append(int(label))

    return exclude_labels
