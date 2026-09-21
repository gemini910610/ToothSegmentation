from post_processing.utils import get_bone_surface, get_tooth_surface

UNERUPTED_BONE_RATIO_THRESHOLD = 0.7


def detect_unerupted_labels(tooth_volume, bone_mask, labels):
    detected = []
    for label in labels:
        tooth_mask = tooth_volume == label
        tooth_surface = get_tooth_surface(tooth_mask)
        tooth_surface_count = tooth_surface.sum()
        if tooth_surface_count == 0:
            continue

        bone_surface = get_bone_surface(tooth_mask, bone_mask)
        ratio = bone_surface.sum() / tooth_surface_count
        if ratio > UNERUPTED_BONE_RATIO_THRESHOLD:
            detected.append(int(label))

    return detected

def detect_fake_labels(tooth_volume, bone_mask, labels):
    detected = []
    for label in labels:
        tooth_mask = tooth_volume == label
        bone_surface = get_bone_surface(tooth_mask, bone_mask)
        if not bone_surface.any():
            detected.append(int(label))

    return detected
