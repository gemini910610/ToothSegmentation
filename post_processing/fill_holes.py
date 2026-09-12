import numpy
from scipy import ndimage

from post_processing.utils import ROIVolume


def _fill_background_holes(mask):
    background = ~mask
    labels, _ = ndimage.label(background)

    border_labels = numpy.concatenate([labels[0].ravel(), labels[-1].ravel(), labels[:, 0].ravel(), labels[:, -1].ravel(), labels[:, :, 0].ravel(), labels[:, :, -1].ravel()])
    border_labels = numpy.unique(border_labels)
    border_labels = border_labels[border_labels > 0]

    holes = background & ~numpy.isin(labels, border_labels)

    return mask | holes

def fill_holes(tooth_volume, bone_volume):
    tooth_volume = tooth_volume.copy()
    bone_volume = (bone_volume > 0).astype(numpy.uint8)

    objects = ndimage.find_objects(tooth_volume)
    for label, slices in enumerate(objects, 1):
        if slices is None:
            continue

        roi = tooth_volume[slices]
        mask = ndimage.binary_fill_holes(roi == label)
        roi[mask] = label

    roi = ROIVolume(bone_volume.astype(bool) | (tooth_volume > 0))
    filled = _fill_background_holes(roi.volume)
    tooth_filled = _fill_background_holes(tooth_volume[roi.slices] > 0)
    holes = filled & ~roi.volume & ~tooth_filled

    bone_volume[roi.slices][holes] = 1

    return tooth_volume, bone_volume

if __name__ == '__main__':
    from pathlib import Path

    from constants.mode import Mode
    from post_processing.utils import iterate_datasets
    from tui.apps import parse_segmentation_args

    args = parse_segmentation_args(required_mode=Mode.CLEANED)
    segmentation_experiment_name = args['segmentation_exp']
    datasets = args['datasets']

    for dataset, patient in iterate_datasets(segmentation_experiment_name, datasets):
        tooth_path = Path('outputs') / segmentation_experiment_name / dataset / patient / f'{Mode.CLEANED}.npy'
        tooth_volume = numpy.load(tooth_path)
        bone_path = tooth_path.parent / f'{Mode.BONE_CONNECTED_COMPONENT}.npy'
        bone_volume = numpy.load(bone_path)

        tooth_volume, bone_volume = fill_holes(tooth_volume, bone_volume)

        output_path = tooth_path.parent / f'{Mode.TOOTH_FILLED}.npy'
        numpy.save(output_path, tooth_volume)
        output_path = bone_path.parent / f'{Mode.BONE_FILLED}.npy'
        numpy.save(output_path, bone_volume)
