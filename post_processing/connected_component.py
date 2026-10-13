import cc3d
import numpy
from scipy import ndimage

from post_processing.utils import ROIVolume


def relabel(volume):
    labels = numpy.unique(volume)
    labels = labels[labels > 0]

    if labels.size == 0:
        return numpy.zeros_like(volume, dtype=numpy.uint8)

    lookup_table = numpy.zeros(labels.max() + 1, dtype=numpy.uint8)
    lookup_table[labels] = numpy.arange(2, len(labels) + 2, dtype=numpy.uint8)
    volume = lookup_table[volume]

    return volume

def _remove_outlier(volume):
    objects = ndimage.find_objects(volume)

    valid_labels = []
    centroids = []
    heights = []

    for label, slices in enumerate(objects, 1):
        if slices is None:
            continue

        roi = volume[slices] == label

        z_min = slices[2].start
        z_max = slices[2].stop
        centroid = numpy.argwhere(roi)[:, 2].mean() + z_min
        height = z_max - z_min

        valid_labels.append(label)
        centroids.append(centroid)
        heights.append(height)

    valid_labels = numpy.array(valid_labels, dtype=numpy.uint8)
    centroids = numpy.array(centroids, dtype=numpy.float32)

    centroid_median = numpy.median(centroids)
    distances = numpy.abs(centroids - centroid_median)
    height_median = numpy.median(heights)
    outlier_labels = valid_labels[distances > 1.5 * height_median]

    lookup_table = numpy.arange(volume.max() + 1, dtype=numpy.uint8)
    lookup_table[outlier_labels] = 0
    volume = lookup_table[volume]

    return volume

def filter_connected_component(volume, is_bone=False, voxel_threshold=0):
    roi = ROIVolume(volume)

    components = cc3d.connected_components(roi.volume, connectivity=6, binary_image=True)
    components = cc3d.dust(components, voxel_threshold, connectivity=6, precomputed_ccl=True)

    if is_bone:
        volume = cc3d.largest_k(components, k=3, precomputed_ccl=True)
        roi.volume = relabel(volume)
    else:
        volume = relabel(components)
        roi.volume = _remove_outlier(volume)

    volume = roi.restore()

    return volume

if __name__ == '__main__':
    from pathlib import Path

    from constants.label import Label
    from constants.mode import Mode
    from post_processing.stage import advance_stage
    from post_processing.utils import iterate_datasets
    from tui.apps import SwitchThresholdApp, parse_args

    def configure_parser(parser):
        parser.add_argument('segmentation_exp', type=str)
        parser.add_argument('datasets', nargs='+', type=str)
        parser.add_argument('target', choices=['tooth', 'bone'])
        parser.add_argument('threshold', type=int)

    args = parse_args(SwitchThresholdApp, configure_parser, rename={'exp': 'segmentation_exp'}, app_kwargs={'required_mode': Mode.PREDICT})

    segmentation_experiment_name = args['segmentation_exp']
    datasets = args['datasets']
    target_name = args['target']
    voxel_threshold = args['threshold']

    target = Label.TOOTH if target_name == 'tooth' else Label.BONE

    mode = Mode.TOOTH_CONNECTED_COMPONENT if target == Label.TOOTH else Mode.BONE_CONNECTED_COMPONENT

    for dataset, patient in iterate_datasets(segmentation_experiment_name, datasets):
        patient_dir = Path('outputs') / segmentation_experiment_name / dataset / patient
        volume = numpy.load(patient_dir / f'{Mode.PREDICT}.npy')

        volume = volume == target
        volume = filter_connected_component(volume, target == Label.BONE, voxel_threshold)

        numpy.save(patient_dir / f'{mode}.npy', volume)

        advance_stage(patient_dir, mode)
