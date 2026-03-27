import numpy

from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.segmentation import watershed
from scripts.post_processing.connected_component import get_bounding_box

def split_k_component(volume, h, k=None, crop=False):
    if crop:
        origin_shape = volume.shape
        x0, x1, y0, y1, z0, z1 = get_bounding_box(volume)
        volume = volume[x0:x1, y0:y1, z0:z1]

    distance = ndimage.distance_transform_edt(volume).astype(numpy.float32, copy=False)
    distance = ndimage.gaussian_filter(distance, sigma=1)

    peaks = h_maxima(distance, h) & volume

    if k is None:
        markers, _ = ndimage.label(peaks)
    else:
        peak_indices = numpy.flatnonzero(peaks)
        peak_values = distance.ravel()[peak_indices]

        indices = numpy.argpartition(peak_values, -k)[-k:]
        peak_indices = peak_indices[indices]

        coordinates = numpy.column_stack(numpy.unravel_index(peak_indices, peaks.shape))
        markers = numpy.zeros_like(volume, dtype=numpy.uint8)
        markers[*coordinates.T] = numpy.arange(1, len(coordinates) + 1, dtype=numpy.uint8)

    volume = watershed(-distance, markers, mask=volume)
    if not crop:
        return volume

    new_volume = numpy.zeros(origin_shape, dtype=numpy.uint8)
    new_volume[x0:x1, y0:y1, z0:z1] = volume
    return new_volume

def split_component(volume):
    component_count = volume.max()
    new_volume = numpy.zeros_like(volume, dtype=numpy.uint8)

    index = 1
    components = ndimage.find_objects(volume, max_label=component_count)
    for label, slices in enumerate(components, 1):
        if slices is None:
            continue
        roi = volume[slices] == label
        split_components = split_k_component(roi, h=3)
        new_roi = new_volume[slices]
        mask = split_components > 0
        new_roi[mask] = split_components[mask] + (index - 1)
        index += split_components.max()

    return new_volume

if __name__ == '__main__':
    import os

    from argparse import ArgumentParser
    from src.config import load_config
    from src.console import track
    from src.dataset import get_fold
    from scripts.tools.widgets import Mode

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('--threshold', type=int, default=7500)
    args = parser.parse_args()

    experiment_name = args.exp
    voxel_threshold = args.threshold

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    config.split_file_path = os.path.join('logs', experiment_name, f'{config.split_filename}.json')

    for fold in range(1, config.num_folds + 1):
        _, valid_dataset_patients = get_fold(config.split_file_path, fold)
        for dataset, patients in valid_dataset_patients.items():
            for patient in track(patients, desc=f'Fold {fold} {dataset}'):
                cc_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.TOOTH_CONNECTED_COMPONENT}.npy')
                volume = numpy.load(cc_path)
                volume = split_component(volume)

                watershed_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.WATERSHED}.npy')
                numpy.save(watershed_path, volume)
