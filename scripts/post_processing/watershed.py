import numpy

from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

def split_k_component(volume, h, k=None):
    distance = ndimage.distance_transform_edt(volume).astype(numpy.float32, copy=False)
    distance = ndimage.gaussian_filter(distance, sigma=1)

    peaks = h_maxima(distance, h) & volume

    if k is None:
        markers, _ = ndimage.label(peaks)
    else:
        coordinates = numpy.argwhere(peaks)
        peak_distance = distance[*coordinates.T]
        top_k = numpy.argpartition(peak_distance, -k)[-k:]
        coordinates = coordinates[top_k]
        markers = numpy.zeros_like(volume, dtype=numpy.int32)
        for i, coordinate in enumerate(coordinates, 1):
            markers[*coordinate] = i

    volume = watershed(-distance, markers, mask=volume)

    return volume

def split_component(volume):
    component_count = volume.max()
    new_volume = numpy.zeros_like(volume, dtype=numpy.int32)

    index = 1
    components = ndimage.find_objects(volume, max_label=component_count)
    for label, slices in enumerate(components, 1):
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
    from .connected_component import filter_connected_component

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
                cc_volume_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, 'cc_volume_1.npy')
                if os.path.exists(cc_volume_path):
                    volume = numpy.load(cc_volume_path)
                else:
                    predict_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, 'volume.npy')
                    volume = numpy.load(predict_path)

                    volume = volume == 1
                    volume = filter_connected_component(volume, voxel_threshold)

                volume = split_component(volume)

                watershed_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, 'watershed_volume.npy')
                numpy.save(watershed_path, volume)
