import numpy

from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

def split_component(volume):
    component_count = volume.max()
    new_volume = numpy.zeros_like(volume)

    index = 1
    components = ndimage.find_objects(volume, max_label=component_count)
    for label, slices in enumerate(components, 1):
        roi = volume[slices] == label

        distance = ndimage.distance_transform_edt(roi).astype(numpy.float32, copy=False)
        distance = ndimage.gaussian_filter(distance, sigma=1)

        peaks = h_maxima(distance, h=3) & roi
        markers, _ = ndimage.label(peaks)

        split_components = watershed(-distance, markers, mask=roi)
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
    parser.add_argument('--threshold', type=int, default=3500)
    args = parser.parse_args()

    experiment_name = args.exp
    voxel_threshold = args.threshold

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))

    for fold in range(1, config.num_folds + 1):
        _, valid_dataset_patients = get_fold(config.split_filename, fold)
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
