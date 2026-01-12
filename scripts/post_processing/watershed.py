import numpy

from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

def split_component(volume):
    origin_shape = volume.shape
    coordinates = numpy.argwhere(volume)
    x0, y0, z0 = coordinates.min(axis=0)
    x1, y1, z1 = coordinates.max(axis=0)

    x0 = max(x0 - 10, 0)
    y0 = max(y0 - 10, 0)
    z0 = max(z0 - 10, 0)

    x1 = min(x1 + 10, volume.shape[0])
    y1 = min(y1 + 10, volume.shape[1])
    z1 = min(z1 + 10, volume.shape[2])

    volume = volume[x0:x1, y0:y1, z0:z1]

    distance = ndimage.distance_transform_edt(volume)
    distance = ndimage.gaussian_filter(distance, sigma=1)

    peaks = h_maxima(distance, h=3) & volume
    markers, _ = ndimage.label(peaks)

    volume = watershed(-distance, markers, mask=volume)

    origin_volume = numpy.zeros(origin_shape, dtype=numpy.int32)
    origin_volume[x0:x1, y0:y1, z0:z1] = volume

    return origin_volume

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
                predict_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, 'volume.npy')
                volume = numpy.load(predict_path)

                volume = volume == 1
                volume = filter_connected_component(volume, voxel_threshold, binary=True)
                volume = split_component(volume)

                pp_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, 'watershed_volume.npy')
                numpy.save(pp_path, volume)
