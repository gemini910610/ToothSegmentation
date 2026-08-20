import numpy
from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


def split_k_component(volume, h, k=None):
    distances = ndimage.distance_transform_edt(volume).astype(numpy.float32)
    distances = ndimage.gaussian_filter(distances, sigma=1)

    peaks = h_maxima(distances, h) & volume

    if k is None:
        markers, _ = ndimage.label(peaks)
    else:
        peak_coordinates = numpy.argwhere(peaks)
        peak_distances = distances[*peak_coordinates.T]

        peak_indices = numpy.argsort(peak_distances)[-k:]
        peak_coordinates = peak_coordinates[peak_indices]

        markers = numpy.zeros_like(volume, dtype=numpy.uint8)
        markers[*peak_coordinates.T] = numpy.arange(1, len(peak_coordinates) + 1, dtype=numpy.uint8)

    volume = watershed(-distances, markers, mask=volume)

    return volume

def split_component(volume):
    new_volume = numpy.zeros_like(volume, dtype=numpy.uint8)

    next_label = 2
    objects = ndimage.find_objects(volume)
    for label, slices in enumerate(objects, 1):
        if slices is None:
            continue

        roi = volume[slices] == label
        components = split_k_component(roi, h=3)

        new_roi = new_volume[slices]
        mask = components > 0
        new_roi[mask] = components[mask] - 1 + next_label
        next_label += components.max()

    return new_volume

if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser
    from pathlib import Path

    from constants.mode import Mode
    from post_processing.utils import iterate_datasets
    from tui.apps import BaseApp

    if len(sys.argv) == 1:
        args = BaseApp().run()
        if args is None:
            sys.exit()

        segmentation_experiment_name = args['experiment_name']
        datasets = args['datasets']
    else:
        parser = ArgumentParser()
        parser.add_argument('segmentation_exp', type=str)
        parser.add_argument('datasets', nargs='+', type=str)
        args = parser.parse_args()

        segmentation_experiment_name = args.segmentation_exp
        datasets = args.datasets

    for dataset, patient in iterate_datasets(segmentation_experiment_name, datasets):
        volume_path = Path('outputs') / segmentation_experiment_name / dataset / patient / f'{Mode.TOOTH_CONNECTED_COMPONENT}.npy'
        volume = numpy.load(volume_path)

        volume = split_component(volume)

        output_path = volume_path.parent / f'{Mode.WATERSHED}.npy'
        numpy.save(output_path, volume)
