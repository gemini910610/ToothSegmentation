from pathlib import Path

import numpy

from utils.console import track


class ROIVolume:
    def __init__(self, volume):
        self.origin_shape = volume.shape
        self.slices = self._get_bounding_box(volume)
        self.volume = volume[self.slices]

    def _get_bounding_box(self, volume):
        xs = numpy.where(volume.any(axis=(1, 2)))[0]
        ys = numpy.where(volume.any(axis=(0, 2)))[0]
        zs = numpy.where(volume.any(axis=(0, 1)))[0]

        return (slice(xs[0], xs[-1] + 1), slice(ys[0], ys[-1] + 1), slice(zs[0], zs[-1] + 1))

    def restore(self):
        origin_volume = numpy.zeros(self.origin_shape, dtype=numpy.uint8)
        origin_volume[self.slices] = self.volume

        return origin_volume

def iterate_datasets(experiment_name, datasets):
    for dataset in datasets:
        dataset_dir = Path('outputs') / experiment_name / dataset
        for patient in track(list(dataset_dir.iterdir()), desc=dataset):
            yield dataset, patient.name
