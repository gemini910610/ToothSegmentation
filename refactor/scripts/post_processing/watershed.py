import numpy

from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from refactor.scripts.post_processing.connected_component import get_bounding_box

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

    index = 2
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

    from refactor.scripts.post_processing.iterators import iterate_datasets
    from refactor.scripts.tools.widgets import Mode

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
    for dataset, patient in iterate_datasets(experiment_name, datasets):
        cc_path = os.path.join('outputs', experiment_name, dataset, patient, f'{Mode.TOOTH_CONNECTED_COMPONENT}.npy')
        volume = numpy.load(cc_path)
        volume = split_component(volume)

        watershed_path = os.path.join('outputs', experiment_name, dataset, patient, f'{Mode.WATERSHED}.npy')
        numpy.save(watershed_path, volume)
