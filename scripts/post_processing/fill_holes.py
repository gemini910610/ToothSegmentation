import numpy

from scipy import ndimage

from scripts.post_processing.connected_component import get_bounding_box

def fill_holes(tooth_volume, bone_volume):
    tooth_volume = tooth_volume.copy()
    bone_volume = bone_volume.copy()
    bone_volume = (bone_volume > 0).astype(numpy.uint8)

    objects = ndimage.find_objects(tooth_volume)

    for label, slices in enumerate(objects, 1):
        if slices is None:
            continue

        roi = tooth_volume[slices]
        mask = roi == label
        mask = ndimage.binary_fill_holes(mask)
        roi[mask] = label

    volume = (bone_volume > 0) | (tooth_volume > 0)
    x0, x1, y0, y1, z0, z1 = get_bounding_box(volume)
    mask = volume[x0:x1, y0:y1, z0:z1]
    filled = ndimage.binary_fill_holes(mask)

    tooth_mask = tooth_volume[x0:x1, y0:y1, z0:z1]
    tooth_filled = ndimage.binary_fill_holes(tooth_mask)

    holes = filled & ~mask & ~tooth_filled
    roi = bone_volume[x0:x1, y0:y1, z0:z1]
    roi[holes] = 1

    return tooth_volume, bone_volume

if __name__ == '__main__':
    import os

    from argparse import ArgumentParser

    from scripts.post_processing.iterators import iterate_datasets
    from scripts.tools.widgets import Mode

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
    for dataset, patient in iterate_datasets(experiment_name, datasets):
        output_dir = os.path.join('outputs', experiment_name, dataset, patient)
        tooth_volume_path = os.path.join(output_dir, f'{Mode.REMOVED}.npy')
        bone_volume_path = os.path.join(output_dir, f'{Mode.BONE_CONNECTED_COMPONENT}.npy')
        tooth_volume = numpy.load(tooth_volume_path)
        bone_volume = numpy.load(bone_volume_path)

        tooth_volume, bone_volume = fill_holes(tooth_volume, bone_volume)

        tooth_filled_path = os.path.join(output_dir, f'{Mode.TOOTH_FILLED}.npy')
        bone_filled_path = os.path.join(output_dir, f'{Mode.BONE_FILLED}.npy')
        numpy.save(tooth_filled_path, tooth_volume)
        numpy.save(bone_filled_path, bone_volume)
