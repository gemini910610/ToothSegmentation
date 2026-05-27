import cc3d
import numpy

from scripts.tools.widgets import Label

def remove_cropped(volume, voxel_threshold=0):
    volume = volume.copy()
    component_count = volume.max()

    remove_labels = numpy.union1d(volume[:, :, :3], volume[:, :, -3:])
    remove_labels = remove_labels[remove_labels != 0]

    lookup_table = numpy.arange(component_count + 1, dtype=numpy.uint8)
    lookup_table[remove_labels] = numpy.arange(Label.CROPPED, Label.CROPPED + len(remove_labels), dtype=numpy.uint8)

    volume = lookup_table[volume]

    volume = cc3d.dust(volume, voxel_threshold, connectivity=6)

    return volume

if __name__ == '__main__':
    import os

    from argparse import ArgumentParser

    from scripts.post_processing.iterators import iterate_datasets
    from scripts.tools.widgets import Mode

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('--threshold', type=int, default=7500)
    args = parser.parse_args()

    experiment_name = args.exp
    voxel_threshold = args.threshold

    datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
    for dataset, patient in iterate_datasets(experiment_name, datasets):
        volume_path = os.path.join('outputs', experiment_name, dataset, patient, f'{Mode.REFINE_MERGE}.npy')
        volume = numpy.load(volume_path)
        volume = remove_cropped(volume, voxel_threshold)

        cleaned_path = os.path.join('outputs', experiment_name, dataset, patient, f'{Mode.CLEANED}.npy')
        numpy.save(cleaned_path, volume)
