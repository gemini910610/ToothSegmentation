import numpy

from scipy import ndimage
from src.console import track
from scripts.tools.widgets import Label
from scripts.post_processing.tooth_slice import find_normal_vectors, extract_single_tooth, align_crop_tooth, get_slices, normalize_slice, restore_coordinates, split_surface

def compute_bone_loss(segmentation_volume, image_volume, cej_finder):
    normal_vectors = find_normal_vectors(segmentation_volume)

    labels = numpy.unique(segmentation_volume)
    labels = labels[(labels > 1) & (labels < Label.CROPPED)] # exclude background, bone and removed teeth

    bone_losses = {}
    for label in track(labels):
        normal_vector = normal_vectors[label]
        filtered_volume, transform_meta = extract_single_tooth(segmentation_volume, normal_vector, tooth_label=label, bone_label=1)
        aligned_volume, aligned_image = align_crop_tooth(filtered_volume, image_volume, transform_meta)

        slices = get_slices(aligned_volume, aligned_image, count=64, reverse=label // 10 in {2, 3})

        origin_segmentation_slices = []
        segmentation_slices = []
        image_slices = []
        tooth_slices = []
        for origin_segmentation_slice, image_slice, tooth_slice in slices:
            segmentation_slice, image_slice, tooth_slice = normalize_slice(origin_segmentation_slice, image_slice, tooth_slice)

            origin_segmentation_slices.append(origin_segmentation_slice)
            segmentation_slices.append(segmentation_slice)
            image_slices.append(image_slice)
            tooth_slices.append(tooth_slice)

        points = cej_finder.find(origin_segmentation_slices, tooth_slices)
        points = restore_coordinates(points, aligned_volume.shape, transform_meta)
        points, upper_surface, lower_surface = split_surface(points, filtered_volume)
        cej_surface = upper_surface if label // 10 in {1, 2} else lower_surface

        tooth_volume = filtered_volume == Label.TOOTH
        bone_volume = filtered_volume == Label.BONE
        bone_dilation = ndimage.binary_dilation(bone_volume)
        bone_surface = tooth_volume & bone_dilation

        cej_surface_count = len(cej_surface)
        bone_surface_count = bone_surface.sum()

        if cej_surface_count == 0 or bone_surface_count == 0:
            bone_loss = 0
        else:
            bone_loss = (cej_surface_count - bone_surface_count) / cej_surface_count
            bone_loss = round(bone_loss * 100)

        bone_losses[label] = bone_loss

    return bone_losses

if __name__ == '__main__':
    import os

    from src.console import Table
    from argparse import ArgumentParser
    from src.config import load_config
    from scripts.post_processing.iterators import iterate_fold_patients
    from scripts.tools.widgets import Mode
    from keypoint.predict import CEJFinder

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    config.split_file_path = os.path.join('logs', experiment_name, f'{config.split_filename}.json')

    cej_finder = CEJFinder('logs/keypoint_baseline/best.pth')

    bone_losses = []

    for fold, dataset, patient in iterate_fold_patients(config):
        volume_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.POST_PROCESSING}.npy')
        volume = numpy.load(volume_path)
        image_volume_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.IMAGE}.npy')
        image_volume = numpy.load(image_volume_path)

        bone_losses = compute_bone_loss(volume, image_volume, cej_finder)

        print(f'{dataset}/{patient}')
        Table(
            ['Label', 'Bone Loss (%)'],
            *[
                [label, bone_loss]
                for label, bone_loss in bone_losses.items()
            ]
        ).display()
