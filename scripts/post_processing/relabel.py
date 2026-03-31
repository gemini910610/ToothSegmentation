import numpy

from scipy import ndimage
from scipy.interpolate import CubicSpline
from scripts.tools.widgets import Label

def split_teeth_jaw(tooth_volume, bone_volume):
    upper_tooth = []
    lower_tooth = []

    objects = ndimage.find_objects(tooth_volume)

    for label, slices in enumerate(objects, 1):
        if slices is None:
            continue

        roi = tooth_volume[slices] == label

        tooth_center = ndimage.center_of_mass(roi)
        tooth_center = numpy.array(tooth_center)
        tooth_center += (slices[0].start, slices[1].start, slices[2].start)

        padding_slices = [
            slice(
                max(0, origin_slice.start - 1),
                min(shape, origin_slice.stop + 1)
            )
            for shape, origin_slice in zip(tooth_volume.shape, slices)
        ]

        tooth_roi = tooth_volume[*padding_slices] == label
        bone_roi = bone_volume[*padding_slices]

        tooth_dilation = ndimage.binary_dilation(tooth_roi)
        intersection = bone_roi & tooth_dilation
        if not numpy.any(intersection):
            continue

        intersection_center = ndimage.center_of_mass(intersection)
        intersection_center = numpy.array(intersection_center)
        intersection_center += (padding_slices[0].start, padding_slices[1].start, padding_slices[2].start)

        upward = intersection_center[2] > tooth_center[2]

        (upper_tooth if upward else lower_tooth).append({
            'label': label,
            'slices': slices,
            'roi': roi,
            'center': tooth_center
        })

    if len(upper_tooth) > 0:
        upper_tooth = sort_angle(upper_tooth)
    if len(lower_tooth) > 0:
        lower_tooth = sort_angle(lower_tooth)

    return upper_tooth, lower_tooth

def sort_angle(tooth_list, center=None):
    def get_value(tooth):
        if 'center' in tooth:
            tooth = tooth['center']
        return tooth[:2]

    points = numpy.array([get_value(tooth) for tooth in tooth_list])
    if center is None:
        center = points.mean(0)

    vectors = center - points
    angles = numpy.arctan2(*vectors.T)
    order = numpy.argsort(angles)
    return [tooth_list[index] for index in order]

def estimate_missing_points(points):
    if len(points) < 2:
        return []

    differences = points[1:] - points[:-1]
    distances = numpy.linalg.norm(differences, axis=1)

    t = numpy.concatenate([[0], numpy.cumsum(distances)])
    curve = CubicSpline(t, points, bc_type='natural', axis=0)

    missing_points = []
    threshold = numpy.median(distances)
    for i, distance in enumerate(distances):
        k = round(distance / threshold) - 1
        if k < 1:
            continue

        t_new = numpy.linspace(t[i], t[i+1], k + 2)[1:-1]
        xy_new = curve(t_new)

        for point in xy_new:
            missing_points.append(numpy.array(point))

    return missing_points

def split_left_right(points, default):
    if len(points) < 3:
        return default

    missing_points = estimate_missing_points(points)
    if len(missing_points) > 0:
        points = numpy.concatenate([points, missing_points])

    order = numpy.argsort(points[:, 1])
    points = points[order]
    k = int(len(points) * 0.5) // 2 * 2
    if k < 3:
        return default

    points = points[:k]
    center = numpy.median(points[:, 0])
    return center

def relabel_volume(tooth_volume, bone_volume):
    tooth_volume = tooth_volume.copy()

    removed_mask = tooth_volume >= Label.CROPPED
    removed_tooth = tooth_volume[removed_mask]
    tooth_volume[removed_mask] = 0
    upper_tooth, lower_tooth = split_teeth_jaw(tooth_volume, bone_volume)

    quadrants = {1: [], 2: [], 3: [], 4: []}

    for is_lower, tooth_list in enumerate((upper_tooth, lower_tooth)):
        points = numpy.array([tooth['center'][:2] for tooth in tooth_list])
        center = split_left_right(points, tooth_volume.shape[0] / 2)

        for tooth in tooth_list:
            x = tooth['center'][0]
            is_left = x < center
            quadrant = (
                1 if not is_lower and is_left else
                2 if not is_lower and not is_left else
                3 if is_lower and not is_left else
                4
            )
            quadrants[quadrant].append(tooth)

    volume = numpy.zeros_like(tooth_volume, dtype=numpy.uint8)
    center = (volume.shape[0] / 2, volume.shape[1] / 2)
    for quadrant, tooth_list in quadrants.items():
        if len(tooth_list) == 0:
            continue

        tooth_list = sort_angle(tooth_list, center)
        if quadrant in {2, 3}:
            tooth_list = tooth_list[::-1]
        for index, tooth in enumerate(tooth_list, 1):
            volume[tooth['slices']][tooth['roi']] = quadrant * 10 + index

    volume[removed_mask] = removed_tooth

    return volume

if __name__ == '__main__':
    import os

    from argparse import ArgumentParser
    from src.config import load_config
    from src.console import track
    from src.dataset import get_fold
    from scripts.tools.widgets import Mode

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    config.split_file_path = os.path.join('logs', experiment_name, f'{config.split_filename}.json')

    for fold in range(1, config.num_folds + 1):
        _, valid_dataset_patients = get_fold(config.split_file_path, fold)
        for dataset, patients in valid_dataset_patients.items():
            for patient in track(patients, desc=f'Fold {fold} {dataset}'):
                output_dir = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient)
                tooth_volume_path = os.path.join(output_dir, f'{Mode.TOOTH_FILLED}.npy')
                bone_volume_path = os.path.join(output_dir, f'{Mode.BONE_FILLED}.npy')
                tooth_volume = numpy.load(tooth_volume_path)
                bone_volume = numpy.load(bone_volume_path)

                tooth_volume = relabel_volume(tooth_volume, bone_volume)

                relabeled_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.RELABELED}.npy')
                numpy.save(relabeled_path, tooth_volume)
