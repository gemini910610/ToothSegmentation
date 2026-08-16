import numpy
from scipy import ndimage
from scipy.interpolate import make_splprep

from constants.label import Label


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

    if upper_tooth:
        upper_tooth = sort_angle(upper_tooth)
    if lower_tooth:
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

    vectors = points - center
    angles = numpy.arctan2(*vectors.T)
    angles = (angles + numpy.pi * 2) % (numpy.pi * 2)
    order = numpy.argsort(angles)
    return [tooth_list[index] for index in order]

def estimate_missing_points(points):
    if len(points) < 4:
        return []

    differences = points[1:] - points[:-1]
    distances = numpy.linalg.norm(differences, axis=1)

    curve, u = make_splprep(points.T)

    missing_points = []
    threshold = numpy.median(distances)
    for i, distance in enumerate(distances):
        k = round(distance / threshold) - 1
        if k < 1:
            continue

        u_new = numpy.linspace(u[i], u[i+1], k + 2)[1:-1]
        x_new, y_new = curve(u_new)

        for point in zip(x_new, y_new):
            missing_points.append(numpy.array(point))

    return missing_points

def split_left_right(points, default):
    if len(points) == 0:
        return default, []

    missing_points = estimate_missing_points(points)
    if missing_points:
        points = numpy.concatenate([points, missing_points])

    order = numpy.argsort(points[:, 1])
    points = points[order]
    k = int(len(points) * 0.5) // 2 * 2
    if k < 3:
        return default, []

    points = points[:k]
    center = numpy.median(points[:, 0])
    return center, missing_points

def get_quadrant(is_lower, is_left):
    if not is_lower and is_left:
        return 1

    if not is_lower and not is_left:
        return 2

    if is_lower and not is_left:
        return 3

    return 4

def relabel_volume(tooth_volume, bone_volume):
    tooth_volume = tooth_volume.copy()

    exclude_mask = tooth_volume >= Label.TRUNCATED
    exclude_tooth = tooth_volume[exclude_mask]
    tooth_volume[exclude_mask] = 0
    upper_tooth, lower_tooth = split_teeth_jaw(tooth_volume, bone_volume)

    quadrants = {1: [], 2: [], 3: [], 4: []}

    for is_lower, tooth_list in enumerate((upper_tooth, lower_tooth)):
        points = numpy.array([tooth['center'][:2] for tooth in tooth_list])
        center, missing_points = split_left_right(points, tooth_volume.shape[0] / 2)

        for tooth in tooth_list:
            x = tooth['center'][0]
            is_left = x < center
            quadrant = get_quadrant(is_lower, is_left)
            quadrants[quadrant].append(tooth)

        for point in missing_points:
            x = point[0]
            is_left = x < center
            quadrant = get_quadrant(is_lower, is_left)
            quadrants[quadrant].append({
                'label': None,
                'center': point
            })

    volume = numpy.zeros_like(tooth_volume, dtype=numpy.uint8)
    center = (volume.shape[0] / 2, volume.shape[1] / 2)
    for quadrant, tooth_list in quadrants.items():
        if not tooth_list:
            continue

        tooth_list = sort_angle(tooth_list, center)
        if quadrant in {2, 3}:
            tooth_list = tooth_list[::-1]
        for index, tooth in enumerate(tooth_list, 1):
            if tooth['label'] is None:
                continue
            volume[tooth['slices']][tooth['roi']] = quadrant * 10 + index

    volume[exclude_mask] = exclude_tooth

    return volume

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

        experiment_name = args['experiment_name']
        datasets = args['datasets']
    else:
        parser = ArgumentParser()
        parser.add_argument('exp', type=str)
        parser.add_argument('datasets', nargs='+', type=str)
        args = parser.parse_args()

        experiment_name = args.exp
        datasets = args.datasets

    for dataset, patient in iterate_datasets(experiment_name, datasets):
        tooth_path = Path('outputs') / experiment_name / dataset / patient / f'{Mode.TOOTH_FILLED}.npy'
        tooth_volume = numpy.load(tooth_path)
        bone_path = tooth_path.parent / f'{Mode.BONE_FILLED}.npy'
        bone_volume = numpy.load(bone_path)

        volume = relabel_volume(tooth_volume, bone_volume)

        output_path = tooth_path.parent / f'{Mode.RELABELED}.npy'
        numpy.save(output_path, volume)
