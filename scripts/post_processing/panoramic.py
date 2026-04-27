import numpy
import cv2

from scipy import ndimage
from scipy.interpolate import make_splprep
from scripts.tools.widgets import VolumeColorizer, Label
from scripts.post_processing.relabel import split_teeth_jaw

def extract_curve(tooth_list):
    points = numpy.array([tooth['center'][:2] for tooth in tooth_list])
    curve, _ = make_splprep(points.T)

    dense_u = numpy.linspace(0, 1, 100)
    curve_points = curve(dense_u).T
    distances = numpy.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1)
    s = numpy.concatenate([[0], numpy.cumsum(distances)])
    uniform_s = numpy.linspace(0, s[-1], 400)
    uniform_u = numpy.interp(uniform_s, s, dense_u)
    curve_points = curve(uniform_u).T

    tangents = curve(uniform_u, 1).T
    tangents = tangents / numpy.linalg.norm(tangents, axis=1, keepdims=True)

    step = uniform_s[1] - uniform_s[0]
    previous_vector = -tangents[0] * step
    next_vector = tangents[-1] * step
    padding = 40
    previous_points = [curve_points[0] + previous_vector * i for i in range(padding, 0, -1)]
    next_points = [curve_points[-1] + next_vector * i for i in range(1, padding + 1)]
    curve_points = numpy.concatenate([previous_points, curve_points, next_points])
    tangents = numpy.concatenate([[tangents[0]] * len(previous_points), tangents, [tangents[-1]] * len(next_points)])

    normals = numpy.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    normals = normals / numpy.linalg.norm(normals, axis=1, keepdims=True)

    return curve_points, normals

def sample_panoramic(image_volume, segmentation_volume, tooth_list):
    curve_points, normals = extract_curve(tooth_list)

    min_z = min(tooth['slices'][2].start for tooth in tooth_list)
    max_z = max(tooth['slices'][2].stop for tooth in tooth_list)
    center_z = (min_z + max_z) / 2
    z = numpy.array([0, 0, 1])
    vertical = max_z - min_z

    widths = numpy.array([tooth['slices'][0].stop - tooth['slices'][0].start for tooth in tooth_list])
    heights = numpy.array([tooth['slices'][1].stop - tooth['slices'][1].start for tooth in tooth_list])
    horizontal = numpy.concatenate([widths, heights]).max()

    image_panoramic = []
    segmentation_panoramic = []

    for point, normal in zip(curve_points, normals):
        u = numpy.linspace(-horizontal / 2, horizontal / 2, horizontal)
        v = numpy.linspace(vertical / 2, -vertical / 2, vertical)
        uu, vv = numpy.meshgrid(u, v, indexing='ij')

        point = numpy.array([point[0], point[1], center_z])
        normal = numpy.array([normal[0], normal[1], 0])
        coordinates = point + uu[:, :, None] * normal + vv[:, :, None] * z
        coordinates = coordinates.transpose(2, 0, 1)

        image = ndimage.map_coordinates(image_volume, coordinates, order=0)
        image = image.mean(0)
        image_panoramic.append(image)

        segmentation = ndimage.map_coordinates(segmentation_volume, coordinates, order=0)
        segmentation = segmentation.max(0)
        segmentation_panoramic.append(segmentation)

    image_panoramic = numpy.stack(image_panoramic)
    image_panoramic = image_panoramic[::-1].T

    segmentation_panoramic = numpy.stack(segmentation_panoramic)
    segmentation_panoramic = segmentation_panoramic[::-1].T

    return image_panoramic, segmentation_panoramic

def get_panoramic(image_volume, segmentation_volume):
    tooth_volume = segmentation_volume.copy()
    tooth_volume[(segmentation_volume < 2) | (segmentation_volume >= Label.UNERUPTED)] = 0
    bone_volume = segmentation_volume == 1

    jaw_labels = []
    jaw_image_panoramic = []
    jaw_segmentation_panoramic = []

    upper_tooth, lower_tooth = split_teeth_jaw(tooth_volume, bone_volume)

    for tooth_list in (upper_tooth, lower_tooth):
        if len(tooth_list) == 0:
            jaw_labels.append([])
            continue
        labels = [tooth['label'] for tooth in tooth_list[::-1]]
        jaw_mask = numpy.isin(segmentation_volume, labels)
        jaw_image_volume = numpy.where(jaw_mask, image_volume, 0)
        jaw_segmentation_volume = numpy.where(jaw_mask, segmentation_volume, 0)
        image_panoramic, segmentation_panoramic = sample_panoramic(jaw_image_volume, jaw_segmentation_volume, tooth_list)

        jaw_labels.append(labels)
        jaw_image_panoramic.append(image_panoramic)
        jaw_segmentation_panoramic.append(segmentation_panoramic)

    image_panoramic = numpy.vstack(jaw_image_panoramic)
    segmentation_panoramic = numpy.vstack(jaw_segmentation_panoramic)

    low = image_panoramic.min()
    high = image_panoramic.max()
    image_panoramic = (image_panoramic - low) / (high - low)
    image_panoramic = (image_panoramic * 255).astype(numpy.uint8)

    return jaw_labels, image_panoramic, segmentation_panoramic

def draw_label(image_panoramic, segmentation_panoramic):
    image_panoramic = cv2.cvtColor(image_panoramic, cv2.COLOR_GRAY2RGB)
    image_panoramic = cv2.resize(image_panoramic, (0, 0), fx=2, fy=2)
    segmentation_panoramic = cv2.resize(segmentation_panoramic, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

    tooth_count = segmentation_panoramic.max()
    palette = VolumeColorizer.glasbey_palette(tooth_count)

    labels = numpy.unique(segmentation_panoramic)
    labels = labels[labels > 0]
    for label in labels:
        area = segmentation_panoramic == label
        area = ndimage.binary_fill_holes(area)
        dilation = ndimage.binary_dilation(area)
        erosion = ndimage.binary_erosion(area)
        edge = dilation & ~erosion

        color = palette[label - 1][:3]
        image_panoramic[edge] = color

        points = numpy.argwhere(area)
        center = points.mean(0)[::-1]
        (width, height), baseline = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        x = int(center[0] - width / 2)
        y = int(center[1] + (height - baseline) / 2)
        cv2.putText(image_panoramic, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image_panoramic, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    image_panoramic = cv2.cvtColor(image_panoramic, cv2.COLOR_RGB2BGR)
    return image_panoramic

if __name__ == '__main__':
    import os

    from argparse import ArgumentParser
    from src.config import load_config
    from scripts.post_processing.iterators import iterate_fold_patients
    from scripts.tools.widgets import Mode

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    config.split_file_path = os.path.join('logs', experiment_name, f'{config.split_filename}.json')

    for fold, dataset, patient in iterate_fold_patients(config):
        volume_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.POST_PROCESSING}.npy')
        volume = numpy.load(volume_path)
        image_volume_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.IMAGE}.npy')
        image_volume = numpy.load(image_volume_path)

        jaw_labels, image_panoramic, segmentation_panoramic = get_panoramic(image_volume, volume)
        image_panoramic = draw_label(image_panoramic, segmentation_panoramic)

        print(f'{dataset}/{patient}')
        print(jaw_labels[0])
        print(jaw_labels[1])
        cv2.imshow('panoramic', image_panoramic)
        cv2.waitKey(0)
