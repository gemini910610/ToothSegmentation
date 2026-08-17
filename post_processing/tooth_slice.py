import numpy
from scipy import ndimage
from scipy.interpolate import make_splprep

from constants.label import Label
from post_processing.relabel import split_teeth_jaw


def extract_single_tooth(tooth_mask, bone_mask, bone_coordinates, normal_vector, padding=25):
    tooth_coordinates = numpy.argwhere(tooth_mask)
    tooth_center = tooth_coordinates.mean(0)

    centered = tooth_coordinates - tooth_center
    cov = centered.T @ centered
    eigen_values, eigen_vectors = numpy.linalg.eigh(cov)
    order = numpy.argsort(eigen_values)[::-1]
    axis = eigen_vectors[:, order]

    axis_z = axis[:, 0]
    if axis_z[2] < 0:
        axis_z *= -1 # 朝上
    axis_y = numpy.cross(axis_z, normal_vector) # 朝右
    axis_y = axis_y / numpy.linalg.norm(axis_y)
    axis_x = numpy.cross(axis_y, axis_z) # 朝前
    axis_x = axis_x / numpy.linalg.norm(axis_x)
    rotate_matrix = numpy.stack([axis_x, axis_y, axis_z], axis=1)

    uvw_tooth = (tooth_coordinates - tooth_center) @ rotate_matrix
    length = numpy.abs([uvw_tooth.min(0), uvw_tooth.max(0)]).max(0) + padding
    u_min, v_min, w_min = -length
    u_max, v_max, w_max = length

    coordinates = numpy.concatenate([tooth_coordinates, bone_coordinates])
    uvw = (coordinates - tooth_center) @ rotate_matrix

    mask = (
        (uvw[:, 0] >= u_min) & (uvw[:, 0] <= u_max) &
        (uvw[:, 1] >= v_min) & (uvw[:, 1] <= v_max) &
        (uvw[:, 2] >= w_min) & (uvw[:, 2] <= w_max)
    )
    coordinates = coordinates[mask]

    filtered_segmentation = numpy.zeros_like(tooth_mask, dtype=numpy.uint8)
    filtered_segmentation[*coordinates.T] = tooth_mask[*coordinates.T] * Label.TOOTH + bone_mask[*coordinates.T] * Label.BONE

    transform_meta = {
        'center': tooth_center,
        'length': length,
        'axes': [axis_x, axis_y, axis_z]
    }

    return filtered_segmentation, transform_meta

def align_crop_tooth(segmentation_volume, image_coefficients, transform_meta):
    center = transform_meta['center']
    length = transform_meta['length']
    axes = transform_meta['axes']

    u_min, v_min, w_min = -length
    u_max, v_max, w_max = length

    u = numpy.arange(numpy.ceil(u_max - u_min).astype(numpy.int32)) + u_min
    v = numpy.arange(numpy.ceil(v_max - v_min).astype(numpy.int32)) + v_min
    w = numpy.arange(numpy.ceil(w_max - w_min).astype(numpy.int32)) + w_min
    u, v, w = numpy.meshgrid(u, v, w, indexing='ij')
    uvw = numpy.stack([u, v, w], axis=-1)

    rotate_matrix = numpy.stack(axes, axis=1)
    coordinates = uvw @ rotate_matrix.T + center
    coordinates = coordinates.transpose(3, 0, 1, 2)
    aligned_segmentation = ndimage.map_coordinates(segmentation_volume, coordinates, order=0)
    aligned_image = ndimage.map_coordinates(image_coefficients, coordinates, order=3, prefilter=False)

    return aligned_segmentation, aligned_image

def oblique_slice(segmentation_volume, image_coefficients, grid, theta, offset=90):
    center = grid['center']
    hh = grid['hh']
    vv = grid['vv']

    radian = numpy.deg2rad(theta + offset)

    xs = center[0] + hh * numpy.cos(radian)
    ys = center[1] + hh * numpy.sin(radian)
    zs = vv

    coordinates = numpy.stack([xs, ys, zs], axis=0)
    segmentation_slice = ndimage.map_coordinates(segmentation_volume, coordinates, order=0)
    segmentation_slice = segmentation_slice.T[::-1]
    image_slice = ndimage.map_coordinates(image_coefficients, coordinates, order=3, prefilter=False)
    image_slice = image_slice.T[::-1]

    tooth_slice = segmentation_slice == Label.TOOTH
    tooth_slice = ndimage.binary_fill_holes(tooth_slice)

    mask = segmentation_slice > 0
    mask = ndimage.binary_fill_holes(mask)
    mask = mask & ~tooth_slice

    segmentation_slice = tooth_slice * Label.TOOTH + mask * Label.BONE

    return segmentation_slice, image_slice

def get_slices(segmentation_volume, image_volume, count=8, reverse=False, offset=90):
    image_coefficients = ndimage.spline_filter(image_volume, order=3)

    width, height, depth = segmentation_volume.shape
    size = numpy.sqrt(width ** 2 + height ** 2)
    h_line = numpy.linspace(-size / 2, size / 2, int(size))
    v_line = numpy.arange(depth)
    hh, vv = numpy.meshgrid(h_line, v_line, indexing='ij')
    grid = {
        'center': numpy.array([width, height, depth]) / 2,
        'hh': hh,
        'vv': vv
    }

    slices = []
    for theta in numpy.linspace(0, 360, count, endpoint=False):
        if reverse:
            theta *= -1
        segmentation_slice, image_slice = oblique_slice(segmentation_volume, image_coefficients, grid, theta, offset)
        slices.append((segmentation_slice, image_slice))

    return slices

def normalize_slice(segmentation_slice, image_slice):
    segmentation_slice = segmentation_slice / 2 * 255
    segmentation_slice = segmentation_slice.astype(numpy.uint8)

    image_slice = image_slice.clip(0, 255)
    image_slice = image_slice.astype(numpy.uint8)

    return segmentation_slice, image_slice

def find_normal_vectors(segmentation_volume):
    tooth_volume = segmentation_volume.copy()
    tooth_volume[(segmentation_volume < 2) | ((segmentation_volume >= Label.UNERUPTED) & (segmentation_volume < Label.MISSING_FDI))] = 0
    bone_volume = segmentation_volume == 1

    upper_tooth, lower_tooth = split_teeth_jaw(tooth_volume, bone_volume)

    normal_vectors = {}

    for tooth_list in (upper_tooth, lower_tooth):
        if len(tooth_list) < 2:
            continue

        points = numpy.array([tooth['center'][:2] for tooth in tooth_list])
        center = points.mean(0)
        vectors = points - center

        curve, u = make_splprep(points.T)

        tangents = curve(u, 1).T
        normals = numpy.zeros_like(tangents)
        normals[:, 0] = tangents[:, 1]
        normals[:, 1] = -tangents[:, 0]

        normals[numpy.sum(normals * vectors, axis=1) < 0] *= -1

        for normal, tooth in zip(normals, tooth_list):
            normal_vectors[tooth['label']] = (*normal, 0)

    return normal_vectors

if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser
    from pathlib import Path

    import cv2

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

    output_dir = Path('output_slices') / experiment_name
    if output_dir.exists():
        raise FileExistsError(f'folder "{output_dir}" already exists')

    for dataset, patient in iterate_datasets(experiment_name, datasets):
        fdi_volume_path = Path('outputs') / experiment_name / dataset / patient / f'{Mode.FDI}.npy'
        if not fdi_volume_path.exists():
            continue

        fdi_volume = numpy.load(fdi_volume_path)
        bone_volume_path = fdi_volume_path.parent / f'{Mode.BONE_FILLED}.npy'
        bone_volume = numpy.load(bone_volume_path)
        image_volume_path = fdi_volume_path.parent / f'{Mode.IMAGE}.npy'
        image_volume = numpy.load(image_volume_path)

        segmentation_volume = fdi_volume + (bone_volume > 0)
        bone_mask = segmentation_volume == 1
        bone_coordinates = numpy.argwhere(bone_mask)
        image_coefficients = ndimage.spline_filter(image_volume, order=3)

        normal_vectors = find_normal_vectors(segmentation_volume)

        labels = numpy.unique(segmentation_volume)
        labels = labels[(labels > 1) & (labels < Label.TRUNCATED)] # exclude background, bone and removed teeth

        for label in labels:
            slice_dir = output_dir / dataset / patient / str(label)
            slice_dir.mkdir(parents=True)

            normal_vector = normal_vectors[label]
            filtered_volume, transform_meta = extract_single_tooth(segmentation_volume == label, bone_mask, bone_coordinates, normal_vector)
            aligned_volume, aligned_image = align_crop_tooth(filtered_volume, image_coefficients, transform_meta)
            slices = get_slices(aligned_volume, aligned_image, reverse=label // 10 in {2, 3})

            degrees = numpy.linspace(0, 360, len(slices), endpoint=False, dtype=numpy.int32)
            for degree, (segmentation_slice, image_slice) in zip(degrees, slices):
                segmentation_slice, image_slice = normalize_slice(segmentation_slice, image_slice)

                segmentation_path = slice_dir / f'segmentation_{degree}.png'
                cv2.imwrite(segmentation_path, segmentation_slice)
                image_path = segmentation_path.parent / f'image_{degree}.png'
                cv2.imwrite(image_path, image_slice)
