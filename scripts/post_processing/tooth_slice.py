import numpy

from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree
from scripts.tools.widgets import Label
from scripts.post_processing.relabel import split_teeth_jaw
from collections import defaultdict, deque

def extract_single_tooth(segmentation_volume, normal_vector, tooth_label, bone_label, padding=25):
    tooth_mask = segmentation_volume == tooth_label
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

    coordinates = numpy.argwhere((segmentation_volume == tooth_label) | (segmentation_volume == bone_label))
    uvw = (coordinates - tooth_center) @ rotate_matrix

    mask = (
        (uvw[:, 0] >= u_min) & (uvw[:, 0] <= u_max) &
        (uvw[:, 1] >= v_min) & (uvw[:, 1] <= v_max) &
        (uvw[:, 2] >= w_min) & (uvw[:, 2] <= w_max)
    )
    coordinates = coordinates[mask]

    filtered_segmentation = numpy.zeros_like(segmentation_volume, dtype=numpy.uint8)
    values = segmentation_volume[*coordinates.T]
    filtered_segmentation[*coordinates.T] = (values == tooth_label) * Label.TOOTH + (values == bone_label) * Label.BONE

    transform_meta = {
        'center': tooth_center,
        'length': length,
        'axes': [axis_x, axis_y, axis_z]
    }

    return filtered_segmentation, transform_meta

def align_crop_tooth(segmentation_volume, image_volume, transform_meta):
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
    aligned_image = ndimage.map_coordinates(image_volume, coordinates, order=3)

    return aligned_segmentation, aligned_image

def oblique_slice(segmentation_volume, image_volume, theta, offset=90):
    width, height, depth = segmentation_volume.shape
    center = numpy.array([width, height, depth]) / 2

    size = numpy.sqrt(width ** 2 + height ** 2)
    h_line = numpy.linspace(-size / 2, size / 2, int(size))
    v_line = numpy.arange(depth)

    hh, vv = numpy.meshgrid(h_line, v_line, indexing='ij')

    radian = numpy.deg2rad(theta + offset)

    xs = center[0] + hh * numpy.cos(radian)
    ys = center[1] + hh * numpy.sin(radian)
    zs = vv

    coordinates = numpy.stack([xs, ys, zs], axis=0)
    segmentation_slice = ndimage.map_coordinates(segmentation_volume, coordinates, order=0)
    segmentation_slice = segmentation_slice.T[::-1]
    image_slice = ndimage.map_coordinates(image_volume, coordinates, order=3)
    image_slice = image_slice.T[::-1]

    return segmentation_slice, image_slice

def get_slices(segmentation_volume, image_volume, count=8, reverse=False, offset=90):
    slices = []
    for theta in numpy.linspace(0, 360, count, endpoint=False):
        if reverse:
            theta *= -1
        segmentation_slice, image_slice = oblique_slice(segmentation_volume, image_volume, theta, offset)
        tooth_slice = (segmentation_slice == 1) * image_slice
        slices.append((segmentation_slice, image_slice, tooth_slice))
    return slices

def normalize_slice(segmentation_slice, image_slice, tooth_slice):
    segmentation_slice = segmentation_slice / 2 * 255
    segmentation_slice = segmentation_slice.astype(numpy.uint8)

    image_slice = image_slice.clip(0, 255)
    image_slice = image_slice.astype(numpy.uint8)

    tooth_slice = tooth_slice.clip(0, 255)
    tooth_slice = tooth_slice.astype(numpy.uint8)

    return segmentation_slice, image_slice, tooth_slice

def find_normal_vectors(segmentation_volume):
    tooth_volume = segmentation_volume.copy()
    tooth_volume[(segmentation_volume < 2) | (segmentation_volume >= Label.UNERUPTED)] = 0
    bone_volume = segmentation_volume == 1

    upper_tooth, lower_tooth = split_teeth_jaw(tooth_volume, bone_volume)

    normal_vectors = {}

    for tooth_list in (upper_tooth, lower_tooth):
        if len(tooth_list) < 2:
            continue

        points = numpy.array([tooth['center'][:2] for tooth in tooth_list])
        center = points.mean(0)
        vectors = points - center

        differences = points[1:] - points[:-1]
        distances = numpy.linalg.norm(differences, axis=1)

        t = numpy.concatenate([[0], numpy.cumsum(distances)])
        curve = CubicSpline(t, points, bc_type='natural', axis=0)

        tangents = curve(t, 1)
        normals = numpy.zeros_like(tangents)
        normals[:, 0] = tangents[:, 1]
        normals[:, 1] = -tangents[:, 0]

        normals[numpy.sum(normals * vectors, axis=1) < 0] *= -1

        for normal, tooth in zip(normals, tooth_list):
            normal_vectors[tooth['label']] = (*normal, 0)

    return normal_vectors

def restore_coordinates(points, aligned_shape, transform_meta, offset=90):
    xs = points[:, 0]
    ys = points[:, 1]

    width, height, depth = aligned_shape

    center = transform_meta['center']
    axes = transform_meta['axes']
    length = transform_meta['length']
    u_min, v_min, w_min = -length

    w = depth - 1 - ys + w_min

    u_size = width
    v_size = height
    size = numpy.sqrt(u_size ** 2 + v_size ** 2)
    hh = xs - size / 2

    thetas = numpy.linspace(0, 360, len(points), endpoint=False)
    radian = numpy.deg2rad(thetas + offset)
    u = (u_size / 2) + hh * numpy.cos(radian) + u_min
    v = (v_size / 2) + hh * numpy.sin(radian) + v_min

    uvw = numpy.stack([u, v, w], axis=1)

    rotate_matrix = numpy.stack(axes, axis=1)
    points = uvw @ rotate_matrix.T + center

    return points

def find_path(coordinates, points):
    tree = KDTree(coordinates)
    _, indices = tree.query(points)

    pairs = tree.query_pairs(1.75) # sqrt(3) 26-connectivity
    neighbors = defaultdict(list)
    for i, j in pairs:
        neighbors[i].append(j)
        neighbors[j].append(i)

    path_indices = []
    length = len(indices)
    for index in range(length):
        path = bfs(neighbors, indices[index], indices[(index + 1) % length])
        path_indices.extend(path[1:])

    path_coordinates = coordinates[path_indices]

    return path_coordinates

def split_surface(points, filtered_segmentation):
    tooth_area = filtered_segmentation == Label.TOOTH
    erosion = ndimage.binary_erosion(tooth_area)
    tooth_surface = tooth_area & ~erosion
    coordinates = numpy.argwhere(tooth_surface)

    path_coordinates = find_path(coordinates, points)

    path_mask = numpy.zeros_like(filtered_segmentation, dtype=bool)
    path_mask[*path_coordinates.T] = True
    structure = ndimage.generate_binary_structure(3, 3)
    path_mask = ndimage.binary_dilation(path_mask, structure)

    surface_mask = numpy.zeros_like(filtered_segmentation, dtype=bool)
    surface_mask[*coordinates.T] = True
    surface_mask[path_mask] = False

    labels = numpy.zeros_like(filtered_segmentation, dtype=numpy.uint8)

    upper_seed = coordinates[numpy.argmax(coordinates[:, 2])]
    lower_seed = coordinates[numpy.argmin(coordinates[:, 2])]

    for label, seed in enumerate((upper_seed, lower_seed), 1):
        seed_mask = numpy.zeros_like(surface_mask)
        seed_mask[*seed.T] = True
        seed_surface = ndimage.binary_propagation(seed_mask, structure, surface_mask)
        labels[seed_surface] = label

    unassigned_surface = tooth_surface & (labels == 0)
    unassigned_surface[*path_coordinates.T] = False
    unassigned_coordinates = numpy.argwhere(unassigned_surface)

    assigned_mask = labels > 0
    assigned_coordinates = numpy.argwhere(assigned_mask)
    assigned_labels = labels[assigned_mask]

    tree = KDTree(assigned_coordinates)
    _, indices = tree.query(unassigned_coordinates)

    labels[*unassigned_coordinates.T] = assigned_labels[indices]

    upper_surface = numpy.argwhere(labels == 1)
    lower_surface = numpy.argwhere(labels == 2)

    return path_coordinates, upper_surface, lower_surface

def bfs(neighbors, start_index, end_index):
    if start_index == end_index:
        return [start_index]

    queue = deque([[start_index]])
    visited = {start_index}

    while queue:
        path = queue.popleft()
        node = path[-1]

        for neighbor in neighbors[node]:
            if neighbor == end_index:
                return path + [neighbor]

            if neighbor not in visited:
                queue.append(path + [neighbor])
                visited.add(neighbor)

    return None

if __name__ == '__main__':
    import os
    import cv2

    from argparse import ArgumentParser
    from src.config import load_config
    from scripts.post_processing.iterators import iterate_fold_patients
    from scripts.tools.widgets import Mode

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    experiment_name = args.exp
    output_dir = args.output

    if os.path.exists(output_dir):
        raise FileExistsError(f'folder "{output_dir}" already exists')

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    config.split_file_path = os.path.join('logs', experiment_name, f'{config.split_filename}.json')

    for fold, dataset, patient in iterate_fold_patients(config):
        volume_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.POST_PROCESSING}.npy')
        volume = numpy.load(volume_path)
        image_volume_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.IMAGE}.npy')
        image_volume = numpy.load(image_volume_path)

        normal_vectors = find_normal_vectors(volume)

        labels = numpy.unique(volume)
        labels = labels[(labels > 1) & (labels < Label.CROPPED)] # exclude background, bone and removed teeth

        for label in labels:
            output_path = os.path.join(output_dir, dataset, patient, str(label))
            os.makedirs(output_path)

            normal_vector = normal_vectors[label]
            filtered_volume, transform_meta = extract_single_tooth(volume, normal_vector, tooth_label=label, bone_label=1)
            aligned_volume, aligned_image = align_crop_tooth(filtered_volume, image_volume, transform_meta)
            slices = get_slices(aligned_volume, aligned_image, reverse=label // 10 in {2, 3})

            degrees = numpy.linspace(0, 360, len(slices), endpoint=False)
            for degree, (segmentation_slice, image_slice, tooth_slice) in zip(degrees, slices):
                if degree.is_integer():
                    degree = int(degree)
                segmentation_slice, image_slice, tooth_slice = normalize_slice(segmentation_slice, image_slice, tooth_slice)
                for title, image in {'segmentation': segmentation_slice, 'image': image_slice, 'tooth': tooth_slice}.items():
                    cv2.imwrite(os.path.join(output_path, f'{title}_{degree}.png'), image)
