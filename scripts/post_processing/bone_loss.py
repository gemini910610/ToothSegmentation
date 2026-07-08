import numpy

from scipy import ndimage
from scipy.interpolate import RBFInterpolator

from scripts.tools.widgets import Label

def split_surface(points, filtered_segmentation, transform_meta):
    tooth_area = filtered_segmentation == Label.TOOTH
    erosion = ndimage.binary_erosion(tooth_area)
    tooth_surface = tooth_area & ~erosion
    coordinates = numpy.argwhere(tooth_surface)

    center = transform_meta['center']
    axes = transform_meta['axes']
    rotate_matrix = numpy.stack(axes, axis=1)
    aligned_points = (points - center) @ rotate_matrix
    aligned_coordinates = (coordinates - center) @ rotate_matrix

    xy = aligned_points[:, :2]
    z = aligned_points[:, 2]

    split_plane = RBFInterpolator(xy, z)

    surface_xy = aligned_coordinates[:, :2]
    surface_z = aligned_coordinates[:, 2]
    split_z = split_plane(surface_xy)

    upper_mask = surface_z > split_z
    lower_mask = surface_z <= split_z

    upper_surface = coordinates[upper_mask]
    lower_surface = coordinates[lower_mask]

    return upper_surface, lower_surface

def split_volume(points, filtered_segmentation, transform_meta):
    tooth_area = filtered_segmentation == Label.TOOTH
    coordinates = numpy.argwhere(tooth_area)

    center = transform_meta['center']
    axes = transform_meta['axes']
    rotate_matrix = numpy.stack(axes, axis=1)
    aligned_points = (points - center) @ rotate_matrix
    aligned_coordinates = (coordinates - center) @ rotate_matrix

    xy = aligned_points[:, :2]
    z = aligned_points[:, 2]

    split_plane = RBFInterpolator(xy, z)

    volume_xy = aligned_coordinates[:, :2]
    volume_z = aligned_coordinates[:, 2]
    split_z = split_plane(volume_xy)

    upper_mask = volume_z > split_z
    lower_mask = volume_z <= split_z

    upper_volume = coordinates[upper_mask]
    lower_volume = coordinates[lower_mask]

    return upper_volume, lower_volume

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

def slice_bone_loss(segmentation_slice, point, label):
    tooth_slice = segmentation_slice == Label.TOOTH
    bone_slice = segmentation_slice == Label.BONE

    tooth_erosion = ndimage.binary_erosion(tooth_slice)
    tooth_surface = tooth_slice & ~tooth_erosion

    bone_dilation = ndimage.binary_dilation(bone_slice)
    bone_surface = tooth_slice & bone_dilation

    _, left_y, _, right_y = point

    height, width = segmentation_slice.shape
    center = width // 2

    heights = numpy.arange(height)[:, None]
    if label // 10 in {1, 2}:
        left_mask = heights < left_y
        right_mask = heights < right_y
    else:
        left_mask = heights > left_y
        right_mask = heights > right_y
    left_surface = tooth_surface[:, :center] & left_mask
    right_surface = tooth_surface[:, center:] & right_mask

    cej_count = left_surface.sum() + right_surface.sum()
    bone_count = bone_surface.sum()
    bone_loss = 0 if cej_count == 0 else (cej_count - bone_count) / cej_count
    bone_loss = round(bone_loss * 100)

    return bone_loss

def split_area(coordinates, left_point, right_point, is_upper):
    xy = coordinates[:, [1, 0]]
    vector = right_point - left_point
    side = numpy.cross(xy - left_point, vector)

    if is_upper:
        mask = side > 0
    else:
        mask = side < 0

    return mask

def slice_area_bone_loss(segmentation_slice, point, label):
    tooth_slice = segmentation_slice == Label.TOOTH
    bone_slice = segmentation_slice == Label.BONE

    bone_dilation = ndimage.binary_dilation(bone_slice)
    bone_surface = tooth_slice & bone_dilation

    area_mask = numpy.zeros_like(tooth_slice, dtype=numpy.uint8)

    y, x = numpy.nonzero(bone_surface)

    if len(x) == 0:
        return 100, area_mask

    center = (x.min() + x.max()) / 2
    left_mask = x < center
    right_mask = x > center

    left_x = x[left_mask]
    left_y = y[left_mask]
    right_x = x[right_mask]
    right_y = y[right_mask]

    if len(left_x) == 0 or len(right_x) == 0:
        return 100, area_mask

    if label // 10 in {1, 2}:
        index_left = numpy.argmin(left_x - left_y)
        index_right = numpy.argmax(right_x + right_y)
    else:
        index_left = numpy.argmin(left_x + left_y)
        index_right = numpy.argmax(right_x - right_y)

    left_bone = numpy.array([left_x[index_left], left_y[index_left]])
    right_bone = numpy.array([right_x[index_right], right_y[index_right]])

    left_cej_x, left_cej_y, right_cej_x, right_cej_y = point
    left_cej = numpy.array([left_cej_x, left_cej_y])
    right_cej = numpy.array([right_cej_x, right_cej_y])

    coordinates = numpy.argwhere(tooth_slice)

    cej_mask = split_area(coordinates, left_cej, right_cej, label // 10 in {1, 2})
    cej_area = cej_mask.sum()

    bone_mask = split_area(coordinates, left_bone, right_bone, label // 10 in {1, 2})
    bone_area = bone_mask.sum()

    bone_loss = 0 if cej_area == 0 else (cej_area - bone_area) / cej_area
    bone_loss = round(bone_loss * 100)

    area_mask[*coordinates[cej_mask].T] = 1
    area_mask[*coordinates[bone_mask].T] = 2

    return bone_loss, area_mask

def volume_bone_loss(filtered_volume, aligned_volume, points, label, transform_meta):
    points = restore_coordinates(points, aligned_volume.shape, transform_meta)
    upper_surface, lower_surface = split_surface(points, filtered_volume, transform_meta)
    cej_surface = upper_surface if label // 10 in {1, 2} else lower_surface

    tooth_volume = filtered_volume == Label.TOOTH
    bone_volume = filtered_volume == Label.BONE

    bone_dilation = ndimage.binary_dilation(bone_volume)
    bone_surface = tooth_volume & bone_dilation

    cej_surface_count = len(cej_surface)
    bone_surface_count = bone_surface.sum()

    bone_loss = 0 if cej_surface_count == 0 else (cej_surface_count - bone_surface_count) / cej_surface_count
    bone_loss = round(bone_loss * 100)

    return bone_loss
