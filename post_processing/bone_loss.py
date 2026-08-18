import numpy
from scipy import ndimage
from scipy.interpolate import RBFInterpolator


def split_volume(points, filtered_tooth_area, transform_meta, use_surface=False, aligned=False):
    if use_surface:
        filtered_tooth_area = get_tooth_surface(filtered_tooth_area)
    coordinates = numpy.argwhere(filtered_tooth_area)

    center = transform_meta['center']
    axes = transform_meta['axes']
    rotate_matrix = numpy.stack(axes, axis=1)
    aligned_coordinates = (coordinates - center) @ rotate_matrix
    aligned_points = points if aligned else (points - center) @ rotate_matrix

    xy = aligned_points[:, :2]
    z = aligned_points[:, 2]

    split_plane = RBFInterpolator(xy, z)

    xy = aligned_coordinates[:, :2]
    z = aligned_coordinates[:, 2]
    split_z = split_plane(xy)

    upper_mask = z > split_z
    lower_mask = z <= split_z

    upper_surface = coordinates[upper_mask]
    lower_surface = coordinates[lower_mask]

    return upper_surface, lower_surface

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

    aligned_points = numpy.stack([u, v, w], axis=1)

    rotate_matrix = numpy.stack(axes, axis=1)
    points = aligned_points @ rotate_matrix.T + center

    return points, aligned_points

def get_bone_loss(cej, bone):
    bone_loss = 0 if cej == 0 else (cej - bone) / cej
    bone_loss = round(bone_loss * 100)

    return bone_loss

def get_tooth_surface(tooth_area):
    erosion = ndimage.binary_erosion(tooth_area)
    tooth_surface = tooth_area & ~erosion

    return tooth_surface

def get_bone_surface(tooth_area, bone_area):
    bone_dilation = ndimage.binary_dilation(bone_area)
    bone_surface = tooth_area & bone_dilation

    return bone_surface

def slice_contour_bone_loss(tooth_surface, bone_surface, point, is_upper):
    _, left_y, _, right_y = point

    height, width = tooth_surface.shape
    center = width // 2

    heights = numpy.arange(height)[:, None]
    if is_upper:
        left_mask = heights < left_y
        right_mask = heights < right_y
    else:
        left_mask = heights > left_y
        right_mask = heights > right_y
    left_surface = tooth_surface[:, :center] & left_mask
    right_surface = tooth_surface[:, center:] & right_mask

    cej_count = left_surface.sum() + right_surface.sum()
    bone_count = bone_surface.sum()
    bone_loss = get_bone_loss(cej_count, bone_count)

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

def slice_area_bone_loss(tooth_slice, bone_surface, point, is_upper):
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

    if is_upper:
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

    cej_mask = split_area(coordinates, left_cej, right_cej, is_upper)
    cej_area = cej_mask.sum()

    bone_mask = split_area(coordinates, left_bone, right_bone, is_upper)
    bone_area = bone_mask.sum()

    bone_loss = get_bone_loss(cej_area, bone_area)

    area_mask[*coordinates[cej_mask].T] = 1
    area_mask[*coordinates[bone_mask].T] = 2

    return bone_loss, area_mask

def volume_bone_loss(filtered_tooth_volume, filtered_bone_volume, aligned_points, is_upper, transform_meta):
    upper_surface, lower_surface = split_volume(aligned_points, filtered_tooth_volume, transform_meta, use_surface=True, aligned=True)
    cej_surface = upper_surface if is_upper else lower_surface

    bone_surface = get_bone_surface(filtered_tooth_volume, filtered_bone_volume)

    cej_surface_count = len(cej_surface)
    bone_surface_count = bone_surface.sum()

    bone_loss = get_bone_loss(cej_surface_count, bone_surface_count)

    return bone_loss
