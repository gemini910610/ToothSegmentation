import numpy

from scipy import ndimage

from scripts.post_processing.tooth_slice import align_crop_tooth, extract_single_tooth, find_normal_vectors, get_slices, normalize_slice, restore_coordinates, split_surface
from scripts.tools.widgets import Label
from src.console import track

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

    y, x = numpy.nonzero(bone_surface)

    if label // 10 in {1, 2}:
        index_left = numpy.argmin(x - y)
        index_right = numpy.argmax(x + y)
    else:
        index_left = numpy.argmin(x + y)
        index_right = numpy.argmax(x - y)

    left_bone = numpy.array([x[index_left], y[index_left]])
    right_bone = numpy.array([x[index_right], y[index_right]])

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

    area_mask = numpy.zeros_like(tooth_slice, dtype=numpy.uint8)
    area_mask[*coordinates[cej_mask].T] = 1
    area_mask[*coordinates[bone_mask].T] = 2

    return bone_loss, area_mask

def volume_bone_loss(segmentation_volume, image_volume, cej_finder):
    normal_vectors = find_normal_vectors(segmentation_volume)

    labels = numpy.unique(segmentation_volume)
    labels = labels[(labels > 1) & (labels < Label.CROPPED)] # exclude background, bone, removed teeth and missing FDI teeth

    bone_losses = {}
    for label in track(labels):
        normal_vector = normal_vectors[label]
        filtered_volume, transform_meta = extract_single_tooth(segmentation_volume, normal_vector, tooth_label=label, bone_label=1)
        aligned_volume, aligned_image = align_crop_tooth(filtered_volume, image_volume, transform_meta)

        slices = get_slices(aligned_volume, aligned_image, reverse=label // 10 in {2, 3})

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
        points = restore_coordinates(points, filtered_volume, aligned_volume.shape, transform_meta)
        upper_surface, lower_surface = split_surface(points, filtered_volume)
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
