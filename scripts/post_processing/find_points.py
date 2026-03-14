import numpy

from sklearn.cluster import KMeans
from scipy import ndimage
from scripts.tools.widgets import Label

class CEJFinder:
    def __init__(self, tooth_volume):
        mask = tooth_volume != 0
        colors = tooth_volume[mask].reshape(-1, 1)
        self.k_means = KMeans(n_clusters=3, random_state=42).fit(colors)
    def find(self, tooth_slice):
        mask = tooth_slice != 0
        colors = tooth_slice[mask].reshape(-1, 1)

        labels = self.k_means.predict(colors)
        centers = self.k_means.cluster_centers_.reshape(-1)

        order = numpy.argsort(centers)

        ys, xs = numpy.where(mask)
        label = labels == order[-1]
        y = ys[label]
        x = xs[label]

        median = numpy.median(x)
        index_left = x <= median
        index_right = x > median

        x_left = x[index_left]
        y_left = y[index_left]
        index_lb = numpy.argmax(y_left - x_left)
        left_bottom = [x_left[index_lb], y_left[index_lb]]

        x_right = x[index_right]
        y_right = y[index_right]
        index_rb = numpy.argmax(y_right + x_right)
        right_bottom = [x_right[index_rb], y_right[index_rb]]

        return left_bottom, right_bottom

def find_bone_point(segmentation_slice):
    segmentation_slice = segmentation_slice.copy()

    tooth_mask = segmentation_slice == Label.TOOTH
    filled = ndimage.binary_fill_holes(tooth_mask)
    holes = filled & ~tooth_mask
    segmentation_slice[holes] = Label.TOOTH

    bone_tooth_mask = segmentation_slice != Label.BACKGROUND
    filled = ndimage.binary_fill_holes(bone_tooth_mask)
    holes = filled & ~bone_tooth_mask
    segmentation_slice[holes] = Label.BONE

    tooth_mask = segmentation_slice == Label.TOOTH
    bone_mask = segmentation_slice == Label.BONE
    background_mask = segmentation_slice == Label.BACKGROUND

    tooth_dilation = ndimage.binary_dilation(tooth_mask)
    bone_dilation = ndimage.binary_dilation(bone_mask)
    background_dilation = ndimage.binary_dilation(background_mask)
    bone_points = tooth_dilation & bone_dilation & background_dilation

    labels, count = ndimage.label(bone_points)

    centroids = ndimage.center_of_mass(bone_points, labels, range(1, count + 1))
    centroids = numpy.array(centroids)

    if len(centroids) == 0:
        return [0, 0], [0, 0]

    xs = centroids[:, 1]

    mean = (min(xs) + max(xs)) / 2
    left_mask = xs <= mean
    right_mask = xs > mean

    left_centroids = centroids[left_mask]
    right_centroids = centroids[right_mask]

    if len(left_centroids) == 0 or len(right_centroids) == 0:
        return [0, 0], [0, 0]

    left_index = numpy.argmin(left_centroids[:, 0])
    right_index = numpy.argmin(right_centroids[:, 0])

    left_bone = numpy.round(left_centroids[left_index][::-1]).astype(numpy.int32)
    right_bone = numpy.ceil(right_centroids[right_index][::-1]).astype(numpy.int32)

    return left_bone, right_bone

def ensure_upward(segmentation_volume, image_volume, tooth_volume, center):
    tooth_mask = segmentation_volume == Label.TOOTH
    bone_mask = segmentation_volume == Label.BONE
    bone_dilation = ndimage.binary_dilation(bone_mask)
    tooth_surface = tooth_mask & bone_dilation

    coordinates = numpy.argwhere(tooth_surface)
    _, _, surface_z = coordinates.mean(0)
    coordinates = numpy.argwhere(tooth_mask)
    _, _, center_z = coordinates.mean(0)

    if surface_z > center_z:
        segmentation_volume = segmentation_volume[:,:,::-1]
        image_volume = image_volume[:,:,::-1]
        tooth_volume = tooth_volume[:,:,::-1]
        center = center.copy()
        center[2] = segmentation_volume.shape[2] - center[2] - 1

    return segmentation_volume, image_volume, tooth_volume, center

def is_single_root(tooth_volume):
    coordinates = numpy.argwhere(tooth_volume)
    top_z = coordinates[coordinates[:,2].argmax()][2]
    bottom_z = coordinates[coordinates[:,2].argmin()][2]
    height = top_z - bottom_z + 1

    root_top_z = bottom_z + height // 2

    consecutive_split = 0
    max_consecutive_split = 0

    for z in range(bottom_z, root_top_z + 1):
        slice_mask = tooth_volume[:, :, z]
        slice_mask = ndimage.binary_opening(slice_mask)
        slice_mask = ndimage.binary_fill_holes(slice_mask)

        _, count = ndimage.label(slice_mask)
        if count <= 1:
            consecutive_split = 0
            continue
        else:
            consecutive_split += 1
            max_consecutive_split = max(max_consecutive_split, consecutive_split)

    return max_consecutive_split < 10

def find_root(segmentation_volume):
    tooth_volume = segmentation_volume == Label.TOOTH
    tooth_volume = ndimage.binary_fill_holes(tooth_volume)
    tooth_volume = ndimage.binary_opening(tooth_volume, iterations=3)
    tooth_center = ndimage.center_of_mass(tooth_volume)

    tooth_erosion = ndimage.binary_erosion(tooth_volume)
    tooth_surface = tooth_volume & ~tooth_erosion

    coordinates = numpy.argwhere(tooth_surface)
    top_z = coordinates[coordinates[:,2].argmax()][2]
    bottom_z = coordinates[coordinates[:,2].argmin()][2]

    center_x = numpy.rint(tooth_center[0]).astype(numpy.int32)
    center_y = numpy.rint(tooth_center[1]).astype(numpy.int32)
    center_z = numpy.rint(tooth_center[2]).astype(numpy.int32)
    zs = numpy.argwhere(tooth_surface[center_x, center_y])
    zs = zs[zs < center_z]
    center_bottom = zs.max()
    if (top_z - center_bottom) / (top_z - bottom_z) > 0.9 and is_single_root(tooth_volume):
        return [coordinates[coordinates[:,2].argmin()]]

    z_score = top_z - coordinates[:,2]
    center_score = numpy.linalg.norm(coordinates - tooth_center, axis=1)
    surface_distances = z_score + center_score

    distances = numpy.zeros(segmentation_volume.shape, dtype=numpy.float32)
    distances[*coordinates.T] = surface_distances

    center_scores = numpy.zeros_like(distances)
    center_scores[*coordinates.T] = center_score

    local_max = ndimage.maximum_filter(distances, size=(9, 9, 9))
    mask = tooth_surface & (distances == local_max) & (distances > numpy.percentile(surface_distances, 90))

    coordinates = numpy.argwhere(mask)
    coordinates = coordinates[coordinates[:, 2] < (top_z + bottom_z) / 2]

    remove = set()
    for i in range(len(coordinates) - 1):
        for j in range(i + 1, len(coordinates)):
            coordinate_i = coordinates[i]
            coordinate_j = coordinates[j]
            if numpy.linalg.norm(coordinate_i - coordinate_j) < 6.5:
                remove.add(i if distances[*coordinate_i.T] < distances[*coordinate_j.T] else j)
    coordinates = [coordinate for i, coordinate in enumerate(coordinates) if i not in remove]

    return coordinates
