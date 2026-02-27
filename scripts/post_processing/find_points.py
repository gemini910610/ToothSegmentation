import numpy

from sklearn.cluster import KMeans
from scipy import ndimage
from scripts.tools.widgets import Label
from sklearn.cluster import DBSCAN
from skimage.morphology import h_maxima

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

def find_root(segmentation_volume):
    tooth_volume = segmentation_volume == Label.TOOTH
    tooth_volume = ndimage.binary_fill_holes(tooth_volume)

    tooth_erosion = ndimage.binary_erosion(tooth_volume)
    tooth_surface = tooth_volume & ~tooth_erosion

    coordinates = numpy.argwhere(tooth_surface)
    top_z = coordinates[coordinates[:,2].argmax()][2]
    bottom_z = coordinates[coordinates[:,2].argmin()][2]
    surface_distances = coordinates[:,2]
    distances = numpy.zeros(segmentation_volume.shape, dtype=numpy.float32)
    distances[*coordinates.T] = surface_distances

    h = (top_z - coordinates[coordinates[:,2].argmin()][2]) * 0.02
    mask = h_maxima(-distances, h=h)
    coordinates = numpy.argwhere(mask)
    coordinates = coordinates[coordinates[:, 2] < (top_z + bottom_z) / 2]

    roots = []
    clustering = DBSCAN(eps=3, min_samples=1).fit(coordinates)
    labels = clustering.labels_
    for label in numpy.unique(labels):
        if label == -1:
            continue
        root = coordinates[labels == label].mean(0).round().astype(numpy.int32)
        roots.append(root)

    return roots
