import numpy

from sklearn.cluster import KMeans
from scipy import ndimage
from scripts.tools.widgets import Label
from skimage.morphology import h_minima

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

    xs = centroids[:, 1]

    mean = (min(xs) + max(xs)) / 2
    left_mask = xs <= mean
    right_mask = xs > mean

    left_centroids = centroids[left_mask]
    right_centroids = centroids[right_mask]

    left_index = numpy.argmin(left_centroids[:, 0])
    right_index = numpy.argmin(right_centroids[:, 0])

    left_bone = numpy.round(left_centroids[left_index][::-1]).astype(numpy.int32)
    right_bone = numpy.ceil(right_centroids[right_index][::-1]).astype(numpy.int32)

    return left_bone, right_bone

def ensure_upward(segmentation_volume, image_volume, tooth_volume, center):
    tooth_mask = segmentation_volume == Label.TOOTH
    z = tooth_mask.any((0, 1)).argmax()
    x, y = numpy.argwhere(tooth_mask[:, :, z])[0]

    if z != 0 and segmentation_volume[x, y, z - 1] != Label.BONE:
        segmentation_volume = segmentation_volume[:,:,::-1]
        image_volume = image_volume[:,:,::-1]
        tooth_volume = tooth_volume[:,:,::-1]
        center = center.copy()
        center[2] = segmentation_volume.shape[2] - center[2] - 1

    return segmentation_volume, image_volume, tooth_volume, center

def find_root(segmentation_volume):
    tooth_volume = segmentation_volume == Label.TOOTH

    any_xy = tooth_volume.any(2) # z
    z_min_index = numpy.argmax(tooth_volume, 2).astype(numpy.int32)
    z_min_index[~any_xy] = segmentation_volume.shape[2]

    z_min = ndimage.gaussian_filter(z_min_index, sigma=1)
    min_mask = h_minima(z_min, h=3) & any_xy

    labels, count = ndimage.label(min_mask)
    min_points = []
    for label in range(1, count + 1):
        xs, ys = numpy.where(labels == label)
        index = numpy.argmin(z_min_index[xs, ys])
        x, y = int(xs[index]), int(ys[index])
        z = int(z_min_index[x, y])
        min_points.append([x, y, z])

    return min_points
