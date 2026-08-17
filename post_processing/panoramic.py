import copy
from pathlib import Path

import cv2
import numpy
from scipy import ndimage
from scipy.interpolate import make_splprep

from constants.color import GLASBEY_PALETTE
from constants.label import Label
from post_processing.relabel import split_teeth_jaw


def synthetic_tooth(tooth_list, volume_width):
    if len(tooth_list) == 1:
        x, y, z = tooth_list[0]['center']
        fake_tooth = tooth_list[0] | {'center': (volume_width - x, y, z)}
        tooth_list = [tooth_list[0], fake_tooth]
        tooth_list.sort(key=lambda tooth: tooth['center'][0], reverse=True)

    left_x, left_y, left_z = tooth_list[0]['center']
    right_x, right_y, right_z = tooth_list[1]['center']
    center_x = (left_x + right_x) / 2
    center_y = (left_y + right_y) / 2 / 2
    center_z = (left_z + right_z) / 2

    center = ((left_x + center_x) / 2, (left_y + center_y) / 2, (left_z + center_z) / 2)
    fake_tooth = tooth_list[0] | {'center': center}
    tooth_list.insert(1, fake_tooth)

    center = ((center_x + right_x) / 2, (center_y + right_y) / 2, (center_z + right_z) / 2)
    fake_tooth = tooth_list[0] | {'center': center}
    tooth_list.insert(2, fake_tooth)

    return tooth_list

def extract_curve(tooth_list, volume_width, base_width=400, padding=40):
    if len(tooth_list) < 4:
        tooth_list = synthetic_tooth(tooth_list, volume_width)

    points = numpy.array([tooth['center'][:2] for tooth in tooth_list])
    curve, _ = make_splprep(points.T)

    dense_u = numpy.linspace(0, 1, 100)
    curve_points = curve(dense_u).T
    distances = numpy.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1)
    s = numpy.concatenate([[0], numpy.cumsum(distances)])
    uniform_s = numpy.linspace(0, s[-1], base_width)
    uniform_u = numpy.interp(uniform_s, s, dense_u)
    curve_points = curve(uniform_u).T

    tangents = curve(uniform_u, 1).T
    tangents = tangents / numpy.linalg.norm(tangents, axis=1, keepdims=True)

    step = uniform_s[1] - uniform_s[0]
    previous_vector = -tangents[0] * step
    next_vector = tangents[-1] * step
    previous_points = [curve_points[0] + previous_vector * i for i in range(padding, 0, -1)]
    next_points = [curve_points[-1] + next_vector * i for i in range(1, padding + 1)]
    curve_points = numpy.concatenate([previous_points, curve_points, next_points])
    tangents = numpy.concatenate([[tangents[0]] * len(previous_points), tangents, [tangents[-1]] * len(next_points)])

    normals = numpy.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    normals = normals / numpy.linalg.norm(normals, axis=1, keepdims=True)

    d1 = curve(uniform_u, 1).T
    d2 = curve(uniform_u, 2).T
    kappa = numpy.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / numpy.linalg.norm(d1, axis=1) ** 3
    radius = 1 / kappa

    return curve_points, normals, radius

def sample_panoramic(image_volume, segmentation_volume, tooth_list):
    curve_points, normals, radius = extract_curve(tooth_list, segmentation_volume.shape[0])

    min_z = min(tooth['slices'][2].start for tooth in tooth_list)
    max_z = max(tooth['slices'][2].stop for tooth in tooth_list)
    center_z = (min_z + max_z) / 2
    z = numpy.array([0, 0, 1])
    vertical = max_z - min_z

    widths = numpy.array([tooth['slices'][0].stop - tooth['slices'][0].start for tooth in tooth_list])
    heights = numpy.array([tooth['slices'][1].stop - tooth['slices'][1].start for tooth in tooth_list])
    horizontal = numpy.concatenate([widths, heights]).max()
    horizontal = min(horizontal, int(2 * radius.min() * 0.9))

    u = numpy.linspace(-horizontal / 2, horizontal / 2, horizontal)
    v = numpy.linspace(vertical / 2, -vertical / 2, vertical)
    uu, vv = numpy.meshgrid(u, v, indexing='ij')

    image_panoramic = []
    segmentation_panoramic = []
    for point, normal in zip(curve_points, normals):
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
    upper_tooth = [tooth for tooth in upper_tooth if tooth['label'] < Label.TRUNCATED]
    lower_tooth = [tooth for tooth in lower_tooth if tooth['label'] < Label.TRUNCATED]

    for tooth_list in (upper_tooth, lower_tooth):
        if not tooth_list:
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

    labels = numpy.unique(segmentation_panoramic)
    labels = labels[labels > 0]
    for label in labels:
        area = segmentation_panoramic == label
        area = ndimage.binary_fill_holes(area)

        color = GLASBEY_PALETTE[label]

        points = numpy.argwhere(area)
        center = points.mean(0)[::-1]
        (width, height), baseline = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        x = int(center[0] - width / 2)
        y = int(center[1] + (height - baseline) / 2)
        cv2.putText(image_panoramic, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 10, cv2.LINE_AA)
        cv2.putText(image_panoramic, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    image_panoramic = cv2.cvtColor(image_panoramic, cv2.COLOR_RGB2BGR)

    return image_panoramic

def IoU(data_a, data_b):
    a_center_x, a_center_y = data_a['center']
    a_width, a_height = data_a['box_size']
    a_area = a_width * a_height
    a_left = a_center_x - a_width / 2
    a_right = a_center_x + a_width / 2
    a_top = a_center_y - a_height / 2
    a_bottom = a_center_y + a_height / 2

    b_center_x, b_center_y = data_b['center']
    b_width, b_height = data_b['box_size']
    b_area = b_width * b_height
    b_left = b_center_x - b_width / 2
    b_right = b_center_x + b_width / 2
    b_top = b_center_y - b_height / 2
    b_bottom = b_center_y + b_height / 2

    left = max(a_left, b_left)
    right = min(a_right, b_right)
    top = max(a_top, b_top)
    bottom = min(a_bottom, b_bottom)

    intersection = max(0, right - left) * max(0, bottom - top)
    union = a_area + b_area - intersection
    return intersection / union

def merge_box(tooth_data, threshold=0.9):
    box_count = len(tooth_data)
    overlaps = {
        index: []
        for index in range(box_count)
    }
    for i in range(box_count - 1):
        for j in range(i + 1, box_count):
            iou = IoU(tooth_data[i], tooth_data[j])
            if iou > threshold:
                overlaps[i].append(j)
                overlaps[j].append(i)

    visited = [False] * box_count
    merged_data = []
    for index in range(box_count):
        if visited[index]:
            continue

        group = []
        queue = [index]
        visited[index] = True
        while queue:
            current = queue.pop(0)
            group.append(current)
            for neighbor in overlaps[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        best_index = max(group, key=lambda index: tooth_data[index]['bone_loss'])
        data = copy.deepcopy(tooth_data[best_index])
        for index in group:
            data['FDI'].extend(tooth_data[index]['FDI'])
        data['FDI'] = sorted(set(data['FDI']))
        merged_data.append(data)

    return merged_data

def find_panoramic_paths(dataset, patient):
    xray_folder = Path('datasets') / dataset.replace('bone_tooth_mask', 'xray') / patient
    if not xray_folder.is_dir():
        return None

    for folder in xray_folder.iterdir():
        if not folder.name.endswith('Pano'):
            continue

        files = list(folder.iterdir())
        if not files:
            return None

        image_path = files[0]
        prediction_path = Path('pano_predict') / f'{image_path.stem}.txt'
        if not prediction_path.exists():
            return None

        return image_path, prediction_path

    return None

def load_panoramic(image_path, prediction_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    with open(prediction_path) as file:
        data = file.read().strip()

    tooth_data = []
    for line in data.split('\n'):
        fdi, center_x, center_y, width, height, left_crone_x, left_crone_y, left_bone_x, left_bone_y, left_root_x, left_root_y, right_crone_x, right_crone_y, right_bone_x, right_bone_y, right_root_x, right_root_y, bone_loss = line.split(',')
        tooth_data.append({
            'FDI': [fdi],
            'center': (float(center_x), float(center_y)),
            'box_size': (int(width), int(height)),
            'left_crone': (float(left_crone_x), float(left_crone_y)),
            'left_bone': (float(left_bone_x), float(left_bone_y)),
            'left_root': (float(left_root_x), float(left_root_y)),
            'right_crone': (float(right_crone_x), float(right_crone_y)),
            'right_bone': (float(right_bone_x), float(right_bone_y)),
            'right_root': (float(right_root_x), float(right_root_y)),
            'bone_loss': int(bone_loss)
        })
    tooth_data = merge_box(tooth_data)

    return image, tooth_data

def draw_panoramic(image, tooth_data):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for tooth in tooth_data:
        center_x, center_y = tooth['center']
        (width, height), baseline = cv2.getTextSize(tooth['FDI'][0], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        x = int(center_x - width / 2)
        y = int(center_y + (height - baseline) / 2)
        color = GLASBEY_PALETTE[int(tooth['FDI'][0])]
        for fdi in tooth['FDI']:
            cv2.putText(image, fdi, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 10, cv2.LINE_AA)
            cv2.putText(image, fdi, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            y += height + 15
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image
