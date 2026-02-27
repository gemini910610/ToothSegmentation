import numpy

from scripts.tools.widgets import Label
from scipy import ndimage

def rotation_matrix(a, b):
    a = a / numpy.linalg.norm(a)
    b = b / numpy.linalg.norm(b)
    v = numpy.cross(a, b)
    c = numpy.dot(a, b)
    s = numpy.linalg.norm(v)
    matrix = numpy.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=numpy.float32)
    matrix = numpy.eye(3, dtype=numpy.float32) + matrix + matrix @ matrix * ((1 - c) / (s ** 2))
    matrix = matrix.T
    return matrix

def crop_single_tooth(segmentation_volume, image_volume, tooth_label, bone_label, padding=25):
    bone_volume = segmentation_volume == bone_label
    tooth_volume = segmentation_volume == tooth_label

    slices = ndimage.find_objects(tooth_volume)[0]
    tooth_roi = tooth_volume[slices]
    distance = ndimage.distance_transform_edt(tooth_roi).astype(numpy.float32, copy=False)
    center = numpy.array(numpy.unravel_index(distance.argmax(), distance.shape), dtype=numpy.float32)

    x0 = slices[0].start
    y0 = slices[1].start
    z0 = slices[2].start
    x1 = slices[0].stop
    y1 = slices[1].stop
    z1 = slices[2].stop

    length = max(x1 - x0, y1 - y0, z1 - z0)
    center = center + numpy.array([x0, y0, z0], dtype=numpy.float32)
    x0 = max(int(numpy.floor(center[0] - length)), 0)
    y0 = max(int(numpy.floor(center[1] - length)), 0)
    z0 = max(int(numpy.floor(center[2] - length)), 0)
    x1 = min(int(numpy.ceil(center[0] + length)), segmentation_volume.shape[0])
    y1 = min(int(numpy.ceil(center[1] + length)), segmentation_volume.shape[1])
    z1 = min(int(numpy.ceil(center[2] + length)), segmentation_volume.shape[2])

    bone_volume = bone_volume[x0:x1, y0:y1, z0:z1]
    tooth_volume = tooth_volume[x0:x1, y0:y1, z0:z1]
    image_volume = image_volume[x0:x1, y0:y1, z0:z1]
    center = center - numpy.array([x0, y0, z0], dtype=numpy.float32)

    segmentation_volume = tooth_volume * Label.TOOTH + bone_volume * Label.BONE
    segmentation_volume = segmentation_volume.astype(numpy.int32, copy=False)

    coordinates = numpy.column_stack(numpy.nonzero(tooth_volume)).astype(numpy.float32, copy=False)
    coordinates -= coordinates.mean(axis=0)
    covariance = (coordinates.T @ coordinates) / coordinates.shape[0]
    values, vectors = numpy.linalg.eigh(covariance)

    main_axis = vectors[:, numpy.argmax(values)]
    target_axis = numpy.array([0, 0, 1], dtype=numpy.float32)

    matrix = rotation_matrix(main_axis, target_axis)

    offset = center - matrix @ center

    segmentation_volume = ndimage.affine_transform(segmentation_volume, matrix, offset=offset, order=0, prefilter=False, output=numpy.uint8)
    image_volume = ndimage.affine_transform(image_volume, matrix, offset=offset, order=3, prefilter=True, output=numpy.float32)

    tooth_volume = segmentation_volume == Label.TOOTH
    slices = ndimage.find_objects(tooth_volume)[0]
    slice_x, slice_y, slice_z = slices
    x0 = max(0, slice_x.start - padding)
    y0 = max(0, slice_y.start - padding)
    z0 = max(0, slice_z.start - padding)
    x1 = min(segmentation_volume.shape[0], slice_x.stop + padding)
    y1 = min(segmentation_volume.shape[1], slice_y.stop + padding)
    z1 = min(segmentation_volume.shape[2], slice_z.stop + padding)
    segmentation_volume = segmentation_volume[x0:x1, y0:y1, z0:z1]
    image_volume = image_volume[x0:x1, y0:y1, z0:z1]
    tooth_volume = numpy.where(segmentation_volume == Label.TOOTH, image_volume, 0)

    center = center - numpy.array([x0, y0, z0], dtype=numpy.float32)

    return segmentation_volume, image_volume, tooth_volume, center

def rotate_vector(vector, degree):
    theta = numpy.deg2rad(degree)
    cos = numpy.cos(theta)
    sin = numpy.sin(theta)
    matrix = numpy.array([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1]
    ], dtype=numpy.float32)
    vector = matrix @ vector
    vector = vector / numpy.linalg.norm(vector)
    return vector

def oblique_slice(segmentation_volume, image_volume, tooth_volume, center, vector, xx, yy):
    vector1 = numpy.array([0, 0, 1], dtype=numpy.float32)
    vector2 = numpy.cross(vector, vector1)
    vector2 = vector2 / numpy.linalg.norm(vector2)

    x = center[0] + vector1[0] * xx + vector2[0] * yy
    y = center[1] + vector1[1] * xx + vector2[1] * yy
    z = segmentation_volume.shape[2] / 2 + vector1[2] * xx + vector2[2] * yy

    x = numpy.rint(x).astype(numpy.int32)
    y = numpy.rint(y).astype(numpy.int32)
    z = numpy.rint(z).astype(numpy.int32)

    mask = (x >= 0) & (x < segmentation_volume.shape[0]) & (y >= 0) & (y < segmentation_volume.shape[1]) & (z >= 0) & (z < segmentation_volume.shape[2])
    x = x[mask]
    y = y[mask]
    z = z[mask]

    segmentation_slice = numpy.zeros(xx.shape, dtype=numpy.float32)
    segmentation_slice[mask] = segmentation_volume[x, y, z]
    segmentation_slice = numpy.rot90(segmentation_slice)

    image_slice = numpy.zeros(xx.shape, dtype=numpy.float32)
    image_slice[mask] = image_volume[x, y, z]
    image_slice = numpy.rot90(image_slice)

    tooth_slice = numpy.zeros(xx.shape, dtype=numpy.float32)
    tooth_slice[mask] = tooth_volume[x, y, z]
    tooth_slice = numpy.rot90(tooth_slice)

    return segmentation_slice, image_slice, tooth_slice

def normalize_slice(segmentation_slice, image_slice, tooth_slice):
    segmentation_slice = segmentation_slice / 2 * 255
    segmentation_slice = segmentation_slice.astype(numpy.uint8)

    image_slice = image_slice.clip(0, 255)
    image_slice = image_slice.astype(numpy.uint8)

    tooth_slice = tooth_slice.clip(0, 255)
    tooth_slice = tooth_slice.astype(numpy.uint8)

    return segmentation_slice, image_slice, tooth_slice

def get_coordinates(size=256):
    yy, xx = numpy.mgrid[0:size, 0:size].astype(numpy.float32)
    yy = yy - (size - 1) / 2
    xx = xx - (size - 1) / 2
    return xx, yy

def get_slices(segmentation_volume, image_volume, tooth_volume, center):
    xx, yy = get_coordinates(max(segmentation_volume.shape))
    vector = numpy.array([0, 1, 0], dtype=numpy.float32)
    for degree in numpy.linspace(0, 360, 12, endpoint=False, dtype=numpy.int32):
        vec = rotate_vector(vector, degree)
        slices = oblique_slice(segmentation_volume, image_volume, tooth_volume, center, vec, xx, yy)
        yield slices

if __name__ == '__main__':
    import os

    from argparse import ArgumentParser
    from src.config import load_config
    from src.dataset import get_fold
    from src.console import track
    from PIL import Image
    from scripts.tools.widgets import Mode

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    config.split_file_path = os.path.join('logs', experiment_name, f'{config.split_filename}.json')

    for fold in range(1, config.num_folds + 1):
        _, valid_dataset_patients = get_fold(config.split_file_path, fold)
        for dataset, patients in valid_dataset_patients.items():
            for patient in track(patients, desc=f'Fold {fold} {dataset}'):
                segmentation_volume_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.RELABELED}.npy')
                segmentation_volume = numpy.load(segmentation_volume_path)
                image_volume_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.IMAGE}.npy')
                image_volume = numpy.load(image_volume_path)
                for label in range(2, segmentation_volume.max() + 1):
                    output_dir = os.path.join('output_slices', experiment_name, f'Fold_{fold}', dataset, patient, 'tooth_slice', str(label - 1))
                    os.makedirs(output_dir, exist_ok=True)
                    crop_segmentation_volume, crop_image_volume, crop_tooth_volume, center = crop_single_tooth(segmentation_volume, image_volume, tooth_label=label, bone_label=1)
                    slices = get_slices(crop_segmentation_volume, crop_image_volume, crop_tooth_volume, center)
                    for degree, (segmentation_slice, image_slice, tooth_slice) in enumerate(slices):
                        segmentation_slice, image_slice, tooth_slice = normalize_slice(segmentation_slice, image_slice, tooth_slice)
                        segmentation_slice = Image.fromarray(segmentation_slice)
                        segmentation_slice.save(os.path.join(output_dir, f'mask_{degree * 30}.png'))
                        image_slice = Image.fromarray(image_slice)
                        image_slice.save(os.path.join(output_dir, f'image_{degree * 30}.png'))
                        tooth_slice = Image.fromarray(tooth_slice)
                        tooth_slice.save(os.path.join(output_dir, f'tooth_{degree * 30}.png'))
