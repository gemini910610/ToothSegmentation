from collections import defaultdict
from pathlib import Path

import numpy
from scipy import ndimage

from constants.label import Label
from constants.mode import Mode
from keypoints.models import load_model
from keypoints.predict import CEJFinder
from post_processing.bone_loss import (
    get_bone_surface,
    get_tooth_surface,
    restore_coordinates,
    slice_area_bone_loss,
    slice_contour_bone_loss,
    volume_bone_loss,
)
from post_processing.tooth_slice import (
    align_crop_tooth,
    extract_single_tooth,
    find_normal_vectors,
    get_slices,
    normalize_slice,
)
from post_processing.utils import iterate_datasets
from tui.apps import DualExperimentFoldApp, parse_args
from utils.config import load_config
from utils.console import track
from utils.excel import ExcelWriter


def configure_parser(parser):
    parser.add_argument('segmentation_exp', type=str)
    parser.add_argument('keypoints_exp', type=str)
    parser.add_argument('datasets', nargs='+', type=str)
    parser.add_argument('--keypoints-fold', type=int, default=1)

args = parse_args(DualExperimentFoldApp, configure_parser)

segmentation_experiment_name = args['segmentation_exp']
keypoints_experiment_name = args['keypoints_exp']
datasets = args['datasets']
keypoints_fold = args['keypoints_fold']

config = load_config(Path('logs') / keypoints_experiment_name / 'keypoints.toml', fold=keypoints_fold)
keypoints_model = load_model(config)
cej_finder = CEJFinder(keypoints_model, config.device)

writer = ExcelWriter(Path('bone_loss') / f'{segmentation_experiment_name}.xlsx')

patients = [
    (dataset, patient)
    for dataset, patient in iterate_datasets(segmentation_experiment_name, datasets)
    if (Path('outputs') / segmentation_experiment_name / dataset / patient / f'{Mode.FDI}.npy').is_file()
]

for dataset, patient in track(patients):
    patient_dir = Path('outputs') / segmentation_experiment_name / dataset / patient
    fdi_volume = numpy.load(patient_dir / f'{Mode.FDI}.npy')
    bone_volume = numpy.load(patient_dir / f'{Mode.BONE_FILLED}.npy')
    image_volume = numpy.load(patient_dir / f'{Mode.IMAGE}.npy')

    bone_mask = bone_volume > 0
    segmentation_volume = fdi_volume + bone_mask
    bone_coordinates = numpy.argwhere(bone_mask)
    image_coefficients = ndimage.spline_filter(image_volume, order=3)

    normal_vectors = find_normal_vectors(segmentation_volume)

    labels = numpy.unique(segmentation_volume)
    labels = labels[(labels > 1) & (labels < Label.TRUNCATED)] # exclude background, bone and removed teeth

    tooth_infos = []
    patient_segmentation_slices = []
    patient_tooth_slices = []
    for label in track(labels, desc=f'{dataset}/{patient}'):
        is_upper = label // 10 in {1, 2}

        normal_vector = normal_vectors[label]
        filtered_segmentation, transform_meta = extract_single_tooth(segmentation_volume == label, bone_mask, bone_coordinates, normal_vector)
        aligned_volume, aligned_image = align_crop_tooth(filtered_segmentation, image_coefficients, transform_meta)

        slices = get_slices(aligned_volume, aligned_image, reverse=label // 10 in {2, 3})

        segmentation_slices = []
        for raw_segmentation_slice, raw_image_slice in slices:
            normalized_segmentation_slice, normalized_image_slice = normalize_slice(raw_segmentation_slice, raw_image_slice)
            tooth_slice = normalized_image_slice * (normalized_segmentation_slice == 127).astype(numpy.uint8)

            segmentation_slices.append(raw_segmentation_slice)
            patient_tooth_slices.append(tooth_slice)

        patient_segmentation_slices.extend(segmentation_slices)
        tooth_infos.append((label, is_upper, filtered_segmentation, transform_meta, aligned_volume, segmentation_slices))

    patient_points = cej_finder.find(patient_segmentation_slices, patient_tooth_slices)
    patient_points = patient_points.reshape(len(tooth_infos), -1, 4)

    contour_bone_losses = defaultdict(list)
    area_bone_losses = defaultdict(list)
    segmentation_bone_losses = {}
    for points, (label, is_upper, filtered_segmentation, transform_meta, aligned_volume, segmentation_slices) in zip(patient_points, tooth_infos):
        for segmentation_slice, point in zip(segmentation_slices, points):
            tooth_slice = segmentation_slice == Label.TOOTH
            bone_slice = segmentation_slice == Label.BONE
            tooth_surface = get_tooth_surface(tooth_slice)
            bone_surface = get_bone_surface(tooth_slice, bone_slice)

            contour_bone_loss = slice_contour_bone_loss(tooth_surface, bone_surface, point, is_upper)
            area_bone_loss, _ = slice_area_bone_loss(tooth_slice, bone_surface, point, is_upper)

            contour_bone_losses[label].append(contour_bone_loss)
            area_bone_losses[label].append(area_bone_loss)

        filtered_tooth_volume = filtered_segmentation == Label.TOOTH
        filtered_bone_volume = filtered_segmentation == Label.BONE
        _, aligned_points = restore_coordinates(points, aligned_volume.shape, transform_meta)
        segmentation_bone_loss = volume_bone_loss(filtered_tooth_volume, filtered_bone_volume, aligned_points, is_upper, transform_meta)
        segmentation_bone_losses[label] = segmentation_bone_loss

    panoramic_dir = Path('datasets') / dataset / patient / 'panoramic'
    for file in panoramic_dir.iterdir():
        break
    pano_predict_data = (Path('outputs') / 'Panoramic' / dataset / patient / f'{file.stem}.txt').read_text().strip()

    keypoint_bone_losses = {}
    for line in pano_predict_data.split('\n'):
        values = line.split(',')
        fdi = int(values[0])
        bone_loss = int(values[-1])
        keypoint_bone_losses[fdi] = bone_loss

    rows = [
        [
            label,
            segmentation_bone_losses[label],
            keypoint_bone_losses[label],
            *[
                losses[label][index]
                for index in range(8)
                for losses in (contour_bone_losses, area_bone_losses)
            ],
            sum(contour_bone_losses[label]) / len(contour_bone_losses[label]),
            sum(area_bone_losses[label]) / len(area_bone_losses[label]),
            max(contour_bone_losses[label]),
            max(area_bone_losses[label]),
        ]
        for label in labels
    ]
    writer.write(dataset, patient, rows)

writer.save()
