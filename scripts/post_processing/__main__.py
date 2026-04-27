import os
import numpy
import json

from .connected_component import filter_connected_component, remove_outlier
from .watershed import split_component
from .refine_component import refine_component
from .remove_cropped import remove_cropped
from .remove_tooth import remove_tooth
from .fill_holes import fill_holes
from .relabel import relabel_volume
from .iterators import iterate_fold_patients
from src.config import load_config
from argparse import ArgumentParser
from scripts.tools.widgets import Mode, Label

parser = ArgumentParser()
parser.add_argument('exp', type=str)
parser.add_argument('--threshold', type=int, default=7500)
args = parser.parse_args()

experiment_name = args.exp
threshold = args.threshold

with open(os.path.join('refine', f'{experiment_name}.json')) as file:
    tasks = json.load(file)

with open(os.path.join('remove', f'{experiment_name}.json')) as file:
    labels = json.load(file)

config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
config.split_file_path = os.path.join('logs', experiment_name, f'{config.split_filename}.json')

for fold, dataset, patient in iterate_fold_patients(config):
    predict_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.PREDICT}.npy')
    volume = numpy.load(predict_path)

    bone_volume = volume == Label.BONE
    bone_volume = filter_connected_component(bone_volume, target=Label.BONE, voxel_threshold=threshold)

    tooth_volume = volume == Label.TOOTH
    tooth_cc_volume = filter_connected_component(tooth_volume, target=Label.TOOTH, voxel_threshold=threshold)
    tooth_cc_volume = remove_outlier(tooth_cc_volume, voxel_threshold=threshold)
    tooth_watershed_volume = split_component(tooth_cc_volume)
    data = f'{dataset}/{patient}'
    tooth_volume = tooth_watershed_volume if data not in tasks else refine_component(tooth_cc_volume, tooth_watershed_volume, tasks[data])
    tooth_volume = remove_cropped(tooth_volume, voxel_threshold=threshold)
    tooth_volume = tooth_volume if data not in labels else remove_tooth(tooth_volume, labels[data])
    tooth_volume, bone_volume = fill_holes(tooth_volume, bone_volume)
    tooth_volume = relabel_volume(tooth_volume, bone_volume)

    volume = bone_volume + tooth_volume

    pp_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.POST_PROCESSING}.npy')
    numpy.save(pp_path, volume)
