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
from src.config import load_config
from src.dataset import get_fold
from src.console import track
from argparse import ArgumentParser
from scripts.tools.widgets import Mode, Label

parser = ArgumentParser()
parser.add_argument('exp', type=str)
parser.add_argument('--tooth-threshold', type=int, default=7500)
args = parser.parse_args()

experiment_name = args.exp
tooth_threshold = args.tooth_threshold

with open(os.path.join('refine', f'{experiment_name}.json')) as file:
    tasks = json.load(file)

with open(os.path.join('remove', f'{experiment_name}.json')) as file:
    labels = json.load(file)

config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
config.split_file_path = os.path.join('logs', experiment_name, f'{config.split_filename}.json')

for fold in range(1, config.num_folds + 1):
    _, valid_dataset_patients = get_fold(config.split_file_path, fold)
    for dataset, patients in valid_dataset_patients.items():
        for patient in track(patients, desc=f'Fold {fold} {dataset}'):
            predict_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.PREDICT}.npy')
            volume = numpy.load(predict_path)

            bone_volume = volume == Label.BONE
            bone_volume = filter_connected_component(bone_volume, target=Label.BONE)

            tooth_volume = volume == Label.TOOTH
            tooth_cc_volume = filter_connected_component(tooth_volume, target=Label.TOOTH, voxel_threshold=tooth_threshold)
            tooth_cc_volume = remove_outlier(tooth_cc_volume, voxel_threshold=tooth_threshold)
            tooth_watershed_volume = split_component(tooth_cc_volume)
            data = f'{dataset}/{patient}'
            tooth_volume = tooth_watershed_volume if data not in tasks else refine_component(tooth_cc_volume, tooth_watershed_volume, tasks[data])
            tooth_volume = remove_cropped(tooth_volume)
            tooth_volume = tooth_volume if data not in labels else remove_tooth(tooth_volume, labels[data])
            tooth_volume, bone_volume = fill_holes(tooth_volume, bone_volume)
            tooth_volume = relabel_volume(tooth_volume, bone_volume)

            volume = bone_volume + tooth_volume

            pp_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, f'{Mode.POST_PROCESSING}.npy')
            numpy.save(pp_path, volume)
