import os
import numpy
import json

from .connected_component import filter_connected_component
from .watershed import split_component
from .refine_component import refine_component
from src.config import load_config
from src.dataset import get_fold
from src.console import track
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('exp', type=str)
parser.add_argument('--tooth-threshold', type=int, default=7500)
parser.add_argument('--bone-threshold', type=int, default=3500)
args = parser.parse_args()

experiment_name = args.exp
tooth_threshold = args.tooth_threshold
bone_threshold = args.bone_threshold

with open(os.path.join('refine', f'{experiment_name}.json')) as file:
    tasks = json.load(file)

config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
config.split_file_path = os.path.join('logs', experiment_name, f'{config.split_filename}.json')

for fold in range(1, config.num_folds + 1):
    _, valid_dataset_patients = get_fold(config.split_file_path, fold)
    for dataset, patients in valid_dataset_patients.items():
        for patient in track(patients, desc=f'Fold {fold} {dataset}'):
            predict_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, 'volume.npy')
            volume = numpy.load(predict_path)

            bone_volume = volume == 2
            bone_volume = filter_connected_component(bone_volume, bone_threshold, binary=True)

            tooth_volume = volume == 1
            tooth_cc_volume = filter_connected_component(tooth_volume, tooth_threshold)
            tooth_watershed_volume = split_component(tooth_cc_volume)
            data = f'{dataset}/{patient}'
            tooth_volume = tooth_watershed_volume if data not in tasks else refine_component(tooth_cc_volume, tooth_watershed_volume, tasks[data])
            tooth_volume = numpy.where(tooth_volume > 0, tooth_volume + 1, 0)

            volume = bone_volume + tooth_volume

            pp_path = os.path.join('outputs', experiment_name, f'Fold_{fold}', dataset, patient, 'pp_volume.npy')
            numpy.save(pp_path, volume)
