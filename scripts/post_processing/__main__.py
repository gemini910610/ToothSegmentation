import os
import numpy
import json

from argparse import ArgumentParser

from scripts.post_processing.connected_component import filter_connected_component, remove_outlier
from scripts.post_processing.fill_holes import fill_holes
from scripts.post_processing.iterators import iterate_datasets
from scripts.post_processing.refine_merge import refine_merge
from scripts.post_processing.refine_split import refine_split
from scripts.post_processing.relabel import relabel_volume
from scripts.post_processing.remove_cropped import remove_cropped
from scripts.post_processing.remove_tooth import remove_tooth
from scripts.post_processing.watershed import split_component
from scripts.tools.widgets import Label, Mode

parser = ArgumentParser()
parser.add_argument('exp', type=str)
parser.add_argument('--threshold', type=int, default=7500)
args = parser.parse_args()

experiment_name = args.exp
threshold = args.threshold

with open(os.path.join('split', f'{experiment_name}.json')) as file:
    split_tasks = json.load(file)

with open(os.path.join('merge', f'{experiment_name}.json')) as file:
    merge_tasks = json.load(file)

with open(os.path.join('remove', f'{experiment_name}.json')) as file:
    labels = json.load(file)

datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
for dataset, patient in iterate_datasets(experiment_name, datasets):
    predict_path = os.path.join('outputs', experiment_name, dataset, patient, f'{Mode.PREDICT}.npy')
    volume = numpy.load(predict_path)

    bone_volume = volume == Label.BONE
    bone_volume = filter_connected_component(bone_volume, target=Label.BONE, voxel_threshold=threshold)

    tooth_volume = volume == Label.TOOTH
    tooth_cc_volume = filter_connected_component(tooth_volume, target=Label.TOOTH, voxel_threshold=threshold)
    tooth_cc_volume = remove_outlier(tooth_cc_volume, voxel_threshold=threshold)
    tooth_watershed_volume = split_component(tooth_cc_volume)
    data = f'{dataset}/{patient}'
    tooth_volume = tooth_watershed_volume if data not in split_tasks else refine_split(tooth_cc_volume, tooth_watershed_volume, split_tasks[data])
    tooth_volume = tooth_volume if data not in merge_tasks else refine_merge(tooth_volume, merge_tasks[data])
    tooth_volume = remove_cropped(tooth_volume, voxel_threshold=threshold)
    tooth_volume = tooth_volume if data not in labels else remove_tooth(tooth_volume, labels[data])
    tooth_volume, bone_volume = fill_holes(tooth_volume, bone_volume)
    tooth_volume = relabel_volume(tooth_volume, bone_volume)

    volume = bone_volume + tooth_volume

    pp_path = os.path.join('outputs', experiment_name, dataset, patient, f'{Mode.POST_PROCESSING}.npy')
    numpy.save(pp_path, volume)
