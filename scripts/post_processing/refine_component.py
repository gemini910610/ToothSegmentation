import numpy

from scripts.post_processing.watershed import split_k_component

def refine_component(source_volume, destination_volume, tasks):
    new_volume = destination_volume.copy()

    for label, cluster in tasks:
        mask = source_volume == label
        volume = split_k_component(mask, h=1, k=cluster, crop=True)

        new_volume[mask] = 0
        max_label = new_volume.max()
        position = volume > 0
        new_volume[position] = volume[position] + max_label

    labels = numpy.unique(new_volume)
    labels = labels[labels != 0]
    lookup_table = numpy.zeros(labels.max() + 1, dtype=numpy.int32)
    lookup_table[labels] = numpy.arange(1, len(labels) + 1, dtype=numpy.int32)

    new_volume = lookup_table[new_volume]

    return new_volume

if __name__ == '__main__':
    import os
    import json

    from argparse import ArgumentParser
    from src.config import load_config
    from scripts.tools.visualize import get_patient_fold_mapping, DataManager, Mode
    from src.console import track

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    with open(os.path.join('refine', f'{experiment_name}.json')) as file:
        tasks = json.load(file)

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    patient_fold_map = get_patient_fold_mapping(config)

    data_manager = DataManager(experiment_name, patient_fold_map, [Mode.CONNECTED_COMPONENT, Mode.WATERSHED], [1])

    for data, task in track(tasks.items()):
        cc_volume, watershed_volume = data_manager.load_data(data)
        new_volume = refine_component(cc_volume, watershed_volume, task)
        data_manager.save_data(data, new_volume, 'refine_volume.npy')
