import numpy

def refine_merge(volume, tasks):
    max_label = volume.max()
    lookup_table = numpy.arange(max_label + 1, dtype=numpy.uint8)
    for labels in tasks:
        for label in labels:
            lookup_table[label] = labels[0]

    volume = lookup_table[volume]
    return volume

if __name__ == '__main__':
    import json
    import os

    from argparse import ArgumentParser

    from refactor.scripts.post_processing.iterators import iterate_datasets
    from refactor.scripts.tools.widgets import Mode

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    with open(os.path.join('merge', f'{experiment_name}.json')) as file:
        tasks = json.load(file)

    datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
    for dataset, patient in iterate_datasets(experiment_name, datasets):
        refine_split_path = os.path.join('outputs', experiment_name, dataset, patient, f'{Mode.REFINE_SPLIT}.npy')
        refine_split_volume = numpy.load(refine_split_path)
        volume = refine_split_volume if f'{dataset}/{patient}' not in tasks else refine_merge(refine_split_volume, tasks[f'{dataset}/{patient}'])

        refine_merge_path = os.path.join('outputs', experiment_name, dataset, patient, f'{Mode.REFINE_MERGE}.npy')
        numpy.save(refine_merge_path, volume)
