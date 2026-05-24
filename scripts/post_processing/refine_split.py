import numpy

from scripts.post_processing.watershed import split_k_component

def refine_split(source_volume, destination_volume, tasks):
    new_volume = destination_volume.copy()

    for label, cluster in tasks.items():
        label = int(label)
        mask = source_volume == label
        volume = split_k_component(mask, h=1, k=cluster, crop=True)

        new_volume[mask] = 0
        max_label = new_volume.max()
        position = volume > 0
        new_volume[position] = volume[position] + max_label

    return new_volume

if __name__ == '__main__':
    import json
    import os

    from argparse import ArgumentParser

    from scripts.post_processing.iterators import iterate_datasets
    from scripts.tools.widgets import Mode

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    with open(os.path.join('split', f'{experiment_name}.json')) as file:
        tasks = json.load(file)

    datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
    for dataset, patient in iterate_datasets(experiment_name, datasets):
        cc_path = os.path.join('outputs', experiment_name, dataset, patient, f'{Mode.TOOTH_CONNECTED_COMPONENT}.npy')
        cc_volume = numpy.load(cc_path)
        watershed_path = os.path.join('outputs', experiment_name, dataset, patient, f'{Mode.WATERSHED}.npy')
        watershed_volume = numpy.load(watershed_path)
        volume = watershed_volume if f'{dataset}/{patient}' not in tasks else refine_split(cc_volume, watershed_volume, tasks[f'{dataset}/{patient}'])

        refine_split_path = os.path.join('outputs', experiment_name, dataset, patient, f'{Mode.REFINE_SPLIT}.npy')
        numpy.save(refine_split_path, volume)
