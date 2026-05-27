import numpy

from scripts.tools.widgets import Label

def remove_tooth(volume, data):
    remove_types = {
        'Unerupted Tooth': Label.UNERUPTED,
        'Residual Root': Label.RESIDUAL,
        'Fake Tooth': Label.FAKE
    }

    lookup_table = numpy.arange(volume.max() + 1, dtype=numpy.uint8)
    for remove_type, labels in data.items():
        start_label = remove_types[remove_type]
        lookup_table[labels] = numpy.arange(start_label, start_label + len(labels), dtype=numpy.uint8)
    volume = lookup_table[volume]
    return volume

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

    with open(os.path.join('remove', f'{experiment_name}.json')) as file:
        labels = json.load(file)

    datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
    for dataset, patient in iterate_datasets(experiment_name, datasets):
        volume_path = os.path.join('outputs', experiment_name, dataset, patient, f'{Mode.CLEANED}.npy')
        volume = numpy.load(volume_path)
        data = f'{dataset}/{patient}'
        if data in labels:
            volume = remove_tooth(volume, labels[data])

        removed_path = os.path.join('outputs', experiment_name, dataset, patient, f'{Mode.REMOVED}.npy')
        numpy.save(removed_path, volume)
