import numpy

from scripts.tools.widgets import Label

def relabel_fdi(volume, fdi_mapping):
    tooth_count = volume.max()
    lookup_table = numpy.arange(tooth_count + 1, dtype=numpy.uint8)
    lookup_table[1] = 1
    missing_label = Label.MISSING_FDI
    labels = numpy.unique(volume)
    labels = labels[(labels > 1) & (labels < Label.CROPPED)]
    for label in labels:
        if str(label) not in fdi_mapping:
            lookup_table[label] = missing_label
            missing_label += 1
        else:
            lookup_table[label] = int(fdi_mapping[str(label)])

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

    with open(os.path.join('fdi', f'{experiment_name}.json')) as file:
        fdi_mapping = json.load(file)

    datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
    for dataset, patient in iterate_datasets(experiment_name, datasets):
        if f'{dataset}/{patient}' not in fdi_mapping:
            continue

        output_dir = os.path.join('outputs', experiment_name, dataset, patient)
        volume_path = os.path.join(output_dir, f'{Mode.POST_PROCESSING}.npy')
        volume = numpy.load(volume_path)

        volume = relabel_fdi(volume, fdi_mapping[f'{dataset}/{patient}'])

        fdi_path = os.path.join('outputs', experiment_name, dataset, patient, f'{Mode.FDI}.npy')
        numpy.save(fdi_path, volume)
