if __name__ == '__main__':
    import numpy
    import os

    from argparse import ArgumentParser
    from collections import defaultdict

    from keypoint.predict import CEJFinder
    from scripts.post_processing.bone_loss import volume_bone_loss
    from scripts.post_processing.iterators import iterate_datasets
    from scripts.tools.widgets import Mode
    from src.console import Table

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    cej_finder = CEJFinder('logs/keypoint_baseline/best.pth')

    datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
    for dataset, patient in iterate_datasets(experiment_name, datasets):
        patient_folder = os.path.join('outputs', experiment_name, dataset, patient)
        segmentation_volume_path = os.path.join(patient_folder, f'{Mode.FDI}.npy')
        image_volume_path = os.path.join(patient_folder, f'{Mode.IMAGE}.npy')
        if not os.path.exists(segmentation_volume_path):
            continue

        segmentation_volume = numpy.load(segmentation_volume_path)
        image_volume = numpy.load(image_volume_path)

        segmentation_bone_losses = volume_bone_loss(segmentation_volume, image_volume, cej_finder)

        xray_folder = os.path.join('datasets', dataset.replace('bone_tooth_mask', 'xray'), patient)
        for folder in os.listdir(xray_folder):
            if folder.endswith('Pano'):
                folder = os.path.join(xray_folder, folder)
                filename = os.listdir(folder)[0][:-4]
        with open(f'pano_predict/{filename}.txt') as file:
            data = file.read().strip()

        keypoint_bone_losses = defaultdict(list)
        for line in data.split('\n'):
            fdi, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, bone_loss = line.split(',')
            keypoint_bone_losses[int(fdi)].append(bone_loss)

        print(f'{dataset}/{patient}')
        with Table(['FDI', '3D', '2D']) as table:
            for fdi in sorted(segmentation_bone_losses.keys()):
                if fdi not in keypoint_bone_losses:
                    continue
                segmentation_bone_loss = segmentation_bone_losses[fdi]
                keypoint_bone_loss = ', '.join(keypoint_bone_losses[fdi])

                table.add_row([fdi, segmentation_bone_loss, keypoint_bone_loss])
