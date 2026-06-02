from openpyxl import Workbook
from openpyxl.styles import Alignment

class ExcelWriter:
    def __init__(self, path):
        self.path = path
        self.workbook = Workbook()
        self.sheet = self.workbook.active

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._init_header()

    def _init_header(self):
        center_bottom = Alignment(horizontal='center', vertical='bottom')
        center = Alignment(horizontal='center', vertical='center')

        for column, value in enumerate(['Dataset', 'Patient', 'FDI', '3D', '2D'], 1):
            cell = self.sheet.cell(1, column, value)
            cell.alignment = center_bottom
            self.sheet.merge_cells(start_row=1, start_column=column, end_row=2, end_column=column)

        for offset, value in enumerate([*[f'{degree}°' for degree in range(0, 360, 45)], 'Mean', 'Max']):
            column = 6 + offset * 2
            cell = self.sheet.cell(1, column, value)
            cell.alignment = center
            self.sheet.merge_cells(start_row=1, start_column=column, end_row=1, end_column=column + 1)
            for column, value in enumerate(['Curve', 'Area'], column):
                cell = self.sheet.cell(2, column, value)
                cell.alignment = center

        for column, value in enumerate(['3D', 'Curve 0°', 'Area 0°', 'Mean Curve', 'Mean Area', 'Max Curve', 'Max Area'], 26):
            cell = self.sheet.cell(1, column, f'2D vs {value}')
            cell.alignment = center_bottom
            self.sheet.merge_cells(start_row=1, start_column=column, end_row=2, end_column=column)

        self.sheet.freeze_panes = 'C3'

    def write(self, dataset, patient, data):
        start_row = self.sheet.max_row + 1
        for row, values in enumerate(data, start_row):
            for column, value in enumerate(values, 3):
                self.sheet.cell(row, column, value)
        end_row = self.sheet.max_row

        center = Alignment(horizontal='center', vertical='center')
        for column, value in enumerate([dataset, patient], 1):
            cell = self.sheet.cell(start_row, column, value)
            cell.alignment = center
            self.sheet.merge_cells(start_row=start_row, start_column=column, end_row=end_row, end_column=column)

    def save(self):
        self.workbook.save(self.path)

def get_symbol(bone_loss_1, bone_loss_2, rate=2):
    if bone_loss_1 == 0 and bone_loss_2 == 0:
        return None
    if bone_loss_1 == 0 or bone_loss_2 == 0:
        return '▲'
    if max(bone_loss_1, bone_loss_2) / min(bone_loss_1, bone_loss_2) > rate:
        return '▲'
    return None

if __name__ == '__main__':
    import numpy
    import os

    from argparse import ArgumentParser
    from collections import defaultdict
    from matplotlib import pyplot

    from keypoint.predict import CEJFinder
    from scripts.post_processing.bone_loss import slice_area_bone_loss, slice_bone_loss, volume_bone_loss
    from scripts.post_processing.tooth_slice import align_crop_tooth, extract_single_tooth, find_normal_vectors, get_slices, normalize_slice
    from scripts.post_processing.iterators import iterate_datasets
    from scripts.tools.widgets import Label, Mode
    from src.console import Table, track

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    cej_finder = CEJFinder('logs/keypoint_baseline/best.pth')

    writer = ExcelWriter(os.path.join('bone_loss', experiment_name, 'bone_loss.xlsx'))

    all_bone_losses = defaultdict(list)

    datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
    for dataset, patient in iterate_datasets(experiment_name, datasets):
        patient_folder = os.path.join('outputs', experiment_name, dataset, patient)
        segmentation_volume_path = os.path.join(patient_folder, f'{Mode.FDI}.npy')
        segmentation_volume = numpy.load(segmentation_volume_path)
        image_volume_path = os.path.join(patient_folder, f'{Mode.IMAGE}.npy')
        image_volume = numpy.load(image_volume_path)

        normal_vectors = find_normal_vectors(segmentation_volume)

        labels = numpy.unique(segmentation_volume)
        labels = labels[(labels > 1) & (labels < Label.CROPPED)] # exclude background, bone, removed teeth and missing FDI teeth

        bone_losses = defaultdict(list)
        area_bone_losses = defaultdict(list)
        segmentation_bone_losses = {}
        for label in track(labels, desc=patient):
            normal_vector = normal_vectors[label]
            filtered_volume, transform_meta = extract_single_tooth(segmentation_volume, normal_vector, tooth_label=label, bone_label=1)
            aligned_volume, aligned_image = align_crop_tooth(filtered_volume, image_volume, transform_meta)

            slices = get_slices(aligned_volume, aligned_image, reverse=label // 10 in {2, 3})

            segmentation_slices = []
            tooth_slices = []
            for segmentation_slice, image_slice, tooth_slice in slices:
                _, _, tooth_slice = normalize_slice(segmentation_slice, image_slice, tooth_slice)

                segmentation_slices.append(segmentation_slice)
                tooth_slices.append(tooth_slice)

            points = cej_finder.find(segmentation_slices, tooth_slices)

            for segmentation_slice, point in zip(segmentation_slices, points):
                bone_loss = slice_bone_loss(segmentation_slice, point, label)
                area_bone_loss, _ = slice_area_bone_loss(segmentation_slice, point, label)

                bone_losses[label].append(bone_loss)
                area_bone_losses[label].append(area_bone_loss)

            segmentation_bone_loss = volume_bone_loss(filtered_volume, aligned_volume, points, label, transform_meta)
            segmentation_bone_losses[label] = segmentation_bone_loss

        xray_folder = os.path.join('datasets', dataset.replace('bone_tooth_mask', 'xray'), patient)
        for folder in os.listdir(xray_folder):
            if folder.endswith('Pano'):
                folder = os.path.join(xray_folder, folder)
                filename = os.listdir(folder)[0][:-4]
        with open(os.path.join('pano_predict', f'{filename}.txt')) as file:
            data = file.read().strip()

        keypoint_bone_losses = {}
        for line in data.split('\n'):
            values = line.split(',')
            fdi = int(values[0])
            bone_loss = int(values[-1])
            keypoint_bone_losses[fdi] = bone_loss

        data = [
            [
                label,
                segmentation_bone_losses[label],
                keypoint_bone_losses[label],
                *[
                    losses[label][index]
                    for index in range(8)
                    for losses in (bone_losses, area_bone_losses)
                ],
                sum(bone_losses[label]) / len(bone_losses[label]),
                sum(area_bone_losses[label]) / len(area_bone_losses[label]),
                max(bone_losses[label]),
                max(area_bone_losses[label]),
                *[
                    get_symbol(keypoint_bone_losses[label], loss)
                    for loss in [
                        segmentation_bone_losses[label],
                        bone_losses[label][0],
                        area_bone_losses[label][0],
                        sum(bone_losses[label]) / len(bone_losses[label]),
                        sum(area_bone_losses[label]) / len(area_bone_losses[label]),
                        max(bone_losses[label]),
                        max(area_bone_losses[label])
                    ]
                ]
            ]
            for label in labels
        ]
        writer.write(dataset, patient, data)

        all_bone_losses['3D'].extend([segmentation_bone_losses[label] for label in labels])
        all_bone_losses['2D'].extend([keypoint_bone_losses[label] for label in labels])
        all_bone_losses['Curve 0°'].extend([bone_losses[label][0] for label in labels])
        all_bone_losses['Area 0°'].extend([area_bone_losses[label][0] for label in labels])
        all_bone_losses['Mean Curve'].extend([sum(bone_losses[label]) / len(bone_losses[label]) for label in labels])
        all_bone_losses['Mean Area'].extend([sum(area_bone_losses[label]) / len(area_bone_losses[label]) for label in labels])
        all_bone_losses['Max Curve'].extend([max(bone_losses[label]) for label in labels])
        all_bone_losses['Max Area'].extend([max(area_bone_losses[label]) for label in labels])

    writer.save()

    all_bone_losses = {
        key: numpy.array(value)
        for key, value in all_bone_losses.items()
    }
    errors = {
        key: all_bone_losses[key] - all_bone_losses['2D']
        for key in all_bone_losses
        if key != '2D'
    }

    with Table() as table:
        for key in errors:
            table.add_row([f'{key} - 2D', f'{errors[key].mean():.6f} ± {errors[key].std():.6f}'])

    for key in errors:
        pyplot.hist(errors[key], bins=100)
        pyplot.title(f'{key} - 2D')
        pyplot.tight_layout()
        pyplot.savefig(f'bone_loss/{experiment_name}/{key}-2D.png')
        pyplot.close()
