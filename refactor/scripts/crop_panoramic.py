import os
import cv2

from collections import defaultdict

def load_panoramic(dataset, patient):
    xray_folder = os.path.join('datasets', dataset.replace('bone_tooth_mask', 'xray'), patient)
    for folder in os.listdir(xray_folder):
        if folder.endswith('Pano'):
            folder = os.path.join(xray_folder, folder)
            filename = os.listdir(folder)[0]
            pano = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            filename = filename[:-4]
            break
    with open(f'pano_predict/{filename}.txt') as file:
        lines = file.read().strip().split('\n')
        data = [
            {
                'FDI': int(fdi),
                'center': (round(float(center_x)), round(float(center_y))),
                'box_size': (int(width), int(height)),
                'left_crone': (round(float(left_crone_x)), round(float(left_crone_y))),
                'left_bone': (round(float(left_bone_x)), round(float(left_bone_y))),
                'left_root': (round(float(left_root_x)), round(float(left_root_y))),
                'right_crone': (round(float(right_crone_x)), round(float(right_crone_y))),
                'right_bone': (round(float(right_bone_x)), round(float(right_bone_y))),
                'right_root': (round(float(right_root_x)), round(float(right_root_y))),
                'bone_loss': int(bone_loss)
            }
            for line in lines
            for fdi, center_x, center_y, width, height, left_crone_x, left_crone_y, left_bone_x, left_bone_y, left_root_x, left_root_y, right_crone_x, right_crone_y, right_bone_x, right_bone_y, right_root_x, right_root_y, bone_loss in [line.split(',')]
        ]
    return pano, data

def crop_panoramic(pano, data):
    images = defaultdict(list)
    for tooth in data:
        image = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)

        image = cv2.circle(image, tooth['left_crone'], 2, (255, 0, 0), 2)
        image = cv2.circle(image, tooth['right_crone'], 2, (255, 0, 0), 2)
        image = cv2.circle(image, tooth['left_bone'], 2, (0, 255, 0), 2)
        image = cv2.circle(image, tooth['right_bone'], 2, (0, 255, 0), 2)
        image = cv2.circle(image, tooth['left_root'], 2, (0, 0, 255), 2)
        image = cv2.circle(image, tooth['right_root'], 2, (0, 0, 255), 2)

        left = tooth['center'][0] - tooth['box_size'][0] // 2
        top = tooth['center'][1] - tooth['box_size'][1] // 2
        right = tooth['center'][0] + tooth['box_size'][0] // 2
        bottom = tooth['center'][1] + tooth['box_size'][1] // 2
        image = image[top:bottom, left:right]

        (height, _), _ = cv2.getTextSize(str(tooth['bone_loss']), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        image = cv2.putText(image, str(tooth['bone_loss']), (10, height // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5, cv2.LINE_AA)
        image = cv2.putText(image, str(tooth['bone_loss']), (10, height // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images[tooth['FDI']].append(image)

    return images

if __name__ == '__main__':
    from argparse import ArgumentParser

    from refactor.scripts.post_processing.iterators import iterate_datasets

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    datasets = ['bone_tooth_mask', 'bone_tooth_mask_1209', 'bone_tooth_mask_0508']
    for dataset, patient in iterate_datasets(experiment_name, datasets):
        pano, data = load_panoramic(dataset, patient)
        images = crop_panoramic(pano, data)

        output_folder = os.path.join('pano_crop', experiment_name, dataset, patient)
        os.makedirs(output_folder, exist_ok=True)

        for fdi, images in images.items():
            for index, image in enumerate(images, 1):
                cv2.imwrite(os.path.join(output_folder, f'{fdi}_{index}.png'), image)
