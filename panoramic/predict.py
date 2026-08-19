import pickle
import types
from pathlib import Path

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2

from panoramic.vendor.models.experimental import attempt_load
from panoramic.vendor.models.my_transpose_v2_240608_CAB import (
    get_yoshen_transpose_v2_240608,
)
from panoramic.vendor.models.my_transpose_v2_240706_CAB import (
    get_yoshen_transpose_v2_240706,
)
from panoramic.vendor.utils.datasets import LoadImages
from panoramic.vendor.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    set_logging,
    xyxy2xywh,
)


class _CheckpointUnpickler(pickle.Unpickler):
    """Redirects the checkpoint's original `models.*`/`utils.*` class paths to their vendored location."""

    def find_class(self, module, name):
        root = module.split('.', 1)[0]
        if root in ('models', 'utils'):
            module = f'panoramic.vendor.{module}'

        return super().find_class(module, name)

CHECKPOINT_PICKLE_MODULE = types.ModuleType('panoramic_checkpoint_pickle')
CHECKPOINT_PICKLE_MODULE.__dict__.update(pickle.__dict__)
CHECKPOINT_PICKLE_MODULE.Unpickler = _CheckpointUnpickler

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

DOUBLE_ROOT_FDIS = {48, 47, 46, 36, 37, 38}

CLASS_TO_FDI = {
    0: 11, 8: 21,
    1: 12, 9: 22,
    2: 13, 10: 23,
    3: 14, 11: 24,
    4: 15, 12: 25,
    5: 16, 13: 26,
    6: 17, 14: 27,
    7: 18, 15: 28,
    16: 31, 24: 41,
    17: 32, 25: 42,
    18: 33, 26: 43,
    19: 34, 27: 44,
    20: 35, 28: 45,
    21: 36, 29: 46,
    22: 37, 30: 47,
    23: 38, 31: 48,
}

KEYPOINT_TRANSFORM = A.Compose([A.Resize(height=384, width=384), ToTensorV2()])


def load_yolo_model(weights_path, device, img_size=1280):
    set_logging()

    model = attempt_load(str(weights_path), map_location=device, pickle_module=CHECKPOINT_PICKLE_MODULE)
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)

    if device.type != 'cpu':
        model.half()

    return model, stride, img_size

def detect_teeth(image_path, model, stride, img_size, device):
    detections = []
    dataset = LoadImages(str(image_path), img_size=img_size, stride=stride)
    for _, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if device.type != 'cpu' else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.5, 0.01, classes=None, agnostic=False)

        for det in pred:
            if len(det) == 0:
                continue

            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
            for *xyxy, _, cls in det:
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                detections.append((int(cls), *xywh))

    return sort_detections(detections)

def sort_detections(detections):
    class_0_15 = [detection for detection in detections if 0 <= detection[0] <= 15]
    class_16_31 = [detection for detection in detections if 16 <= detection[0] <= 31]

    class_0_15.sort(key=lambda detection: detection[1])
    class_16_31.sort(key=lambda detection: detection[1])

    return [
        (CLASS_TO_FDI[cls], x_center, y_center, width, height)
        for cls, x_center, y_center, width, height in class_0_15 + class_16_31
        if cls in CLASS_TO_FDI
    ]

def load_keypoint_model(num_class, weights_path, device):
    if num_class == 5:
        model = get_yoshen_transpose_v2_240706(nclass=num_class)
    else:
        model = get_yoshen_transpose_v2_240608(nclass=num_class)

    model = model.to(device)
    model.load_state_dict(torch.load(str(weights_path), map_location=device, weights_only=True))
    model.eval()

    return model

def predict_keypoints(model, image, device, img_size=384):
    transformed = KEYPOINT_TRANSFORM(image=image)
    inputs = transformed['image'].unsqueeze(0).float().to(device)

    with torch.no_grad():
        _, heatmap = model(inputs)

    heatmap = torch.nn.functional.interpolate(heatmap, size=(img_size, img_size), mode='bilinear', align_corners=False)

    _, num_keypoints, _, _ = heatmap.shape
    points = []
    for index in range(num_keypoints):
        keypoint_heatmap = heatmap[0, index]
        _, max_index = torch.max(keypoint_heatmap.view(-1), dim=0)
        y = (max_index // keypoint_heatmap.shape[1]).item()
        x = (max_index % keypoint_heatmap.shape[1]).item()
        points.append((x, y))

    return points

def bone_loss(crown1, crown2, bone1, bone2, root1, root2):
    mid_crown = ((crown1[0] + crown2[0]) / 2, (crown1[1] + crown2[1]) / 2)

    length1 = ((crown1[0] - bone1[0]) ** 2 + (crown1[1] - bone1[1]) ** 2) ** 0.5
    length2 = ((crown2[0] - bone2[0]) ** 2 + (crown2[1] - bone2[1]) ** 2) ** 0.5
    length3 = ((mid_crown[0] - root1[0]) ** 2 + (mid_crown[1] - root1[1]) ** 2) ** 0.5
    length4 = ((mid_crown[0] - root2[0]) ** 2 + (mid_crown[1] - root2[1]) ** 2) ** 0.5

    return round(max(length1 / length3, length2 / length4) * 100)

def predict_tooth_keypoints(image, box, single_root_model, double_root_model, device, fdi, crop_size=384):
    height, width = image.shape[:2]
    x_center, y_center, box_width, box_height = box[0] * width, box[1] * height, box[2] * width, box[3] * height

    x1 = int(max(0, x_center - 3 * box_width / 4))
    y1 = int(max(0, y_center - 3 * box_height / 4))
    x2 = int(min(width - 1, x_center + 3 * box_width / 4))
    y2 = int(min(height - 1, y_center + 3 * box_height / 4))

    crop = image[y1:y2, x1:x2]

    if fdi in DOUBLE_ROOT_FDIS:
        crown1, crown2, bone1, bone2, root1, root2 = predict_keypoints(double_root_model, crop, device, crop_size)
    else:
        crown1, crown2, bone1, bone2, root1 = predict_keypoints(single_root_model, crop, device, crop_size)
        root2 = root1

    def to_image_space(point):
        return (
            point[0] / crop_size * (x2 - x1) + x1,
            point[1] / crop_size * (y2 - y1) + y1,
        )

    crown1, bone1, root1, crown2, bone2, root2 = (to_image_space(point) for point in (crown1, bone1, root1, crown2, bone2, root2))

    return (x_center, y_center, x2 - x1, y2 - y1), crown1, bone1, root1, crown2, bone2, root2

class PanoramicModels:
    def __init__(self, yolo_weights, transpose_5p_weights, transpose_6p_weights, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.yolo_model, self.yolo_stride, self.yolo_img_size = load_yolo_model(yolo_weights, device)
        self.single_root_model = load_keypoint_model(5, transpose_5p_weights, device)
        self.double_root_model = load_keypoint_model(6, transpose_6p_weights, device)

def predict_panoramic(image_path, models, output_path):
    image = cv2.imread(str(image_path))

    boxes = detect_teeth(image_path, models.yolo_model, models.yolo_stride, models.yolo_img_size, models.device)

    lines = []
    for fdi, x_center, y_center, width, height in boxes:
        box, crown1, bone1, root1, crown2, bone2, root2 = predict_tooth_keypoints(
            image, (x_center, y_center, width, height), models.single_root_model, models.double_root_model, models.device, fdi
        )
        loss = bone_loss(crown1, crown2, bone1, bone2, root1, root2)

        x_center, y_center, width, height = box
        lines.append(
            f'{fdi},{x_center},{y_center},{width},{height},'
            f'{crown1[0]},{crown1[1]},{bone1[0]},{bone1[1]},{root1[0]},{root1[1]},'
            f'{crown2[0]},{crown2[1]},{bone2[0]},{bone2[1]},{root2[0]},{root2[1]},{loss}'
        )

    output_path = Path(output_path)
    output_path.write_text('\n'.join(lines), encoding='utf-8')

    return output_path

def find_panoramic_images(datasets):
    for dataset in datasets:
        dataset_dir = Path('datasets') / dataset
        for patient_dir in sorted(dataset_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            for pano_dir in sorted(patient_dir.glob('* Pano')):
                for image_path in sorted(pano_dir.iterdir()):
                    if image_path.suffix.lower() in IMAGE_SUFFIXES:
                        yield image_path

if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser

    from utils.console import track

    weights_dir = Path('panoramic/weights')
    output_dir = Path('pano_predict_test')

    if len(sys.argv) == 1:
        from tui.apps import PanoramicPredictApp

        args = PanoramicPredictApp().run()
        if args is None:
            sys.exit()

        datasets = args['datasets']
    else:
        parser = ArgumentParser()
        parser.add_argument('datasets', nargs='+', type=str)
        args = parser.parse_args()

        datasets = args.datasets

    output_dir.mkdir(parents=True, exist_ok=True)

    models = PanoramicModels(
        weights_dir / 'pano_best32type_nolr.pt',
        weights_dir / 'pano_transpose_v2_5p.pkl',
        weights_dir / 'pano_transpose_v2_6p.pkl',
    )

    image_paths = list(find_panoramic_images(datasets))
    for image_path in track(image_paths, desc='panoramic predict'):
        predict_panoramic(image_path, models, output_dir / f'{image_path.stem}.txt')
