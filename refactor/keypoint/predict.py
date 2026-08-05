import cv2
import numpy
import torch

from torchvision import transforms

from refactor.keypoint.models import KeypointModel
from refactor.scripts.tools.widgets import Label

class CEJFinder:
    def __init__(self, model_path):
        model = KeypointModel().cuda()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        self.model = model.eval()
        self.transform = transforms.Resize((224, 224))

    def _preprocess_image(self, image):
        image = torch.from_numpy(image)
        image = image / 255
        image = image.unsqueeze(0)
        image = self.transform(image)
        return image

    def find(self, segmentation_slices, tooth_slices):
        images = torch.stack([self._preprocess_image(image) for image in tooth_slices])
        images = images.cuda()

        with torch.no_grad(), torch.autocast('cuda'):
            predicts = self.model(images)

        points = []
        for segmentation_slice, tooth_slice, (left_x, left_y, right_x, right_y) in zip(segmentation_slices, tooth_slices, predicts):
            height, width = tooth_slice.shape
            left_x = (left_x * width).item()
            left_y = (left_y * height).item()
            right_x = (right_x * width).item()
            right_y = (right_y * height).item()

            point = self._move_to_surface(segmentation_slice, left_x, left_y, right_x, right_y)
            points.append(point)
        points = numpy.array(points)
        return points

    def _move_to_surface(self, segmentation_slice, left_x, left_y, right_x, right_y):
        tooth_area = (segmentation_slice == Label.TOOTH).astype(numpy.uint8)
        contours, _ = cv2.findContours(tooth_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        points = numpy.concatenate(contours)[:, 0]
        xs, ys = points.T

        distances = (xs - left_x) ** 2 + (ys - left_y) ** 2
        index = numpy.argmin(distances)
        left_x = xs[index]
        left_y = ys[index]

        distances = (xs - right_x) ** 2 + (ys - right_y) ** 2
        index = numpy.argmin(distances)
        right_x = xs[index]
        right_y = ys[index]

        return left_x, left_y, right_x, right_y
