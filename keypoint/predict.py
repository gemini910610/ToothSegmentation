import torch
import numpy

from keypoint.models import KeypointModel
from torchvision import transforms

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

    def find(self, tooth_slices):
        images = torch.stack([self._preprocess_image(image) for image in tooth_slices])
        images = images.cuda()

        with torch.no_grad(), torch.autocast('cuda'):
            predicts = self.model(images)

        points = []
        for tooth_slice, (left_x, left_y, right_x, right_y) in zip(tooth_slices, predicts):
            height, width = tooth_slice.shape
            left_x = (left_x * width).item()
            left_y = (left_y * height).item()
            right_x = (right_x * width).item()
            right_y = (right_y * height).item()

            points.append((left_x, left_y, right_x, right_y))
        points = numpy.array(points)
        return points
