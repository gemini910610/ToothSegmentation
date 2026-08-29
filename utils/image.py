from pathlib import Path

import cv2
import numpy


def imread(path):
    buffer = numpy.fromfile(path, dtype=numpy.uint8)
    if buffer.size == 0:
        return None

    return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)


def imwrite(path, image):
    suffix = Path(path).suffix
    _, buffer = cv2.imencode(suffix, image)
    buffer.tofile(path)
