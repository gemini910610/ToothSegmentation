import cv2
import SimpleITK


def save_volume(volume, output_dir):
    output_dir.mkdir(parents=True)
    volume = SimpleITK.GetArrayFromImage(volume)

    for index, image in enumerate(volume, 1):
        image_path = output_dir / f'{index}.png'
        if not image.any():
            continue

        cv2.imwrite(image_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
