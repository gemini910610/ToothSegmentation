import SimpleITK

from utils.image import imwrite


def save_volume(volume, output_dir):
    output_dir.mkdir(parents=True)
    volume = SimpleITK.GetArrayFromImage(volume)

    for index, image in enumerate(volume, 1):
        if not image.any():
            continue

        image_path = output_dir / f'{index}.png'
        imwrite(image_path, image)
