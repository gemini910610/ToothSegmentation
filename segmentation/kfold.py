from pathlib import Path

from segmentation.dataset import find_datasets


def group_data(config=None):
    patients = []
    groups = []

    for dataset in find_datasets():
        folders = [
            folder
            for folder in (Path('datasets') / dataset / 'mask').iterdir()
            if folder.is_dir()
        ]
        folders.sort()

        for folder in folders:
            patients.append(folder.name)
            images = [
                image
                for image in folder.iterdir()
                if image.suffix == '.png'
            ]
            groups.extend([Path(dataset) / folder.name] * len(images))

    return patients, groups
