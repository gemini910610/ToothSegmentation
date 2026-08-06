from pathlib import Path


def group_data(config):
    patients = []
    groups = []

    for dataset in config.datasets:
        image_dir = Path('datasets') / dataset / 'mask'

        folders = [
            folder
            for folder in image_dir.iterdir()
            if folder.is_dir()
        ]
        folders.sort()

        for folder in folders:
            patients.append(folder.stem)
            images = [
                image
                for image in folder.iterdir()
                if image.suffix == '.png'
            ]
            # groups.extend([f'{dataset}/{folder.stem}'] * len(images))
            groups.extend([Path(dataset) / folder.stem] * len(images))

    return patients, groups
