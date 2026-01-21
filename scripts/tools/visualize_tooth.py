import os
import numpy

from argparse import ArgumentParser
from PySide6.QtWidgets import  QApplication
from scripts.tools.visualize import VolumeColorizer, VolumeViewer, DataManager, get_patient_fold_mapping, Mode
from src.config import load_config
from scripts.post_processing.watershed import split_k_component

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('data', type=str)
    parser.add_argument('--label', type=int, default=-1)
    parser.add_argument('--cluster', type=int, default=-1)
    args = parser.parse_args()

    experiment_name = args.exp
    data = args.data
    label = args.label
    cluster = args.cluster

    config = load_config(os.path.join('logs', experiment_name, 'config.toml'))
    patient_fold_map = get_patient_fold_mapping(config)

    app = QApplication([])

    data_manager = DataManager(experiment_name, patient_fold_map, [Mode.CONNECTED_COMPONENT, Mode.WATERSHED], [1])
    cc_volume, watershed_volume = data_manager.load_data(data)
    if label != -1:
        volume = cc_volume == label
        if cluster > 1:
            volume = split_k_component(volume, h=1, k=cluster)
        volume = volume.astype(numpy.int32)

    colorizer = VolumeColorizer()
    cc_volume, _ = colorizer.color_components(cc_volume)
    watershed_volume, _ = colorizer.color_components(watershed_volume)
    if label != -1:
        volume, _ = colorizer.color_components(volume)

    window = VolumeViewer(3, (512, 512))
    window.move(0, 0)
    window.setWindowTitle(f'{experiment_name}/{data}')
    window.views[0].setTitle('Connected Component')
    window.views[0].view.update_volume(cc_volume)
    window.views[1].setTitle('Watershed')
    window.views[1].view.update_volume(watershed_volume)
    if label != -1:
        window.views[2].setTitle(f'Label {label}')
        window.views[2].view.update_volume(volume)

    window.show()

    app.exec()
