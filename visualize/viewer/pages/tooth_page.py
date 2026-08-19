import numpy

from constants.label import Label
from constants.mode import Mode
from post_processing.watershed import split_component
from visualize.viewer.pages.connected_component_page import ConnectedComponentPage


class ToothPage(ConnectedComponentPage):
    label = Label.TOOTH
    is_bone = False
    threshold_label = 'Tooth Threshold'
    default_threshold = 8200
    output_mode = Mode.TOOTH_CONNECTED_COMPONENT

    def _after_save(self, patient_dir):
        watershed_volume = split_component(self.filtered_volume)
        numpy.save(patient_dir / f'{Mode.WATERSHED}.npy', watershed_volume)
