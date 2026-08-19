from constants.label import Label
from constants.mode import Mode
from visualize.viewer.pages.connected_component_page import ConnectedComponentPage


class BonePage(ConnectedComponentPage):
    label = Label.BONE
    is_bone = True
    threshold_label = 'Bone Threshold'
    default_threshold = 45000
    output_mode = Mode.BONE_CONNECTED_COMPONENT
