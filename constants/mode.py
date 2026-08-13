class Mode:
    IMAGE = 'image'
    PREDICT = 'predict'
    TOOTH_CONNECTED_COMPONENT = 'tooth_cc'
    BONE_CONNECTED_COMPONENT = 'bone_cc'
    WATERSHED = 'watershed'

    @staticmethod
    def items():
        return [
            Mode.PREDICT,
            Mode.TOOTH_CONNECTED_COMPONENT,
            Mode.BONE_CONNECTED_COMPONENT,
            Mode.WATERSHED
        ]

    @staticmethod
    def get_title(mode):
        return {
            Mode.PREDICT: 'Predict',
            Mode.TOOTH_CONNECTED_COMPONENT: 'Connected Component (Tooth)',
            Mode.BONE_CONNECTED_COMPONENT: 'Connected Component (Bone)',
            Mode.WATERSHED: 'Watershed'
        }[mode]
