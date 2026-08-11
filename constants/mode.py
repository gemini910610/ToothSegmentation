class Mode:
    IMAGE = 'image'
    PREDICT = 'predict'

    @staticmethod
    def items():
        return [
            Mode.PREDICT
        ]

    @staticmethod
    def get_title(mode):
        return {
            Mode.PREDICT: 'Predict'
        }[mode]
