class Mode:
    IMAGE = 'image'
    PREDICT = 'predict'
    TOOTH_CONNECTED_COMPONENT = 'tooth_cc'
    BONE_CONNECTED_COMPONENT = 'bone_cc'
    WATERSHED = 'watershed'
    REFINE_SPLIT = 'refine_split'
    REFINE_MERGE = 'refine_merge'
    UNTRUNCATED = 'untruncated'
    CLEANED = 'cleaned'
    TOOTH_FILLED = 'tooth_filled'
    BONE_FILLED = 'bone_filled'
    RELABELED = 'relabeled'
    FDI = 'fdi'

    @staticmethod
    def items():
        return [
            Mode.PREDICT,
            Mode.TOOTH_CONNECTED_COMPONENT,
            Mode.BONE_CONNECTED_COMPONENT,
            Mode.WATERSHED,
            Mode.REFINE_SPLIT,
            Mode.REFINE_MERGE,
            Mode.UNTRUNCATED,
            Mode.CLEANED,
            Mode.TOOTH_FILLED,
            Mode.BONE_FILLED,
            Mode.RELABELED,
            Mode.FDI
        ]

    @staticmethod
    def get_title(mode):
        return {
            Mode.PREDICT: 'Predict',
            Mode.TOOTH_CONNECTED_COMPONENT: 'Connected Component (Tooth)',
            Mode.BONE_CONNECTED_COMPONENT: 'Connected Component (Bone)',
            Mode.WATERSHED: 'Watershed',
            Mode.REFINE_SPLIT: 'Refine Split',
            Mode.REFINE_MERGE: 'Refine Merge',
            Mode.UNTRUNCATED: 'Untruncated',
            Mode.CLEANED: 'Cleaned',
            Mode.TOOTH_FILLED: 'Tooth Filled',
            Mode.BONE_FILLED: 'Bone Filled',
            Mode.RELABELED: 'Relabeled',
            Mode.FDI: 'FDI'
        }[mode]
