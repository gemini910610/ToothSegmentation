import numpy
import SimpleITK


def normalize_volume(volume, min_value=-1000, max_value=4500):
    volume = SimpleITK.IntensityWindowing(volume, min_value, max_value) # [min_value, max_value] -> [0, 255]
    volume = SimpleITK.Cast(volume, SimpleITK.sitkUInt8)

    return volume

def resample_volume(volume, new_spacing=None):
    if new_spacing is None:
        new_spacing = 0.25

    size = numpy.array(volume.GetSize())
    spacing = numpy.array(volume.GetSpacing())
    new_spacing = numpy.array([new_spacing] * 3)
    new_size = size * spacing / new_spacing

    new_size = [int(size) for size in new_size]
    new_spacing = [float(spacing) for spacing in new_spacing]

    volume = SimpleITK.Flip(volume, [False, False, True])

    resample = SimpleITK.ResampleImageFilter()
    resample.SetReferenceImage(volume)
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(SimpleITK.sitkLinear)

    volume = resample.Execute(volume)
    volume = SimpleITK.Flip(volume, [False, False, True])

    return volume

def process_volume(volume, min_value=-1000, max_value=4500, new_spacing=None):
    volume = normalize_volume(volume, min_value, max_value)
    volume = resample_volume(volume, new_spacing)

    return volume
