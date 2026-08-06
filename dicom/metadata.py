def get_info(volume):
    _, image_size, image_count = volume.GetSize()
    _, pixel_spacing, slice_thickness = volume.GetSpacing()
    return {
        'count': image_count,
        'slice_thickness': slice_thickness,
        'pixel_spacing': pixel_spacing,
        'size': image_size
    }
