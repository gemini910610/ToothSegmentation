import colorcet


class Color:
    TOOTH = (235, 223, 180)
    BONE = (212, 161, 230)

GLASBEY_PALETTE = [
    None,
    *[
        (int(color[1:3], 16), int(color[3:5], 16), int(color[5:], 16))
        for color in colorcet.glasbey_light
    ]
]
