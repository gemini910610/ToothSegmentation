from textual.containers import Container, Grid, Horizontal


class BaseHorizontal(Horizontal):
    def __init__(self, *, height='auto', align='left'):
        super().__init__()
        self.styles.height = height
        self.styles.align_horizontal = align

class BaseContainer(Container):
    def __init__(self, *, width='auto', height='auto'):
        super().__init__()
        self.styles.width = width
        self.styles.height = height

class BaseGrid(Grid):
    def __init__(self, *, rows, columns, width='auto', height='auto', background=None):
        super().__init__()
        self.styles.grid_size_rows = rows
        self.styles.grid_size_columns = columns
        self.styles.width = width
        self.styles.height = height
        self.styles.background = background
