import pandas

from tbparse import SummaryReader

class Summary:
    def __init__(self, scalars):
        self.scalars = scalars

    def __getitem__(self, query):
        if query == 'step' or query == 'value':
            return self.scalars[query]
        if query.startswith('Fold'):
            mask = self.scalars['fold'] == query
            drop_column = 'fold'
        elif query == 'Train' or query == 'Val':
            mask = self.scalars['mode'] == query
            drop_column = 'mode'
        else:
            mask = self.scalars['metric'] == query
            drop_column = 'metric'
        scalars = self.scalars[mask]
        scalars = scalars.drop(columns=drop_column)
        return Summary(scalars)

    def __repr__(self):
        return repr(self.scalars)

    def __getattr__(self, attribute):
        return getattr(self.scalars, attribute)

    def __len__(self):
        return len(self.scalars)

def load_summary(log_dir):
    reader = SummaryReader(log_dir, extra_columns={'dir_name'})
    scalars = reader.scalars

    dir_names = scalars['dir_name'].unique()
    dir_names = pandas.DataFrame({'dir_name': dir_names})

    pattern = r'(?P<fold>Fold_\d+)/(?P<metric>.+)_(?P<mode>[^/_]+)$'
    extracted = dir_names['dir_name'].str.extract(pattern)

    dir_names = pandas.concat([dir_names, extracted], axis=1)
    scalars = scalars.merge(dir_names, on='dir_name')

    scalars = scalars[['fold', 'metric', 'mode', 'step', 'value']]
    scalars[['fold', 'metric', 'mode']] = scalars[['fold', 'metric', 'mode']].astype('category')

    return Summary(scalars)

def load_excel(filename):
    excel = pandas.read_excel(filename, header=None, skiprows=2)
    key_columns = ['Dataset', 'Patient', 'FDI']
    columns = [
        *key_columns, '3D', '2D',
        *[
            f'{title} {item}°' if isinstance(item, int) else f'{item} {title}'
            for item in [*range(0, 360, 45), 'Mean', 'Max']
            for title in ('Curve', 'Area')
        ]
    ]
    if len(excel.columns) != len(columns):
        raise ValueError(f'Expected {len(columns)} columns, got {len(excel.columns)} in "{filename}"')
    excel.columns = columns
    excel[key_columns] = excel[key_columns].ffill()
    return excel
