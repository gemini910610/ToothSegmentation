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
