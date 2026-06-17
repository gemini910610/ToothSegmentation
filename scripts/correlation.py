import pandas

def format_p_value(p_value):
    stars = ' ***' if p_value < 0.001 else ' **' if p_value < 0.01 else ' *' if p_value < 0.05 else ''
    p = f'{p_value:.6f}'
    if p == '0.000000':
        p = f'{p_value:.2e}'
    return f'{p}{stars}'

if __name__ == '__main__':
    from argparse import ArgumentParser
    from scipy import stats

    from src.console import Table
    from src.summary import load_excel

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('target', type=str)
    args = parser.parse_args()

    experiment_name = args.exp
    target = args.target

    excel = load_excel(f'bone_loss/{experiment_name}.xlsx')
    excel = excel[['3D', '2D', 'Curve 0°', 'Area 0°', 'Mean Curve', 'Mean Area', 'Max Curve', 'Max Area']]

    error_values = {
        f'{target} - X': [],
        f'| {target} - X |': []
    }
    for column in excel.columns.drop(target):
        differences = excel[column] - excel[target]
        mean = differences.mean()
        std = differences.std()
        error_values[f'{target} - X'].append(f'{mean:.4f} ± {std:.2f}')

        differences = differences.abs()
        mean = differences.mean()
        std = differences.std()
        error_values[f'| {target} - X |'].append(f'{mean:.4f} ± {std:.2f}')

    correlations = {
        'pearson': excel.corr()[target].drop(target)
    }

    p_values = {
        'shapiro': [],
        't-test': [],
        'wilcoxon': []
    }

    for column in excel.columns.drop(target):
        differences = excel[target] - excel[column]
        _, p = stats.shapiro(differences)
        p_values['shapiro'].append(format_p_value(p))
        if p > 0.05: # 無法拒絕常態假設 -> 視為近似常態
            # t-test
            _, p = stats.ttest_rel(excel[target], excel[column])
            p_values['t-test'].append(format_p_value(p))
            p_values['wilcoxon'].append('-')
        else: # 拒絕常態假設
            # wilcoxon
            _, p = stats.wilcoxon(excel[target], excel[column])
            p_values['t-test'].append('-')
            p_values['wilcoxon'].append(format_p_value(p))

    data = pandas.DataFrame({
        **error_values,
        **correlations,
        **p_values
    })
    table = Table(
        ['X', *data.columns],
        *[
            [index, *row]
            for index, row in data.iterrows()
        ]
    )
    table.display()
