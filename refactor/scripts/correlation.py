from sklearn.metrics import r2_score

def format_p_value(p_value):
    stars = ' ***' if p_value < 0.001 else ' **' if p_value < 0.01 else ' *' if p_value < 0.05 else ''
    p = f'{p_value:.6f}'
    if p == '0.000000':
        p = f'{p_value:.2e}'
    return f'{p}{stars}'

def calculate_icc(targets, predicts):
    n = len(targets)
    k = 2 # Raters = 2 (Targets vs Predicts)

    # Mean Squares
    mean_per_targets = (targets + predicts) / 2
    grand_mean = (targets.sum() + predicts.sum()) / (2 * n)

    # Sum of Squares
    SST = ((targets - grand_mean) ** 2).sum() + ((predicts - grand_mean) ** 2).sum()
    BSS = k * ((mean_per_targets - grand_mean) ** 2).sum() # Between Subject

    # Rater Mean Differences
    mean_targets = targets.mean()
    mean_predicts = predicts.mean()
    JSS = n * ((mean_targets - grand_mean) ** 2 + (mean_predicts - grand_mean) ** 2) # Between Rater

    ESS = SST - BSS - JSS # Residual Error

    # Mean Squares
    BMS = BSS / (n - 1)
    JMS = JSS / (k - 1)
    EMS = ESS / ((n - 1) * (k - 1))

    # ICC(2, 1) Formula
    # (BMS - EMS) / (BMS + (k-1)EMS + k(JMS - EMS) / n)
    icc = (BMS - EMS) / (BMS + (k-1) * EMS + k * (JMS - EMS) / n)
    return icc

if __name__ == '__main__':
    import json
    import pandas

    from argparse import ArgumentParser
    from scipy import stats

    from refactor.src.console import Table
    from refactor.src.summary import load_excel

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('target', type=str)
    parser.add_argument('--exclude', type=str, choices=['slice', 'tooth'])
    parser.add_argument('--single-only', action='store_true')
    args = parser.parse_args()

    experiment_name = args.exp
    target = args.target
    exclude = args.exclude
    single_only = args.single_only

    excel = load_excel(f'bone_loss/{experiment_name}.xlsx')
    if 'extra' in excel.columns:
        excel = excel.drop(excel[~excel['extra'].isna()].index)
        excel = excel.drop(columns='extra')

    with open(f'exclude/{experiment_name}.json') as file:
        excludes = json.load(file)

    indices = []
    if exclude is not None:
        for dataset, patient, fdi, slices in excludes:
            mask = (excel['Dataset'] == dataset) & (excel['Patient'] == patient) & (excel['FDI'] == fdi)
            row = excel[mask]
            if exclude == 'tooth':
                indices.extend(row.index)
                continue
            if not mask.any():
                continue
            bone_losses = [
                row[f'Curve {degree}°'].values[0]
                for degree in range(0, 360, 45)
                if degree not in slices
            ]
            mean_curve = sum(bone_losses) / len(bone_losses)
            max_curve = max(bone_losses)
            excel.loc[mask, ['Mean Curve', 'Max Curve']] = [mean_curve, max_curve]
            bone_losses = [
                row[f'Area {degree}°'].values[0]
                for degree in range(0, 360, 45)
                if degree not in slices
            ]
            mean_area = sum(bone_losses) / len(bone_losses)
            max_area = max(bone_losses)
            excel.loc[mask, ['Mean Area', 'Max Area']] = [mean_area, max_area]
    excel.drop(indices, inplace=True)

    if single_only:
        excel = excel[~excel['FDI'].isin([14, 16, 17, 18, 24, 26, 27, 28, 36, 37, 38, 46, 47, 48])]

    excel = excel[['3D', '2D', 'Mean Curve', 'Mean Area', 'Max Curve', 'Max Area']]

    error_values = {
        f'| {target} - X |': []
    }
    for column in excel.columns.drop(target):
        differences = (excel[target] - excel[column]).abs()
        mean = differences.mean()
        std = differences.std()
        error_values[f'| {target} - X |'].append(f'{mean:.4f} ± {std:.2f}')

    correlations = {
        'pearson': excel.corr()[target].drop(target),
        'r2': [],
        'icc': []
    }

    for column in excel.columns.drop(target):
        r2 = r2_score(excel[target], excel[column])
        icc = calculate_icc(excel[target], excel[column])
        correlations['r2'].append(r2)
        correlations['icc'].append(icc)

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
