from sklearn.metrics import r2_score


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
    from argparse import ArgumentParser

    import pandas

    from utils.console import Table
    from utils.excel import load_excel

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('--exclude', type=str, choices=['slice', 'tooth'])
    parser.add_argument('--single-only', action='store_true')
    args = parser.parse_args()

    experiment_name = args.exp
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
                row[f'Contour {degree}°'].values[0]
                for degree in range(0, 360, 45)
                if degree not in slices
            ]
            mean_contour = sum(bone_losses) / len(bone_losses)
            max_contour = max(bone_losses)
            excel.loc[mask, ['Mean Contour', 'Max Contour']] = [mean_contour, max_contour]
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

    excel = excel[['3D', '2D', 'Mean Contour', 'Mean Area', 'Max Contour', 'Max Area']]

    error_values = {
        'MAE': []
    }
    for column in excel.columns.drop('3D'):
        differences = (excel['3D'] - excel[column]).abs()
        mean = differences.mean()
        std = differences.std()
        error_values['MAE'].append(f'{mean:.4f} ± {std:.2f}')

    correlations = {
        'Pearson': excel.corr()['3D'].drop('3D'),
        'R² Score': [],
        'ICC': []
    }

    for column in excel.columns.drop('3D'):
        r2 = r2_score(excel['3D'], excel[column])
        icc = calculate_icc(excel['3D'], excel[column])
        correlations['R² Score'].append(r2)
        correlations['ICC'].append(icc)

    data = pandas.DataFrame({
        **error_values,
        **correlations
    })
    table = Table(
        ['', *data.columns],
        *[
            [index, *row]
            for index, row in data.iterrows()
        ]
    )
    table.display()
