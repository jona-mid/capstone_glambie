import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

COLUMNS = {'start_dates', 'end_dates', 'changes', 'errors'}
OUTPUT_DIR = Path('regional_problematic_datasets')

INCLUDED_COLORS = ['#2ecc71', '#27ae60', '#229954', '#1e8449', '#196f3d']
EXCLUDED_COLORS = ['#e74c3c', '#c0392b', '#a93226', '#922b21', '#7b241c']

# INCLUDED_COLORS = ['#2ecc71', '#00acc1', '#558b2f', '#26a69a', '#0277bd', '#7cb342', '#00695c', '#43a047', '#00838f', '#1b5e20']
# EXCLUDED_COLORS = ['#e53935', '#fb8c00', '#d81b60', '#8d6e63', '#f4511e', '#ad1457', '#c62828', '#ef6c00', '#e91e63', '#6d4c41']

#  Load excluded list
excluded_df = pd.read_csv('excluded_datasets_list.csv')
excluded_df = excluded_df[excluded_df['inclusion_possible'].str.strip().str.lower() == 'no']

excluded_set = set()
for _, row in excluded_df.iterrows():
    region = row['region']
    data_group = row['data_group']
    dataset = str(row['dataset']).lower()

    if data_group == 'demdiff_and_glaciological':
        excluded_set.add((region, 'demdiff', dataset))
        excluded_set.add((region, 'glaciological', dataset))
    else:
        excluded_set.add((region, data_group, dataset))

# Datasets to completely remove from plots
SKIP_DATASETS = set()

# Load all datasets
datasets = []
base_dir = Path('.')
skip_files = {'excluded_datasets_list.csv'}
all_csv_files = [f for f in base_dir.rglob('*.csv') if f.name not in skip_files]

for csv_file in all_csv_files:
    try:
        df = pd.read_csv(csv_file)
        if not COLUMNS.issubset(df.columns):
            continue

        parts = csv_file.parts
        if len(parts) < 3:
            continue

        region = parts[-3]
        data_group = parts[-2]
        dataset_name = csv_file.stem

        if dataset_name in SKIP_DATASETS:
            continue

        unit = 'Gt' if data_group == 'gravimetry' else 'm'
        is_excluded = (region, data_group, dataset_name.lower()) in excluded_set

        datasets.append({
            'region': region,
            'data_group': data_group,
            'dataset': dataset_name,
            'unit': unit,
            'is_excluded': is_excluded,
            'data': df,
            'filepath': csv_file
        })
    except Exception:
        continue

# Group by (region, unit)
grouped = defaultdict(list)
for ds in datasets:
    grouped[(ds['region'], ds['unit'])].append(ds)

# Plot per group
for (region, unit), group_datasets in sorted(grouped.items()):
    if not group_datasets:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))

    included = [ds for ds in group_datasets if not ds['is_excluded']]
    excluded = [ds for ds in group_datasets if ds['is_excluded']]

    # Included
    for idx, ds in enumerate(included):
        color = INCLUDED_COLORS[idx % len(INCLUDED_COLORS)]
        dfp = ds['data'].copy()

        dfp['time'] = (dfp['start_dates'] + dfp['end_dates']) / 2
        dfp['errors_abs'] = dfp['errors'].abs()
        dfp = dfp.sort_values('time').reset_index(drop=True)
        if len(dfp) == 0:
            continue

        x_lines, y_lines = [], []
        for _, row in dfp.iterrows():
            x_lines.extend([row['start_dates'], row['end_dates'], np.nan])
            y_lines.extend([row['changes'], row['changes'], np.nan])
        ax.plot(x_lines, y_lines, '-', linewidth=2, alpha=0.7, color=color)

        for _, row in dfp.iterrows():
            x_ribbon = [row['start_dates'], row['end_dates'], row['end_dates'], row['start_dates']]
            y_ribbon = [
                row['changes'] - row['errors_abs'],
                row['changes'] - row['errors_abs'],
                row['changes'] + row['errors_abs'],
                row['changes'] + row['errors_abs'],
            ]
            ax.fill(x_ribbon, y_ribbon, color=color, alpha=0.2, edgecolor='none')

    # Excluded
    for idx, ds in enumerate(excluded):
        color = EXCLUDED_COLORS[idx % len(EXCLUDED_COLORS)]
        dfp = ds['data'].copy()

        dfp['time'] = (dfp['start_dates'] + dfp['end_dates']) / 2
        dfp['errors_abs'] = dfp['errors'].abs()
        dfp = dfp.sort_values('time').reset_index(drop=True)
        if len(dfp) == 0:
            continue

        x_lines, y_lines = [], []
        for _, row in dfp.iterrows():
            x_lines.extend([row['start_dates'], row['end_dates'], np.nan])
            y_lines.extend([row['changes'], row['changes'], np.nan])
        ax.plot(x_lines, y_lines, '-', linewidth=2, alpha=0.7, color=color)

        for _, row in dfp.iterrows():
            x_ribbon = [row['start_dates'], row['end_dates'], row['end_dates'], row['start_dates']]
            y_ribbon = [
                row['changes'] - row['errors_abs'],
                row['changes'] - row['errors_abs'],
                row['changes'] + row['errors_abs'],
                row['changes'] + row['errors_abs'],
            ]
            ax.fill(x_ribbon, y_ribbon, color=color, alpha=0.2, edgecolor='none')

    # Legend
    legend_elements = []
    for idx, ds in enumerate(included):
        color = INCLUDED_COLORS[idx % len(INCLUDED_COLORS)]
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2,
                                         label=f"{ds['dataset']} (Included)"))
    for idx, ds in enumerate(excluded):
        color = EXCLUDED_COLORS[idx % len(EXCLUDED_COLORS)]
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2,
                                         label=f"{ds['dataset']} (Excluded)"))

    unit_label = 'Gravimetry (Gt)' if unit == 'Gt' else 'All Other Methods (m)'
    ax.set_xlabel('Time (year)')
    ax.set_ylabel(f'Change ({unit})')
    ax.set_title(
        f'{region.replace("_", " ").title()} ({unit_label})')
    ax.grid(True, alpha=0.3, linestyle='--')

    if legend_elements:
        ax.legend(handles=legend_elements, loc='best', fontsize=9, ncol=2)

    plt.tight_layout()

    safe_region = region.replace('/', '_').replace('\\', '_')
    output_path = OUTPUT_DIR / f"{safe_region}_{unit}.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()