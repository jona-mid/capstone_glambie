"""
Compare two glambie runs:
  - Reanalysis_RGI_6_Regression_Proportional (filtered datasets)
  - Reanalysis_RGI_6_Regression_Proportional_including_most (including most datasets)

Produces three plots:
  1. RMSE of the difference in consensus mass change (Gt)
  2. MAE of the difference in consensus mass change (Gt)
  3. Mean relative difference (%)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
RUN1 = os.path.join(BASE, "Reanalysis_RGI_6_Regression_Proportional")
RUN2 = os.path.join(BASE, "Reanalysis_RGI_6_Regression_Proportional_including_most")

# Load glacier mass data (year 2000)
GLACIER_MASS_FILE = os.path.join(os.path.dirname(BASE), "relative rates", "glacier_mass_2000.csv")
glacier_mass_df = pd.read_csv(GLACIER_MASS_FILE, sep=";")
glacier_mass_dict = dict(zip(glacier_mass_df["Region"], glacier_mass_df["Mass"]))

# Region mapping
REGIONS = {
    "1_alaska": "Alaska",
    "2_western_canada_us": "W. Canada & US",
    "3_arctic_canada_north": "Arctic Canada N.",
    "4_arctic_canada_south": "Arctic Canada S.",
    "5_greenland_periphery": "Greenland Per.",
    "6_iceland": "Iceland",
    "7_svalbard": "Svalbard",
    "8_scandinavia": "Scandinavia",
    "9_russian_arctic": "Russian Arctic",
    "10_north_asia": "North Asia",
    "11_central_europe": "Central Europe",
    "12_caucasus_middle_east": "Caucasus & M.E.",
    "13_central_asia": "Central Asia",
    "14_south_asia_west": "South Asia W.",
    "15_south_asia_east": "South Asia E.",
    "16_low_latitudes": "Low Latitudes",
    "17_southern_andes": "Southern Andes",
    "18_new_zealand": "New Zealand",
    "19_antarctic_and_subantarctic": "Antarctic & Sub.",
}

# Region name mapping for glacier mass lookup
MASS_LOOKUP = {
    "Alaska": "Alaska",
    "W. Canada & US": "Western Canada and USA",
    "Arctic Canada N.": "Arctic Canada north",
    "Arctic Canada S.": "Arctic Canada south",
    "Greenland Per.": "Greenland periphery",
    "Iceland": "Iceland",
    "Svalbard": "Svalbard and Jan Mayen",
    "Scandinavia": "Scandinavia",
    "Russian Arctic": "Russian Arctic",
    "North Asia": "North Asia",
    "Central Europe": "Central Europe",
    "Caucasus & M.E.": "Caucasus and Middle East",
    "Central Asia": "Central Asia",
    "South Asia W.": "South Asia west",
    "South Asia E.": "South Asia east",
    "Low Latitudes": "Low latitudes",
    "Southern Andes": "Southern Andes",
    "New Zealand": "New Zealand",
    "Antarctic & Sub.": "Antarctic and subantarctic islands",
    "Global": "Global",
}

# --- Compute metrics for each region ---
region_names = []
rmse_values = []
mae_values = []
rel_diff_values = []
rmse_pct_values = []  # RMSE as percentage of total mass
abs_diff_values = []  # Absolute difference in cumulative mass change

for region_dir, display_name in REGIONS.items():
    region_key = "_".join(region_dir.split("_")[1:])
    csv1 = os.path.join(RUN1, region_dir, "consensus", "csvs",
                        f"consensus_calendar_year_gt_{region_key}.csv")
    csv2 = os.path.join(RUN2, region_dir, "consensus", "csvs",
                        f"consensus_calendar_year_gt_{region_key}.csv")

    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    merged = pd.merge(df1, df2, on="start_dates", suffixes=("_run1", "_run2"))
    diff = merged["changes_run1"] - merged["changes_run2"]

    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))

    cumul_run1 = merged["changes_run1"].sum()
    cumul_run2 = merged["changes_run2"].sum()
    rel_diff = (cumul_run2 - cumul_run1) / abs(cumul_run1) * 100.0 if abs(cumul_run1) > 0.01 else 0.0
    abs_diff = cumul_run2 - cumul_run1  # Absolute difference

    # Get glacier mass for this region
    mass_key = MASS_LOOKUP.get(display_name, display_name)
    glacier_mass = glacier_mass_dict.get(mass_key, None)
    rmse_pct = (rmse / glacier_mass * 100.0) if glacier_mass else 0.0
    
    region_names.append(display_name)
    rmse_values.append(rmse)
    mae_values.append(mae)
    rel_diff_values.append(rel_diff)
    rmse_pct_values.append(rmse_pct)
    abs_diff_values.append(abs_diff)
    # print(f"{display_name:25s}  RMSE = {rmse:8.2f} Gt ({rmse_pct:5.2f}% of mass)  |  MAE = {mae:8.2f} Gt  |  Rel. Diff = {rel_diff:+7.2f}%  |  Abs. Diff = {abs_diff:+8.2f} Gt")

# Global
csv1_global = os.path.join(RUN1, "0_global", "consensus", "csvs", "global_gt.csv")
csv2_global = os.path.join(RUN2, "0_global", "consensus", "csvs", "global_gt.csv")
df1 = pd.read_csv(csv1_global)
df2 = pd.read_csv(csv2_global)
merged = pd.merge(df1, df2, on="start_dates", suffixes=("_run1", "_run2"))
diff = merged["changes_run1"] - merged["changes_run2"]

rmse = np.sqrt(np.mean(diff ** 2))
mae = np.mean(np.abs(diff))

cumul_run1 = merged["changes_run1"].sum()
cumul_run2 = merged["changes_run2"].sum()
rel_diff = (cumul_run2 - cumul_run1) / abs(cumul_run1) * 100.0 if abs(cumul_run1) > 0.01 else 0.0
abs_diff = cumul_run2 - cumul_run1  # Absolute difference

# Get global glacier mass
glacier_mass = glacier_mass_dict.get("Global", None)
rmse_pct = (rmse / glacier_mass * 100.0) if glacier_mass else 0.0

region_names.append("Global")
rmse_values.append(rmse)
mae_values.append(mae)
rel_diff_values.append(rel_diff)
rmse_pct_values.append(rmse_pct)
abs_diff_values.append(abs_diff)

# Plot 1: RMSE
fig1, ax1 = plt.subplots(figsize=(14, 7))
bars1 = ax1.barh(range(len(region_names)), rmse_values)
ax1.set_yticks(range(len(region_names)))
ax1.set_yticklabels(region_names)
ax1.invert_yaxis()
ax1.set_xlabel("RMSE of Mass Change Difference (Gt)")
ax1.set_title("RMSE Difference \n(default vs. all datasets)")
max_rmse = max(rmse_values)
for i, (val, bar) in enumerate(zip(rmse_values, bars1)):
    ax1.text(val + 0.02 * max_rmse, i, f"{val:.2f}", va="center")
ax1.set_xlim(0, max_rmse * 1.15)
fig1.tight_layout()
path1 = os.path.join(BASE, "rmse_difference_between_runs.png")
fig1.savefig(path1, dpi=200, bbox_inches="tight")

# Plot 2: MAE
fig2, ax2 = plt.subplots(figsize=(14, 7))
bars2 = ax2.barh(range(len(region_names)), mae_values)
ax2.set_yticks(range(len(region_names)))
ax2.set_yticklabels(region_names)
ax2.invert_yaxis()
ax2.set_xlabel("MAE of Mass Change Difference (Gt)")
ax2.set_title("MAE Difference \n(default vs. all datasets)")
max_mae = max(mae_values)
for i, (val, bar) in enumerate(zip(mae_values, bars2)):
    ax2.text(val + 0.02 * max_mae, i, f"{val:.2f}", va="center")
ax2.set_xlim(0, max_mae * 1.15)
fig2.tight_layout()
path2 = os.path.join(BASE, "mae_difference_between_runs.png")
fig2.savefig(path2, dpi=200, bbox_inches="tight")

# Plot 3: Relative difference
fig3, ax3 = plt.subplots(figsize=(14, 7))
bars3 = ax3.barh(range(len(region_names)), rel_diff_values)
ax3.set_yticks(range(len(region_names)))
ax3.set_yticklabels(region_names)
ax3.invert_yaxis()
ax3.set_xlabel("Relative Change in Cumulative Mass Loss (%)")
ax3.set_title("Relative Difference in Total Mass Change\n"
              "(default vs. all datasets; positive = more loss, negative = less loss)")
ax3.axvline(0, color="black", linewidth=0.8)
max_abs_rel = max(abs(v) for v in rel_diff_values)
for i, (val, bar) in enumerate(zip(rel_diff_values, bars3)):
    if val >= 0:
        ax3.text(val + 0.02 * max_abs_rel, i, f"+{val:.1f}%", va="center")
    else:
        ax3.text(val - 0.02 * max_abs_rel, i, f"{val:.1f}%", va="center", ha="right")
pad = max_abs_rel * 0.15
ax3.set_xlim(min(rel_diff_values) - pad, max(rel_diff_values) + pad)
fig3.tight_layout()
path3 = os.path.join(BASE, "relative_difference_between_runs.png")
fig3.savefig(path3, dpi=200, bbox_inches="tight")

# Plot 4: Absolute mass change difference
fig4, ax4 = plt.subplots(figsize=(14, 7))
bars4 = ax4.barh(range(len(region_names)), abs_diff_values)
ax4.set_yticks(range(len(region_names)))
ax4.set_yticklabels(region_names)
ax4.invert_yaxis()
ax4.set_xlabel("Absolute Mass Change Difference (Gt)")
ax4.set_title("Absolute Mass Change Difference \n(default vs. all datasets)")
ax4.axvline(0, color="black", linewidth=0.8)

max_abs_diff = max(abs(v) for v in abs_diff_values)
for i, (val, rel, bar) in enumerate(zip(abs_diff_values, rel_diff_values, bars4)):
    if val >= 0:
        ax4.text(val + 0.02 * max_abs_diff, i, f"+{val:.2f} ({rel:+.1f}%)", va="center")
    else:
        ax4.text(val - 0.02 * max_abs_diff, i, f"{val:.2f} ({rel:+.1f}%)", va="center", ha="right")

min_val = min(abs_diff_values)
max_val = max(abs_diff_values)
pad = max_abs_diff * 0.30  # Increased padding for longer annotations
ax4.set_xlim(min_val - pad, max_val + pad)
fig4.tight_layout()
path4 = os.path.join(BASE, "absolute_difference_between_runs.png")
fig4.savefig(path4, dpi=200, bbox_inches="tight")

# plt.show()
