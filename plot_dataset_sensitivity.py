import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# Paths and configuration

glambie_runs = Path("input") / "glambie_runs"
input_sensitivity = Path("input") / "sensitivity"
output_sensitivity = Path("output") / "sensitivity"

run1 = glambie_runs / "Reanalysis_RGI_6_default"
run2 = glambie_runs / "Reanalysis_RGI_6_including_most"

glacier_mass_file = input_sensitivity / "glacier_mass_2000.csv"
glacier_mass_df = pd.read_csv(str(glacier_mass_file), sep=";")
glacier_mass_dict = dict(zip(glacier_mass_df["Region"], glacier_mass_df["Mass"]))

# Region mappings

regions_bar = {
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

mass_lookup = {
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

regions_map = {
    "1_alaska":                      ("Alaska",            63.0, -150.0, 1),
    "2_western_canada_us":           ("W. Canada & US",    50.0, -122.0, 2),
    "3_arctic_canada_north":         ("Arctic Canada N.",  77.0,  -82.0, 3),
    "4_arctic_canada_south":         ("Arctic Canada S.",  66.0,  -70.0, 4),
    "5_greenland_periphery":         ("Greenland Per.",    72.0,  -42.0, 5),
    "6_iceland":                     ("Iceland",           65.0,  -19.0, 6),
    "7_svalbard":                    ("Svalbard",          78.0,   17.0, 7),
    "8_scandinavia":                 ("Scandinavia",       67.0,   15.0, 8),
    "9_russian_arctic":              ("Russian Arctic",    77.0,   60.0, 9),
    "10_north_asia":                 ("North Asia",        50.0,   90.0, 10),
    "11_central_europe":             ("Central Europe",    47.0,   11.0, 11),
    "12_caucasus_middle_east":       ("Caucasus & M.E.",   42.0,   44.0, 12),
    "13_central_asia":               ("Central Asia",      40.0,   75.0, 13),
    "14_south_asia_west":            ("South Asia W.",     35.0,   74.0, 14),
    "15_south_asia_east":            ("South Asia E.",     30.0,   90.0, 15),
    "16_low_latitudes":              ("Low Latitudes",     -1.0,  -78.0, 16),
    "17_southern_andes":             ("Southern Andes",   -47.0,  -73.0, 17),
    "18_new_zealand":                ("New Zealand",      -44.0,  170.0, 18),
    "19_antarctic_and_subantarctic": ("Antarctic & Sub.", -70.0,    0.0, 19),
}

label_positions = {
    "Alaska":            (-170.0, 55.0),
    "W. Canada & US":    (-140.0, 42.0),
    "Arctic Canada N.":  (-100.0, 82.0),
    "Arctic Canada S.":  (-85.0, 72.0),
    "Greenland Per.":    (-50.0, 78.0),
    "Iceland":           (-30.0, 68.0),
    "Svalbard":          (10.0, 82.0),
    "Scandinavia":       (5.0, 72.0),
    "Russian Arctic":    (70.0, 82.0),
    "North Asia":        (110.0, 55.0),
    "Central Europe":    (0.0, 50.0),
    "Caucasus & M.E.":   (55.0, 48.0),
    "Central Asia":      (85.0, 45.0),
    "South Asia W.":     (65.0, 30.0),
    "South Asia E.":     (100.0, 25.0),
    "Low Latitudes":     (-90.0, -8.0),
    "Southern Andes":    (-80.0, -52.0),
    "New Zealand":       (175.0, -50.0),
    "Antarctic & Sub.":  (-10.0, -75.0),
}

# Helper functions

def load_consensus_csv(run_dir, region_dir, region_key):
    csv_path = run_dir / region_dir / "consensus" / "csvs" / f"consensus_calendar_year_gt_{region_key}.csv"
    if csv_path.exists():
        return pd.read_csv(str(csv_path))
    return None

def load_global_csv(run_dir):
    csv_path = run_dir / "0_global" / "consensus" / "csvs" / "global_gt.csv"
    if csv_path.exists():
        return pd.read_csv(str(csv_path))
    return None

def compute_metrics(df1, df2):
    merged = pd.merge(df1, df2, on="start_dates", suffixes=("_run1", "_run2"))
    diff = merged["changes_run1"] - merged["changes_run2"]
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    cumul_run1 = merged["changes_run1"].sum()
    cumul_run2 = merged["changes_run2"].sum()
    if abs(cumul_run1) > 0.01:
        rel_diff = (cumul_run2 - cumul_run1) / abs(cumul_run1) * 100.0
    else:
        rel_diff = 0.0
    abs_diff = cumul_run2 - cumul_run1
    return rmse, mae, rel_diff, abs_diff

def compute_metrics_with_mass(df1, df2, glacier_mass):
    rmse, mae, rel_diff, abs_diff = compute_metrics(df1, df2)
    rmse_pct = (rmse / glacier_mass * 100.0) if glacier_mass else 0.0
    return rmse, mae, rel_diff, abs_diff, rmse_pct

# Bar charts

region_names = []
rmse_values = []
mae_values = []
rel_diff_values = []
rmse_pct_values = []
abs_diff_values = []

for region_dir, display_name in regions_bar.items():
    region_key = "_".join(region_dir.split("_")[1:])
    csv1 = run1 / region_dir / "consensus" / "csvs" / f"consensus_calendar_year_gt_{region_key}.csv"
    csv2 = run2 / region_dir / "consensus" / "csvs" / f"consensus_calendar_year_gt_{region_key}.csv"

    df1 = pd.read_csv(str(csv1))
    df2 = pd.read_csv(str(csv2))

    mass_key = mass_lookup.get(display_name, display_name)
    glacier_mass = glacier_mass_dict.get(mass_key, None)

    rmse, mae, rel_diff, abs_diff, rmse_pct = compute_metrics_with_mass(df1, df2, glacier_mass)

    region_names.append(display_name)
    rmse_values.append(rmse)
    mae_values.append(mae)
    rel_diff_values.append(rel_diff)
    rmse_pct_values.append(rmse_pct)
    abs_diff_values.append(abs_diff)

csv1_global = run1 / "0_global" / "consensus" / "csvs" / "global_gt.csv"
csv2_global = run2 / "0_global" / "consensus" / "csvs" / "global_gt.csv"
df1 = pd.read_csv(str(csv1_global))
df2 = pd.read_csv(str(csv2_global))
rmse, mae, rel_diff, abs_diff, rmse_pct = compute_metrics_with_mass(df1, df2, glacier_mass_dict.get("Global", None))
region_names.append("Global")
rmse_values.append(rmse)
mae_values.append(mae)
rel_diff_values.append(rel_diff)
rmse_pct_values.append(rmse_pct)
abs_diff_values.append(abs_diff)

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
path1 = output_sensitivity / "rmse_difference_between_runs.png"
fig1.savefig(str(path1), dpi=200, bbox_inches="tight")

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
path2 = output_sensitivity / "mae_difference_between_runs.png"
fig2.savefig(str(path2), dpi=200, bbox_inches="tight")

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
path3 = output_sensitivity / "relative_difference_between_runs.png"
fig3.savefig(str(path3), dpi=200, bbox_inches="tight")

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
pad = max_abs_diff * 0.30
ax4.set_xlim(min_val - pad, max_val + pad)
fig4.tight_layout()
path4 = output_sensitivity / "absolute_difference_between_runs.png"
fig4.savefig(str(path4), dpi=200, bbox_inches="tight")

plt.close('all')

# Map plots

glacreg_path = input_sensitivity / "GlacReg_2023" / "GTN-G_202307_o1regions.shp"
glacier_regions = gpd.read_file(str(glacreg_path))

data = []
for region_dir, (display_name, lat, lon, region_num) in regions_map.items():
    region_key = "_".join(region_dir.split("_")[1:])
    df1 = load_consensus_csv(run1, region_dir, region_key)
    df2 = load_consensus_csv(run2, region_dir, region_key)
    rmse, mae, rel_diff, abs_diff = compute_metrics(df1, df2)
    data.append({
        "name": display_name,
        "lat": lat,
        "lon": lon,
        "region_num": region_num,
        "rmse": rmse,
        "mae": mae,
        "rel_diff": rel_diff,
        "abs_diff": abs_diff,
    })

df = pd.DataFrame(data)

df1_global = load_global_csv(run1)
df2_global = load_global_csv(run2)
global_metrics = {}
if df1_global is not None and df2_global is not None:
    rmse_g, mae_g, rel_g, abs_diff_g = compute_metrics(df1_global, df2_global)
    global_metrics = {"rmse": rmse_g, "mae": mae_g, "rel_diff": rel_g, "abs_diff": abs_diff_g}

map_configs = [
    {
        "metric": "rmse",
        "title": "RMSE Difference Between Glambie Runs\n(default vs. all datasets)",
        "cmap": "YlOrRd",
        "label": "RMSE (Gt)",
        "diverging": False,
        "output": "map_rmse_difference.png",
        "fmt": ".2f",
        "unit": " Gt",
    },
    {
        "metric": "mae",
        "title": "MAE Difference Between Glambie Runs\n(default vs. all datasets)",
        "cmap": "YlOrRd",
        "label": "MAE (Gt)",
        "diverging": False,
        "output": "map_mae_difference.png",
        "fmt": ".2f",
        "unit": " Gt",
    },
    {
        "metric": "rel_diff",
        "title": "Relative Difference in Total Mass Change\n(default vs. all datasets)",
        "cmap": "RdBu_r",
        "label": "Relative Difference (%)",
        "diverging": True,
        "output": "map_relative_difference.png",
        "fmt": "+.1f",
        "unit": "%",
    },
    {
        "metric": "abs_diff",
        "title": "Absolute Mass Change Difference Between Glambie Runs\n(default vs. all datasets)",
        "cmap": "RdBu_r",
        "label": "Absolute Difference (Gt)",
        "diverging": True,
        "output": "map_absolute_difference.png",
        "fmt": "+.2f",
        "unit": " Gt",
    },
]

for cfg in map_configs:
    metric = cfg["metric"]
    title = cfg["title"]
    cmap = cfg["cmap"]
    label = cfg["label"]
    diverging = cfg["diverging"]
    output_path = output_sensitivity / cfg["output"]
    fmt = cfg["fmt"]
    unit = cfg["unit"]

    values = df[metric].values
    global_val = global_metrics.get(metric, None)

    fig = plt.figure(figsize=(20, 11))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#e0e0e0', edgecolor='#aaaaaa', linewidth=0.3, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='#666666', zorder=2)

    glaciers = cfeature.NaturalEarthFeature(
        'physical', 'glaciated_areas', '50m',
        edgecolor='#2e5f7f', facecolor='#4a8fc4', linewidth=0.2
    )
    ax.add_feature(glaciers, zorder=3, alpha=0.7)

    if glacier_regions is not None:
        ax.add_geometries(
            glacier_regions.geometry,
            crs=ccrs.PlateCarree(),
            facecolor='none',
            edgecolor='#5a5a5a',
            linewidth=1.0,
            linestyle='--',
            zorder=4,
            alpha=0.6
        )
    ax.set_global()

    vals = np.array(values)
    abs_vals = np.abs(vals)

    if global_val is not None:
        size_scale = max(abs_vals.max(), abs(global_val))
    else:
        size_scale = abs_vals.max()

    sizes = (abs_vals / size_scale) * 5000

    if diverging:
        all_vals = list(vals) + ([global_val] if global_val is not None else [])
        vmax_abs = max(abs(min(all_vals)), abs(max(all_vals)))
        vmin, vmax = -vmax_abs, vmax_abs
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        vmin = 0
        all_vals = list(vals) + ([global_val] if global_val is not None else [])
        vmax = max(all_vals)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(
        df["lon"], df["lat"],
        c=vals, s=sizes,
        cmap=cmap, norm=norm,
        edgecolors="0.2", linewidths=1.0,
        zorder=8, alpha=0.75,
        transform=ccrs.PlateCarree()
    )

    for _, row in df.iterrows():
        region_name = row["name"]
        val = row[metric]
        val_str = f"{val:{fmt}}{unit}"

        if region_name in label_positions:
            label_lon, label_lat = label_positions[region_name]

            ax.text(
                label_lon, label_lat,
                f"{region_name}\n{val_str}",
                fontsize=7, ha="center", va='center',
                color="0.1",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.4", alpha=0.85, linewidth=0.5),
                transform=ccrs.PlateCarree(),
                zorder=12
            )

            ax.plot(
                [label_lon, row["lon"]], [label_lat, row["lat"]],
                color='0.5', linewidth=0.4, linestyle='-', alpha=0.5,
                transform=ccrs.PlateCarree(),
                zorder=7
            )

    if global_val is not None:
        global_lon, global_lat = 0.0, -20.0
        global_size = (abs(global_val) / size_scale) * 5000

        ax.scatter(
            [global_lon], [global_lat],
            c=[global_val], s=[global_size * 1.5],
            cmap=cmap, norm=norm,
            edgecolors="0.1", linewidths=1.0,
            zorder=11, alpha=0.9,
            marker="o",
            transform=ccrs.PlateCarree()
        )

        val_str = f"{global_val:{fmt}}{unit}"
        ax.text(
            global_lon, global_lat - 8,
            f"Global\n{val_str}",
            fontsize=9, fontweight="bold", ha="center", va='top',
            color="0.05",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.3", alpha=0.95, linewidth=1),
            transform=ccrs.PlateCarree(),
            zorder=12
        )

    cb = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.03, aspect=20, orientation='vertical')
    cb.set_label(label, fontsize=11)
    cb.ax.tick_params(labelsize=9)

    ax.set_title(title, fontsize=13, pad=15)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200)
    plt.close(fig)
