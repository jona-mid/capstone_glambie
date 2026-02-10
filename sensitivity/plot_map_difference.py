"""
Plot the differences between two glambie runs on a world map per RGI region.

Produces three maps:
  1. RMSE difference (Gt) — bubble size + color
  2. MAE difference (Gt) — bubble size + color
  3. Relative difference in cumulative mass change (%) — diverging color
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore", category=FutureWarning)

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
RUN1 = os.path.join(BASE, "Reanalysis_RGI_6_Regression_Proportional")
RUN2 = os.path.join(BASE, "Reanalysis_RGI_6_Regression_Proportional_including_most")

# Region mapping: directory -> (display name, representative lat, lon, region_number)
REGIONS = {
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

# Manual label positions around map periphery (to avoid overlap)
# These are in data coordinates (lon, lat) placed outside the main map areas
LABEL_POSITIONS = {
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


def load_consensus_csv(run_dir, region_dir, region_key):
    """Load a consensus calendar year GT CSV for a given region."""
    csv_path = os.path.join(
        run_dir, region_dir, "consensus", "csvs",
        f"consensus_calendar_year_gt_{region_key}.csv"
    )
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def load_global_csv(run_dir):
    """Load the global GT consensus CSV."""
    csv_path = os.path.join(run_dir, "0_global", "consensus", "csvs", "global_gt.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def compute_metrics(df1, df2):
    """Compute RMSE, MAE, signed relative difference, and absolute mass change."""
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
    
    # Absolute difference in cumulative mass change (run2 - run1)
    abs_diff = cumul_run2 - cumul_run1

    return rmse, mae, rel_diff, abs_diff


# Compute metrics for each region
data = []
for region_dir, (display_name, lat, lon, region_num) in REGIONS.items():
    region_key = "_".join(region_dir.split("_")[1:])
    df1 = load_consensus_csv(RUN1, region_dir, region_key)
    df2 = load_consensus_csv(RUN2, region_dir, region_key)

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
    # print(f"{display_name:25s}  RMSE = {rmse:8.2f} Gt  |  MAE = {mae:8.2f} Gt  |  Rel. Diff = {rel_diff:+7.2f}%  |  Abs. Diff = {abs_diff:+8.2f} Gt")

df = pd.DataFrame(data)

# Load GlacReg 2023 glacier region boundaries
glacreg_path = os.path.join(BASE, "GlacReg_2023", "GTN-G_202307_o1regions.shp") # https://www.gtn-g.ch/database/GlacReg_2023.zip
glacier_regions = gpd.read_file(glacreg_path)

# Global metrics
df1_global = load_global_csv(RUN1)
df2_global = load_global_csv(RUN2)
global_metrics = {}
rmse_g, mae_g, rel_g, abs_diff_g = compute_metrics(df1_global, df2_global)
global_metrics = {"rmse": rmse_g, "mae": mae_g, "rel_diff": rel_g, "abs_diff": abs_diff_g}
# print(f"{'Global':25s}  RMSE = {rmse_g:8.2f} Gt  |  MAE = {mae_g:8.2f} Gt  |  Rel. Diff = {rel_g:+7.2f}%  |  Abs. Diff = {abs_diff_g:+8.2f} Gt")

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

    # Extract configuration
    metric = cfg["metric"]
    title = cfg["title"]
    cmap = cfg["cmap"]
    label = cfg["label"]
    diverging = cfg["diverging"]
    output_path = os.path.join(BASE, cfg["output"])
    fmt = cfg["fmt"]
    unit = cfg["unit"]
    
    values = df[metric].values
    global_val = global_metrics.get(metric, None)
    
    # Create figure with Robinson projection
    fig = plt.figure(figsize=(20, 11))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    
    # Map base features
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#e0e0e0', edgecolor='#aaaaaa', linewidth=0.3, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='#666666', zorder=2)
    
    # Add glaciated areas
    glaciers = cfeature.NaturalEarthFeature(
        'physical', 'glaciated_areas', '50m',
        edgecolor='#2e5f7f', facecolor='#4a8fc4', linewidth=0.2
    )
    ax.add_feature(glaciers, zorder=3, alpha=0.7)
    
    # Add GlacReg 2023 glacier region boundaries
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
    
    # Compute bubble sizes and color normalization
    vals = np.array(values)
    abs_vals = np.abs(vals)
    
    # Size scaling includes global value
    if global_val is not None:
        size_scale = max(abs_vals.max(), abs(global_val))
    else:
        size_scale = abs_vals.max()
    
    sizes = (abs_vals / size_scale) * 5000 
    
    # Color normalization
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
    
    # Plot data bubbles at region centroids
    sc = ax.scatter(
        df["lon"], df["lat"],
        c=vals, s=sizes,
        cmap=cmap, norm=norm,
        edgecolors="0.2", linewidths=1.0,
        zorder=8, alpha=0.75,  # More opaque for more intense colors
        transform=ccrs.PlateCarree()
    )
    
    # Peripheral annotation labels with connecting lines
    for _, row in df.iterrows():
        region_name = row["name"]
        val = row[metric]
        val_str = f"{val:{fmt}}{unit}"
        
        # Get label position (placed around periphery)
        if region_name in LABEL_POSITIONS:
            label_lon, label_lat = LABEL_POSITIONS[region_name]
            
            # Draw label text
            ax.text(
                label_lon, label_lat,
                f"{region_name}\n{val_str}",
                fontsize=7, ha="center", va='center',
                color="0.1",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.4", alpha=0.85, linewidth=0.5),
                transform=ccrs.PlateCarree(),
                zorder=12
            )
            
            # Draw connecting line from label to bubble
            ax.plot(
                [label_lon, row["lon"]], [label_lat, row["lat"]],
                color='0.5', linewidth=0.4, linestyle='-', alpha=0.5,
                transform=ccrs.PlateCarree(),
                zorder=7
            )
    
    # Global bubble (central position)
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
        
        # Global label
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
    
    # Colorbar
    cb = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.03, aspect=20, orientation='vertical')
    cb.set_label(label, fontsize=11)
    cb.ax.tick_params(labelsize=9)
    
    # Title
    ax.set_title(title, fontsize=13, pad=15)
    
    # Save
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
