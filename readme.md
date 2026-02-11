# Capstone Project

## Contents
- input/ – glambie reference data, glambie runs, input datasets for the glambie algorithm and initial glacer mass data
- output/ – generated plots: datasets plots, relative glacier mass change, sensitivity
- analysis.ipynb: combined script contaning the analysis

## Note on excluded datasets
A small code change as necessary to be able to run excluded datasets: add *shift_timeseries_to_annual_grid_proportionally()*

        # Fix for running excluded datasets that are at annual resolution but not on the desired annual grid
        if ds.data.max_temporal_resolution >= 1 and not ds.timeseries_is_annual_grid(year_type=year_type):
            ds = ds.shift_timeseries_to_annual_grid_proportionally(year_type=year_type)
        datasets.append(ds.convert_timeseries_to_annual_trends(year_type=year_type))
