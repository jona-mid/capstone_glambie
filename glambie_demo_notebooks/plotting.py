import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


def transform_string(input_string):
    transformed_string = input_string.replace('_', ' ')
    capitalized_string = transformed_string.title()
    region_list = capitalized_string.split(' ')[1:]
    combined_string = " ".join(region_list)
    if 'Western Canada Us' in combined_string:
        combined_string = 'Western Canada & USA'
    return combined_string


def timeseries_as_months(fractional_year_array):

    t0 = math.floor(np.min(fractional_year_array) * 12) / 12.
    t1 = math.ceil(np.max(fractional_year_array) * 12) / 12.
    monthly_array = np.arange(math.ceil((t1 - t0 + 0.00001) * 12)) / 12. + t0

    return monthly_array


def create_change_dataframe_for_single_year(glambie_dataframe_dict, global_dict, chosen_year):
  
    names, changes, errors = [], [], [] 

    for key, val in glambie_dataframe_dict.items():
        names.append(transform_string(key))
        changes.append(val.loc[val.dates == float(chosen_year)]['changes'].values[0])
        errors.append(val.loc[val.dates == float(chosen_year)]['errors'].values[0])
        chosen_year_all_regions_df = pd.DataFrame({'region': names, 'change': changes, 'error': errors })
    
    if 'dates' in global_dict.columns:
        total_change = global_dict.loc[global_dict.dates == float(chosen_year)]['changes'].values[0]
    else:
        total_change = global_dict.loc[global_dict.end_dates == float(chosen_year)]['combined_gt'].values[0]

    return chosen_year_all_regions_df, total_change


def apply_vertical_adjustment(timeseries_to_adjust, reference_timeseries):

    # Drop first zero row and any nan rows in each cumulative dataframe
    filtered_df = timeseries_to_adjust[(abs(timeseries_to_adjust['changes']) > 0)]
    
    # Only apply adjustment if there are any non-nan change values left
    if pd.isna(filtered_df.changes[1:]).all():
        adjusted_timeseries = timeseries_to_adjust.copy()
        adjusted_timeseries.changes = np.nan
    else:
        adjustment = None
        adjustment_date = timeseries_as_months([filtered_df.dates.iloc[0]])[0]
        
        row = reference_timeseries[reference_timeseries.dates == adjustment_date]
        reference_change = row.iloc[0].changes
        row_2 = timeseries_to_adjust[timeseries_to_adjust.dates == adjustment_date]
        adjustment = reference_change - row_2.iloc[0].changes
        
        adjusted_timeseries = filtered_df.copy()
        adjusted_timeseries.changes = adjusted_timeseries.changes + adjustment
        
    return adjusted_timeseries


def single_region_derivative_plot(region_dataframe, region_name, unit):
    
    plt.subplots(1, 1, figsize=(12,8))
    plt.xlim(2000,2023)

    plt.plot(region_dataframe.start_dates, region_dataframe.combined_mwe, linewidth=3, zorder=2, label='Combined change')
    plt.fill_between(region_dataframe.start_dates, region_dataframe.combined_mwe - region_dataframe.combined_mwe_errors,
                     region_dataframe.combined_mwe + region_dataframe.combined_mwe_errors, alpha=0.2)
    plt.hlines(0, region_dataframe.start_dates.values[0], region_dataframe.start_dates.values[-1], linestyle='dashed', color='purple')
    plt.xlabel('Year')
    plt.ylabel('Elevation Change [{}]'.format(unit))
    plt.title(transform_string(region_name) + ': change in elevation, 2000 - 2023', fontsize=18)

    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.8)
    return
    

def single_region_cumulative_plot(cumulative_data, cumulative_errors, region_name, unit,
                                  alimetry_data: pd.DataFrame = None, gravimetry_data: pd.DataFrame = None,
                                  demdiff_and_glaciological_data: pd.DataFrame = None):
    
    if alimetry_data is not None:
        alimetry_data_adjusted = apply_vertical_adjustment(alimetry_data, cumulative_data)
        gravimetry_data_adjusted = apply_vertical_adjustment(gravimetry_data, cumulative_data)
    
    plt.subplots(1, 1, figsize=(12,8))
    plt.xlim(2000,2024)

    plt.plot(cumulative_data.dates, cumulative_data.changes, linewidth=5, zorder=1, alpha=0.7, label='Combined change')
    plt.fill_between(cumulative_data.dates, cumulative_data.changes - cumulative_errors.errors, cumulative_data.changes + cumulative_errors.errors, alpha=0.1)
    if alimetry_data is not None:
        plt.plot(alimetry_data_adjusted.dates, alimetry_data_adjusted.changes, linestyle='dashed', zorder=2, alpha=0.9, linewidth=2, label='Altimetry')
        plt.plot(gravimetry_data_adjusted.dates, gravimetry_data_adjusted.changes, linestyle='dashed', alpha=0.9, linewidth=2,  label='Gravimetry')
        plt.plot(demdiff_and_glaciological_data.dates, demdiff_and_glaciological_data.changes, linestyle='dashed', alpha=0.9, linewidth=2, label='DEM Differencing and glaciological')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Change [{}]'.format(unit))
    plt.title(transform_string(region_name) + ': {} of ice loss, 2000 - 2023'.format(unit), fontsize=18)

    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.8)


    return

def two_region_comparison_plot(region_name, comparison_region_name, cumulative_data_all_gt, cumulative_errors_all_gt, cumulative_data_all_gt_comparison,
                               cumulative_errors_all_gt_comparison, cumulative_data_all_mwe, cumulative_errors_all_mwe,
                               cumulative_data_all_mwe_comparison, cumulative_errors_all_mwe_comparison):
    
    _, axs = plt.subplots(1, 2, figsize=(20,8))
    axs[0].set_xlim(2000,2024)

    axs[0].plot(cumulative_data_all_gt.dates, cumulative_data_all_gt.changes, linewidth=3, zorder=2, label='Combined change - ' + transform_string(region_name))
    axs[0].fill_between(cumulative_data_all_gt.dates, cumulative_data_all_gt.changes - cumulative_errors_all_gt.errors,
                        cumulative_data_all_gt.changes + cumulative_errors_all_gt.errors, alpha=0.2)

    axs[0].plot(cumulative_data_all_gt_comparison.dates, cumulative_data_all_gt_comparison.changes, linewidth=3, zorder=2, label='Combined change - ' + transform_string(comparison_region_name))
    axs[0].fill_between(cumulative_data_all_gt_comparison.dates, cumulative_data_all_gt_comparison.changes - cumulative_errors_all_gt_comparison.errors,
                        cumulative_data_all_gt_comparison.changes + cumulative_errors_all_gt_comparison.errors, alpha=0.2)

    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Cumulative Change [Gt]')
    axs[0].grid(True, alpha=0.8)

    
    axs[1].set_xlim(2000,2024)

    axs[1].plot(cumulative_data_all_mwe.dates, cumulative_data_all_mwe.changes, linewidth=3, zorder=2, label='Combined change - ' + transform_string(region_name))
    axs[1].fill_between(cumulative_data_all_mwe.dates, cumulative_data_all_mwe.changes - cumulative_errors_all_mwe.errors,
                        cumulative_data_all_mwe.changes + cumulative_errors_all_mwe.errors, alpha=0.2)

    axs[1].plot(cumulative_data_all_mwe_comparison.dates, cumulative_data_all_mwe_comparison.changes, linewidth=3, zorder=2, label='Combined change - ' + transform_string(comparison_region_name))
    axs[1].fill_between(cumulative_data_all_mwe_comparison.dates, cumulative_data_all_mwe_comparison.changes - cumulative_errors_all_mwe_comparison.errors,
                        cumulative_data_all_mwe_comparison.changes + cumulative_errors_all_mwe_comparison.errors, alpha=0.2)

    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Cumulative Change [metres water equivalent]')

    axs[0].legend(loc = 'lower left', fontsize=16)
    plt.suptitle('Regional comparison: total ice loss, 2000 - 2023', fontsize=18)
    axs[1].grid(True, alpha=0.8)
    return


def global_cumulative_plot(cumulative_data, cumulative_errors, global_dataframe, unit):
    
    _, axs = plt.subplots(1, 1, figsize=(12,8))
    axs.set_ylim(-1.1, 0.05)
    axs.set_xlim(2000,2024)
    axs_2 = axs.twinx()  
    
    axs.hlines(0, 2000, 2024, linewidth=2, colors=['grey'], linestyle='dashed')
    
    axs.bar(global_dataframe.start_dates+0.5, global_dataframe.combined_mwe, yerr=global_dataframe.combined_mwe_errors, capsize=3, color='coral', ecolor='coral', alpha=0.3, zorder=1) 
    axs.set_ylabel('Annual Change [m w.e. yr$^{-1}$]')

    axs_2.plot(cumulative_data.dates, cumulative_data.changes, linewidth=3, zorder=2)
    axs_2.fill_between(cumulative_data.dates, cumulative_data.changes - cumulative_errors.errors, cumulative_data.changes + cumulative_errors.errors, alpha=0.2, zorder=2)

    axs_2.set_xlabel('Year')
    axs_2.set_ylabel('Cumulative Change [{}]'.format(unit))
    axs_2.set_title('Giga tonnes of global ice loss from 2000 - 2023', fontsize=18)

    axs.grid(True, alpha=0.8)
    return


def global_comparison_stacked_region_plot(cumulative_data_all_gt, cumulative_errors_all_gt, cumulative_data_first_region_gt, first_region,
                                          cumulative_data_second_region_gt: pd.DataFrame = None, second_region: str = None,
                                          cumulative_data_third_region_gt: pd.DataFrame = None, third_region: str = None):
    
    _, axs = plt.subplots(1, 1, figsize=(12,8))
    axs.set_xlim(2000,2024)

    axs.plot(cumulative_data_all_gt.dates, cumulative_data_all_gt.changes, linewidth=3, zorder=2, label='Global change')
    axs.fill_between(cumulative_data_all_gt.dates, cumulative_data_all_gt.changes - cumulative_errors_all_gt.errors, cumulative_data_all_gt.changes + cumulative_errors_all_gt.errors, alpha=0.2)
    
    y = [cumulative_data_first_region_gt.changes,cumulative_data_second_region_gt.changes, cumulative_data_third_region_gt.changes]
    
    axs.stackplot(cumulative_data_all_gt.dates, y, labels=[transform_string(first_region), transform_string(second_region), transform_string(third_region)], alpha=0.4)

    axs.set_xlabel('Year')
    axs.set_ylabel('Cumulative Change [Gt]')
    
    axs.legend(loc = 'lower left', fontsize=16)

    axs.set_title('Global ice loss from 2000 - 2023: contributions from {}, {} and {}'.format(transform_string(first_region), transform_string(second_region), transform_string(third_region)), fontsize=18)
    axs.grid(True, alpha=0.8)


def global_comparison_region_plot(cumulative_data_all_gt, cumulative_errors_all_gt, cumulative_data_first_region_gt, cumulative_errors_first_region_gt, first_region,
                                  cumulative_data_second_region_gt: pd.DataFrame = None, cumulative_errors_second_region_gt: pd.DataFrame = None, second_region: str = None,
                                  cumulative_data_third_region_gt: pd.DataFrame = None, cumulative_errors_third_region_gt: pd.DataFrame = None, third_region: str = None, shaded: bool = False):
    
    plt.subplots(1, 1, figsize=(12,8))

    plt.plot(cumulative_data_all_gt.dates, cumulative_data_all_gt.changes, linewidth=3, zorder=2, label='Global change')
    plt.fill_between(cumulative_data_all_gt.dates, cumulative_data_all_gt.changes - cumulative_errors_all_gt.errors, cumulative_data_all_gt.changes + cumulative_errors_all_gt.errors, alpha=0.2)
    
    plt.plot(cumulative_data_first_region_gt.dates, cumulative_data_first_region_gt.changes, linestyle='dotted', linewidth=3, zorder=2, alpha=0.7, label='{}'.format(transform_string(first_region)))
    if shaded:
        plt.fill_between(cumulative_data_first_region_gt.dates, cumulative_data_first_region_gt.changes, alpha=0.2)
    else:
        plt.fill_between(cumulative_data_first_region_gt.dates, cumulative_data_first_region_gt.changes - cumulative_errors_first_region_gt.errors, cumulative_data_first_region_gt.changes + cumulative_errors_first_region_gt.errors, alpha=0.2)
    
    if cumulative_data_second_region_gt is not None:
        plt.plot(cumulative_data_second_region_gt.dates, cumulative_data_second_region_gt.changes, linestyle='dotted', linewidth=3, zorder=2, alpha=0.7, label='{}'.format(transform_string(second_region)))
        plt.fill_between(cumulative_data_second_region_gt.dates, cumulative_data_second_region_gt.changes - cumulative_errors_second_region_gt.errors, cumulative_data_second_region_gt.changes + cumulative_errors_second_region_gt.errors, alpha=0.2)
    
    if cumulative_data_third_region_gt is not None:
        plt.plot(cumulative_data_third_region_gt.dates, cumulative_data_third_region_gt.changes, linestyle='dotted', linewidth=3, zorder=2, alpha=0.7, label='{}'.format(transform_string(third_region)))
        plt.fill_between(cumulative_data_third_region_gt.dates, cumulative_data_third_region_gt.changes - cumulative_errors_third_region_gt.errors, cumulative_data_third_region_gt.changes + cumulative_errors_third_region_gt.errors, alpha=0.2)

    plt.xlabel('Year')
    plt.ylabel('Cumulative Change [Gt]')
    
    plt.legend(loc = 'lower left', fontsize=16)
    if second_region is not None:
        plt.title('Global ice loss from 2000 - 2023: contributions from {}, {} and {}'.format(transform_string(first_region), transform_string(second_region), transform_string(third_region)), fontsize=18)
    plt.legend(loc='lower left')
    
    return


def histogram_of_region_contributions_to_global_loss(glambie_dataframe_dict, global_dict, chosen_year, colors_list, type):

    chosen_year_all_regions_df, total_change = create_change_dataframe_for_single_year(glambie_dataframe_dict, global_dict, chosen_year)
    
    index = np.arange(19)
    # Show proportion of total change that comes from each region
    _, axs = plt.subplots(1, 1, figsize=(10, 10))
    axs.barh(index, (chosen_year_all_regions_df.change / total_change)*100, tick_label=chosen_year_all_regions_df.region, color=colors_list)

    if type == 'cumulative':
        axs.set_xlabel('Percentage of Global Cumulative Ice Loss per Region [%]'.format(chosen_year))
        plt.suptitle('Global Cumulative Ice Loss, 2000 - {} = {} Gt'.format(chosen_year, round(total_change, 2)), fontsize=18)
    
    elif type == 'derivative':
        axs.set_xlabel('Percentage of Global Ice Loss per Region [%]')
        plt.suptitle('Global Ice Loss in {} = {} Gt'.format(chosen_year, round(total_change, 2)), fontsize=18)
        
    axs.grid(True, alpha=0.8, axis='x')

    plt.tight_layout()


def histogram_of_region_contributions_to_global_loss_two_years(glambie_dataframe_dict, global_dict, chosen_year, comparison_year, type):
    
    chosen_year_all_regions_df, total_change = create_change_dataframe_for_single_year(glambie_dataframe_dict, global_dict, chosen_year)
    comparison_year_all_regions_df, total_change_comparison = create_change_dataframe_for_single_year(glambie_dataframe_dict, global_dict, comparison_year)
    bar_width = 0.35
    
    index = np.arange(19)
    _, axs = plt.subplots(1, 1, figsize=(10, 10))
    if type == 'cumulative':
        axs.barh(index, (chosen_year_all_regions_df.change / total_change)*100, bar_width, tick_label=chosen_year_all_regions_df.region, label='2000 - {}'.format(str(chosen_year)))
        axs.barh(index+bar_width, (comparison_year_all_regions_df.change / total_change_comparison)*100, bar_width, label='2000 - {}'.format(str(comparison_year)))
        axs.set_xlabel('Percentage of Global Cumulative Ice Loss per Region [%]')
    elif type == 'derivative':
        axs.barh(index, (chosen_year_all_regions_df.change / total_change)*100, bar_width, tick_label=chosen_year_all_regions_df.region, label='{}'.format(str(chosen_year)))
        axs.barh(index+bar_width, (comparison_year_all_regions_df.change / total_change_comparison)*100, bar_width, label='{}'.format(str(comparison_year)))
        axs.set_xlabel('Percentage of Global Ice Loss per Region [%]')   
    
    axs.grid(True, alpha=0.8, axis='x')
    
    plt.legend(loc='upper right', fontsize=16)
    

def global_stacked_all_regions_plot(cumulative_data_all_gt, glambie_dataframe_dict):

    y2 = {key:val.changes.tolist() for key, val in glambie_dataframe_dict.items()}
    sorted_dict = dict(sorted(y2.items(), key=lambda e: e[1][-1]))
    
    stack_data = list(sorted_dict.values())
    labels = list(sorted_dict.keys())
    
    labels_formatted = [transform_string(a) for a in labels]

    _, axs = plt.subplots(1, 1, figsize=(12,8))
    axs.set_xlim(2000,2024)

    plt.plot(cumulative_data_all_gt.dates, cumulative_data_all_gt.changes, linewidth=5, zorder=2, label='Global change')
    plt.stackplot(cumulative_data_all_gt.dates, stack_data, labels=labels_formatted, alpha=0.3)

    plt.xlabel('Year')
    plt.ylabel('Cumulative Change [Gt]')

    box = axs.get_position()
    axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs.grid(True, alpha=0.8)

    plt.title('Global ice loss between 2000 and 2023', fontsize=18)