import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def two_region_comparison_plot_2(region_name, data_distinction, comparison_data_distinction, cumulative_data_all_gt, cumulative_errors_all_gt, cumulative_data_all_gt_comparison,
                               cumulative_errors_all_gt_comparison, cumulative_data_all_mwe, cumulative_errors_all_mwe,
                               cumulative_data_all_mwe_comparison, cumulative_errors_all_mwe_comparison):
    
    _, axs = plt.subplots(1, 2, figsize=(20,5))
    axs[0].set_xlim(2000,2024)

    axs[0].plot(cumulative_data_all_gt.dates, cumulative_data_all_gt.changes, linewidth=3, zorder=2, label=region_name + ' - ' + data_distinction)
    axs[0].fill_between(cumulative_data_all_gt.dates, cumulative_data_all_gt.changes - cumulative_errors_all_gt.errors,
                        cumulative_data_all_gt.changes + cumulative_errors_all_gt.errors, alpha=0.2)

    axs[0].plot(cumulative_data_all_gt_comparison.dates, cumulative_data_all_gt_comparison.changes, linewidth=3, zorder=2, label=region_name + ' - ' + comparison_data_distinction)
    axs[0].fill_between(cumulative_data_all_gt_comparison.dates, cumulative_data_all_gt_comparison.changes - cumulative_errors_all_gt_comparison.errors,
                        cumulative_data_all_gt_comparison.changes + cumulative_errors_all_gt_comparison.errors, alpha=0.2)

    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Cumulative Change [Gt]')
    axs[0].grid(True, alpha=0.8)

    
    axs[1].set_xlim(2000,2024)

    axs[1].plot(cumulative_data_all_mwe.dates, cumulative_data_all_mwe.changes, linewidth=3, zorder=2, label=region_name + ' - ' + data_distinction)
    axs[1].fill_between(cumulative_data_all_mwe.dates, cumulative_data_all_mwe.changes - cumulative_errors_all_mwe.errors,
                        cumulative_data_all_mwe.changes + cumulative_errors_all_mwe.errors, alpha=0.2)

    axs[1].plot(cumulative_data_all_mwe_comparison.dates, cumulative_data_all_mwe_comparison.changes, linewidth=3, zorder=2, label=region_name + ' - ' + comparison_data_distinction)
    axs[1].fill_between(cumulative_data_all_mwe_comparison.dates, cumulative_data_all_mwe_comparison.changes - cumulative_errors_all_mwe_comparison.errors,
                        cumulative_data_all_mwe_comparison.changes + cumulative_errors_all_mwe_comparison.errors, alpha=0.2)

    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Cumulative Change [m w.e.]')

    axs[0].legend(loc = 'lower left', fontsize=16)
    plt.suptitle(region_name + ': comparison RGI 6 and RGI 7: total ice loss, 2000 - 2023', fontsize=18)
    axs[1].grid(True, alpha=0.8)

    return

#define function that calculates cumulative glacier ice loss
#for both units and 3 regions and plots them
def three_runs_cumulative_comparison_plot(region_name, data_distinction_1, data_distinction_2,data_distinction_3, data_run_1, data_run_2, data_run_3):

    #calculate cumulative for normal
    cum_region_gt_1 = derivative_to_cumulative(data_run_1.start_dates, data_run_1.end_dates, data_run_1.combined_gt)
    cum_region_gt_err_1 = derivative_to_cumulative(data_run_1.start_dates, data_run_1.end_dates, data_run_1.combined_gt_errors, calculate_as_errors=True)

    cum_region_mwe_1 = derivative_to_cumulative(data_run_1.start_dates, data_run_1.end_dates, data_run_1.combined_mwe)
    cum_region_mwe_err_1 = derivative_to_cumulative(data_run_1.start_dates, data_run_1.end_dates, data_run_1.combined_mwe_errors, calculate_as_errors=True)

    #calculate cumulative for 2
    cum_region_gt_2 = derivative_to_cumulative(data_run_2.start_dates, data_run_2.end_dates, data_run_2.combined_gt)
    cum_region_gt_err_2 = derivative_to_cumulative(data_run_2.start_dates, data_run_2.end_dates, data_run_2.combined_gt_errors, calculate_as_errors=True)

    cum_region_mwe_2 = derivative_to_cumulative(data_run_2.start_dates, data_run_2.end_dates, data_run_2.combined_mwe)
    cum_region_mwe_err_2 = derivative_to_cumulative(data_run_2.start_dates, data_run_2.end_dates, data_run_2.combined_mwe_errors, calculate_as_errors=True)

    #calculate cumulative for 3
    cum_region_gt_3 = derivative_to_cumulative(data_run_3.start_dates, data_run_3.end_dates, data_run_3.combined_gt)
    cum_region_gt_err_3 = derivative_to_cumulative(data_run_3.start_dates, data_run_3.end_dates, data_run_3.combined_gt_errors, calculate_as_errors=True)

    cum_region_mwe_3 = derivative_to_cumulative(data_run_3.start_dates, data_run_3.end_dates, data_run_3.combined_mwe)
    cum_region_mwe_err_3 = derivative_to_cumulative(data_run_3.start_dates, data_run_3.end_dates, data_run_3.combined_mwe_errors, calculate_as_errors=True)

    #plot
    _, axs = plt.subplots(1, 2, figsize=(20,5))

    #plot 1
    axs[0].set_xlim(2000,2024)
    axs[0].hlines(0, 2000, 2024, linewidth=2, colors=['grey'], linestyle='dashed')

    axs[0].plot(cum_region_gt_1.dates, cum_region_gt_1.changes, linewidth=3, zorder=2, label=region_name + ' - ' + data_distinction_1)
    axs[0].fill_between(cum_region_gt_1.dates, cum_region_gt_1.changes - cum_region_gt_err_1.errors,
                        cum_region_gt_1.changes + cum_region_gt_err_1.errors, alpha=0.2)

    axs[0].plot(cum_region_gt_2.dates, cum_region_gt_2.changes, linewidth=3, zorder=2, label=region_name + ' - ' + data_distinction_2)
    axs[0].fill_between(cum_region_gt_2.dates, cum_region_gt_2.changes - cum_region_gt_err_2.errors,
                        cum_region_gt_2.changes + cum_region_gt_err_2.errors, alpha=0.2)

    axs[0].plot(cum_region_gt_3.dates, cum_region_gt_3.changes, linewidth=3, zorder=2, label=region_name + ' - ' + data_distinction_3)
    axs[0].fill_between(cum_region_gt_3.dates, cum_region_gt_3.changes - cum_region_gt_err_3.errors,
                        cum_region_gt_3.changes + cum_region_gt_err_3.errors, alpha=0.2)

    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Cumulative Change [Gt]')
    axs[0].grid(True, alpha=0.8)

    #plot 2
    axs[1].set_xlim(2000,2024)
    axs[1].hlines(0, 2000, 2024, linewidth=2, colors=['grey'], linestyle='dashed')

    axs[1].plot(cum_region_mwe_1.dates, cum_region_mwe_1.changes, linewidth=3, zorder=2, label=region_name + ' - ' + data_distinction_1)
    axs[1].fill_between(cum_region_mwe_1.dates, cum_region_mwe_1.changes - cum_region_mwe_err_1.errors,
                        cum_region_mwe_1.changes + cum_region_mwe_err_1.errors, alpha=0.2)

    axs[1].plot(cum_region_mwe_2.dates, cum_region_mwe_2.changes, linewidth=3, zorder=2, label=region_name + ' - ' + data_distinction_2)
    axs[1].fill_between(cum_region_mwe_2.dates, cum_region_mwe_2.changes - cum_region_mwe_err_2.errors,
                        cum_region_mwe_2.changes + cum_region_mwe_err_2.errors, alpha=0.2)

    axs[1].plot(cum_region_mwe_3.dates, cum_region_mwe_3.changes, linewidth=3, zorder=2, label=region_name + ' - ' + data_distinction_3)
    axs[1].fill_between(cum_region_mwe_3.dates, cum_region_mwe_3.changes - cum_region_mwe_err_3.errors,
                        cum_region_mwe_3.changes + cum_region_mwe_err_3.errors, alpha=0.2)


    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Cumulative Change [m w.e.]')

    axs[0].legend(loc = 'lower left', fontsize=16)
    plt.suptitle(region_name + ' cumulative ice loss predicted by models: ' + data_distinction_1 + ', '+ data_distinction_2 + ' and ' + data_distinction_3, fontsize=18)
    axs[1].grid(True, alpha=0.8)

    return

#define plotting function
def three_runs_relative_difference_plot(region_name, data_distinction_1, data_distinction_2, difference_data_1, difference_data_2):

    _, axs = plt.subplots(1, 2, figsize=(20,5))

    #plot 1: gt changes timeseries
    axs[0].set_xlim(2000,2024)

    axs[0].hlines(0, 2000, 2024, linewidth=2, colors=['grey'], linestyle='dashed')
    axs[0].plot(difference_data_1.start_dates + 0.5, difference_data_1.gt_difference, linewidth=3, zorder=2, label=region_name + ' - ' + data_distinction_1)
    axs[0].plot(difference_data_2.start_dates + 0.5, difference_data_2.gt_difference, linewidth=3, zorder=2, label=region_name + ' - ' + data_distinction_2)


    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Relative difference in Mass Change [% Gt]')
    axs[0].grid(True, alpha=0.8)

    ###########3
    #plot 2: mwe changes timeseries
    axs[1].set_xlim(2000,2024)

    axs[1].hlines(0, 2000, 2024, linewidth=2, colors=['grey'], linestyle='dashed')
    axs[1].plot(difference_data_1.start_dates + 0.5, difference_data_1.mwe_difference, linewidth=3, zorder=2, label=region_name + ' - ' + data_distinction_1)
    axs[1].plot(difference_data_2.start_dates + 0.5, difference_data_2.mwe_difference, linewidth=3, zorder=2, label=region_name + ' - ' + data_distinction_2)

    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Relative difference in Mass Change [% m w.e.]')

    axs[0].legend(loc = 'lower left', fontsize=16)
    plt.suptitle(region_name + ': Relative difference in ice loss between runs', fontsize=18)
    axs[1].grid(True, alpha=0.8)

    return

def three_runs_annual_cumulative_plot(region_name, data_distinction_1, data_distinction_2, data_distinction_3,data_run_1, data_run_2, data_run_3):
    # Build cumulative series (Gt) for each run
    cum_1 = derivative_to_cumulative(data_run_1.start_dates, data_run_1.end_dates, data_run_1.combined_gt)
    cum_1_err = derivative_to_cumulative(
        data_run_1.start_dates, data_run_1.end_dates, data_run_1.combined_gt_errors, calculate_as_errors=True
    )

    cum_2 = derivative_to_cumulative(data_run_2.start_dates, data_run_2.end_dates, data_run_2.combined_gt)
    cum_2_err = derivative_to_cumulative(
        data_run_2.start_dates, data_run_2.end_dates, data_run_2.combined_gt_errors, calculate_as_errors=True
    )

    cum_3 = derivative_to_cumulative(data_run_3.start_dates, data_run_3.end_dates, data_run_3.combined_gt)
    cum_3_err = derivative_to_cumulative(
        data_run_3.start_dates, data_run_3.end_dates, data_run_3.combined_gt_errors, calculate_as_errors=True
    )

    _, axs = plt.subplots(1, 2, figsize=(20, 5))

    # Left: annual change (Gt) 
    axs[0].set_xlim(2000, 2024)
    axs[0].hlines(0, 2000, 2024, linewidth=2, colors=["grey"], linestyle="dashed")

    axs[0].plot(data_run_1.start_dates + 0.5, data_run_1.combined_gt, linewidth=3, zorder=2,
                label=region_name + " - " + data_distinction_1)
    axs[0].fill_between(data_run_1.start_dates + 0.5,
                        data_run_1.combined_gt - data_run_1.combined_gt_errors,
                        data_run_1.combined_gt + data_run_1.combined_gt_errors, alpha=0.2)

    axs[0].plot(data_run_2.start_dates + 0.5, data_run_2.combined_gt, linewidth=3, zorder=2,
                label=region_name + " - " + data_distinction_2)
    axs[0].fill_between(data_run_2.start_dates + 0.5,
                        data_run_2.combined_gt - data_run_2.combined_gt_errors,
                        data_run_2.combined_gt + data_run_2.combined_gt_errors, alpha=0.2)

    axs[0].plot(data_run_3.start_dates + 0.5, data_run_3.combined_gt, linewidth=3, zorder=2,
                label=region_name + " - " + data_distinction_3)
    axs[0].fill_between(data_run_3.start_dates + 0.5,
                        data_run_3.combined_gt - data_run_3.combined_gt_errors,
                        data_run_3.combined_gt + data_run_3.combined_gt_errors, alpha=0.2)

    axs[0].set_xlabel("Year")
    axs[0].set_ylabel("Mass Change [Gt]")
    axs[0].grid(True, alpha=0.8)

    # Right: cumulative change (Gt)
    axs[1].set_xlim(2000, 2024)
    axs[1].hlines(0, 2000, 2024, linewidth=2, colors=["grey"], linestyle="dashed")

    axs[1].plot(cum_1.dates, cum_1.changes, linewidth=3, zorder=2,
                label=region_name + " - " + data_distinction_1)
    axs[1].fill_between(cum_1.dates, cum_1.changes - cum_1_err.errors, cum_1.changes + cum_1_err.errors, alpha=0.2)

    axs[1].plot(cum_2.dates, cum_2.changes, linewidth=3, zorder=2,
                label=region_name + " - " + data_distinction_2)
    axs[1].fill_between(cum_2.dates, cum_2.changes - cum_2_err.errors, cum_2.changes + cum_2_err.errors, alpha=0.2)

    axs[1].plot(cum_3.dates, cum_3.changes, linewidth=3, zorder=2,
                label=region_name + " - " + data_distinction_3)
    axs[1].fill_between(cum_3.dates, cum_3.changes - cum_3_err.errors, cum_3.changes + cum_3_err.errors, alpha=0.2)

    axs[1].set_xlabel("Year")
    axs[1].set_ylabel("Cumulative Change [Gt]")
    axs[1].grid(True, alpha=0.8)

    axs[0].legend(loc="lower left", fontsize=16)
    plt.suptitle(region_name + ": annual and cumulative comparison " + data_distinction_1 + ", " + data_distinction_2 + " and " + data_distinction_3, fontsize=18)

    return

def four_runs_annual_cumulative_plot(
    region_name,
    data_distinction_1, data_distinction_2, data_distinction_3, data_distinction_4,
    data_run_1, data_run_2, data_run_3, data_run_4):

    # Build cumulative series (Gt) for each run
    cum_1 = derivative_to_cumulative(data_run_1.start_dates, data_run_1.end_dates, data_run_1.combined_gt)
    cum_1_err = derivative_to_cumulative(
        data_run_1.start_dates, data_run_1.end_dates, data_run_1.combined_gt_errors, calculate_as_errors=True
    )

    cum_2 = derivative_to_cumulative(data_run_2.start_dates, data_run_2.end_dates, data_run_2.combined_gt)
    cum_2_err = derivative_to_cumulative(
        data_run_2.start_dates, data_run_2.end_dates, data_run_2.combined_gt_errors, calculate_as_errors=True
    )

    cum_3 = derivative_to_cumulative(data_run_3.start_dates, data_run_3.end_dates, data_run_3.combined_gt)
    cum_3_err = derivative_to_cumulative(
        data_run_3.start_dates, data_run_3.end_dates, data_run_3.combined_gt_errors, calculate_as_errors=True
    )

    cum_4 = derivative_to_cumulative(data_run_4.start_dates, data_run_4.end_dates, data_run_4.combined_gt)
    cum_4_err = derivative_to_cumulative(
        data_run_4.start_dates, data_run_4.end_dates, data_run_4.combined_gt_errors, calculate_as_errors=True
    )

    _, axs = plt.subplots(1, 2, figsize=(20, 5))

    # Left: annual change (Gt)
    axs[0].set_xlim(2000, 2024)
    axs[0].hlines(0, 2000, 2024, linewidth=2, colors=["grey"], linestyle="dashed")

    axs[0].plot(data_run_1.start_dates + 0.5, data_run_1.combined_gt, linewidth=3, zorder=2,
                label=region_name + " - " + data_distinction_1)
    axs[0].fill_between(data_run_1.start_dates + 0.5,
                        data_run_1.combined_gt - data_run_1.combined_gt_errors,
                        data_run_1.combined_gt + data_run_1.combined_gt_errors, alpha=0.2)

    axs[0].plot(data_run_2.start_dates + 0.5, data_run_2.combined_gt, linewidth=3, zorder=2,
                label=region_name + " - " + data_distinction_2)
    axs[0].fill_between(data_run_2.start_dates + 0.5,
                        data_run_2.combined_gt - data_run_2.combined_gt_errors,
                        data_run_2.combined_gt + data_run_2.combined_gt_errors, alpha=0.2)

    axs[0].plot(data_run_3.start_dates + 0.5, data_run_3.combined_gt, linewidth=3, zorder=2,
                label=region_name + " - " + data_distinction_3)
    axs[0].fill_between(data_run_3.start_dates + 0.5,
                        data_run_3.combined_gt - data_run_3.combined_gt_errors,
                        data_run_3.combined_gt + data_run_3.combined_gt_errors, alpha=0.2)

    axs[0].plot(data_run_4.start_dates + 0.5, data_run_4.combined_gt, linewidth=3, zorder=2,
                label=region_name + " - " + data_distinction_4)
    axs[0].fill_between(data_run_4.start_dates + 0.5,
                        data_run_4.combined_gt - data_run_4.combined_gt_errors,
                        data_run_4.combined_gt + data_run_4.combined_gt_errors, alpha=0.2)

    axs[0].set_xlabel("Year")
    axs[0].set_ylabel("Mass Change [Gt]")
    axs[0].grid(True, alpha=0.8)

    # Right: cumulative change (Gt)
    axs[1].set_xlim(2000, 2024)
    axs[1].hlines(0, 2000, 2024, linewidth=2, colors=["grey"], linestyle="dashed")

    axs[1].plot(cum_1.dates, cum_1.changes, linewidth=3, zorder=2,
                label=region_name + " - " + data_distinction_1)
    axs[1].fill_between(cum_1.dates, cum_1.changes - cum_1_err.errors, cum_1.changes + cum_1_err.errors, alpha=0.2)

    axs[1].plot(cum_2.dates, cum_2.changes, linewidth=3, zorder=2,
                label=region_name + " - " + data_distinction_2)
    axs[1].fill_between(cum_2.dates, cum_2.changes - cum_2_err.errors, cum_2.changes + cum_2_err.errors, alpha=0.2)

    axs[1].plot(cum_3.dates, cum_3.changes, linewidth=3, zorder=2,
                label=region_name + " - " + data_distinction_3)
    axs[1].fill_between(cum_3.dates, cum_3.changes - cum_3_err.errors, cum_3.changes + cum_3_err.errors, alpha=0.2)

    axs[1].plot(cum_4.dates, cum_4.changes, linewidth=3, zorder=2,
                label=region_name + " - " + data_distinction_4)
    axs[1].fill_between(cum_4.dates, cum_4.changes - cum_4_err.errors, cum_4.changes + cum_4_err.errors, alpha=0.2)

    axs[1].set_xlabel("Year")
    axs[1].set_ylabel("Cumulative Change [Gt]")
    axs[1].grid(True, alpha=0.8)

    axs[0].legend(loc="lower left", fontsize=16)
    plt.suptitle(
        region_name + ": annual and cumulative comparison " 
        # + data_distinction_1 + ", " + data_distinction_2 + ", " +
        # data_distinction_3 + " and " + data_distinction_4,
        # fontsize=18
    )
    return


def derivative_to_cumulative(start_dates, end_dates, changes, calculate_as_errors: bool = False):

    contains_no_gaps = [start_date == end_date for start_date, end_date in zip(start_dates[1:], end_dates[:-1])]
    # add an extra row to dataset for each gap, so that it's represented in the cumulative timeseries as no data
    if calculate_as_errors:
        changes = [0, *np.array(pd.Series(np.square(changes)).cumsum())**0.5]
    else:
        changes = [0, *np.array(pd.Series(changes).cumsum())]

    if not all(contains_no_gaps):
        indices_of_gaps = [i for i, x in enumerate(contains_no_gaps) if not (x)]
        start_dates = list(start_dates.copy())
        end_dates = list(end_dates.copy())
        for idx in indices_of_gaps:
            start_date_to_insert = end_dates[idx]
            end_date_to_insert = start_dates[idx + 1]
            # add no data row
            start_dates.insert(idx + 1, start_date_to_insert)
            end_dates.insert(idx + 1, end_date_to_insert)
            changes.insert(idx + 2, None)  # already in cumulative, hence +2
            # add last row before gap again after gap
            start_dates.insert(idx + 2, start_dates[idx + 1])
            end_dates.insert(idx + 2, end_dates[idx + 1])
            changes.insert(idx + 3, changes[idx + 1])  # already in cumulative, hence +3
    dates = [start_dates[0], *end_dates]

    if calculate_as_errors:
        cumulative_data = pd.DataFrame({'dates': dates, 'errors': changes})
    else:
        cumulative_data = pd.DataFrame({'dates': dates, 'changes': changes})
      
    return cumulative_data
