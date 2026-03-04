import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.colors as colors
import matplotlib.cm as cmx

def two_region_comparison_plot(region_name, data_distinction, comparison_data_distinction, cumulative_data_all_gt, cumulative_errors_all_gt, cumulative_data_all_gt_comparison,
                               cumulative_errors_all_gt_comparison, cumulative_data_all_mwe, cumulative_errors_all_mwe,
                               cumulative_data_all_mwe_comparison, cumulative_errors_all_mwe_comparison):
    
    _, axs = plt.subplots(1, 2, figsize=(20,8))
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
    axs[1].set_ylabel('Cumulative Change [metres water equivalent]')

    axs[0].legend(loc = 'lower left', fontsize=16)
    plt.suptitle(region_name + ': comparison RGI 6 and RGI 7: total ice loss, 2000 - 2023', fontsize=18)
    axs[1].grid(True, alpha=0.8)


    return
