import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pickle import load
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import datetime

from eoles.write_output import get_total_system_costs, comparison_simulations_new, plot_typical_week, plot_typical_demand, plot_residual_demand, colormap_simulations
from pathlib import Path
from project.write_output import plot_compare_scenarios


if __name__ == '__main__':

    folderpath = Path('eoles/outputs/marginal_20240222_154330')

    # Load data
    scenarios = pd.read_csv(folderpath / Path('scenarios.csv'), index_col=0)

    dict_output = {}
    # list all files in a folder with path folderpath

    for path in folderpath.iterdir():
        if path.is_dir():
            dict_output[path.name.split('_')[1]] = path

    # Process outputs
    output = get_total_system_costs(dict_output)
    output['passed'] = output['passed'].to_frame().T.rename(index={0: 'passed'})
    output = pd.concat([output[k] for k in output.keys()], axis=0)
    new_index = ['passed', 'Total costs']
    # create a new index, with a reordering of output, with new_index as the first two index, and the rest following in the same order
    new_index.extend([k for k in output.index if k not in new_index])
    output = output.reindex(index=new_index)

    scenarios_complete = pd.concat([scenarios, output.T], axis=1)
    scenarios_complete.to_csv(folderpath / Path('scenarios_complete.csv'))

    scenarios_complete['ban'] = scenarios_complete['ban'].fillna('No Ban')
    multi_index = pd.MultiIndex.from_arrays([scenarios_complete.index.to_series().str.replace('-ban', '', regex=False), scenarios_complete['ban']], names=('Scenario', 'Ban_Status'))
    scenarios_complete.index = multi_index
    scenarios_complete = scenarios_complete.drop(columns='ban')
    scenarios_complete = scenarios_complete.sort_index()
    # scenarios_complete['Total costs'] = np.random.randn(len(scenarios_complete))

    # groupby level 'Scenario' from index, and do the difference of the column 'Total costs' for the two lines involved in each groupby.
    scenarios_complete['Total costs'] = scenarios_complete['Total costs'].groupby(level='Scenario').diff()
    # drop if level 'Ban_Status' is 'Ban'
    difference_costs = scenarios_complete[scenarios_complete.index.get_level_values('Ban_Status') != 'Ban']['Total costs'].droplevel('Ban_Status')

    # # Plots
    # sns.boxplot(data=scenarios_complete, x='learning', y='Total costs', hue='biogas')
    # plt.show()
    #
    # variables = ['learning', 'elasticity', 'capacity']
    # # Create a figure and a set of subplots
    # fig, axes = plt.subplots(1, len(variables), figsize=(15, 5), sharey=True)
    # # Loop through the variables and create a boxplot for each
    # for i, variable in enumerate(variables):
    #     sns.boxplot(data=scenarios_complete, x=variable, y='Total costs', hue='biogas', ax=axes[i])
    #     axes[i].set_title(variable)  # Set the title to the variable name
    #     if i > 0:  # Only show the legend for the first plot to avoid repetition
    #         axes[i].get_legend().remove()
    # plt.tight_layout()  # Adjust the layout to make sure everything fits without overlapping
    # plt.show()