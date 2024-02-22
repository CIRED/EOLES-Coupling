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

    # Load data
    scenarios = pd.read_csv(Path('eoles') / Path('inputs') / Path('xps') / Path('20240221_2') / Path('scenarios.csv'),
                            index_col=0)

    dict_output = {}
    # list all files in a folder with path folderpath
    folderpath = Path('eoles') / Path('outputs') / Path('20240221')
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
    scenarios_complete.to_csv(Path('eoles') / Path('outputs') / Path('20240221') / Path('scenarios_complete.csv'))

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