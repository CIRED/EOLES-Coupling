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
    scenarios = pd.read_csv(Path('eoles') / Path('inputs') / Path('xps') / Path('20240220') / Path('scenarios.csv'),
                            index_col=0)

    dict_output = {}
    # list all files in a folder with path folderpath
    folderpath = Path('eoles') / Path('outputs') / Path('20240220')
    for path in folderpath.iterdir():
        if path.is_dir():
            dict_output[path.name.split('_')[1]] = path

    # Process outputs
    output = get_total_system_costs(dict_output)
    output = pd.concat([output[k] for k in output.keys()], axis=0)

    scenarios_complete = pd.concat([scenarios, output.T], axis=1)
    scenarios_complete.to_csv(Path('eoles') / Path('outputs') / Path('20240220') / Path('scenarios_complete.csv'))

    # Plots
    sns.boxplot(data=scenarios_complete, x='learning', y='Total costs', hue='biogas')
    plt.show()

    L = ['learning', 'elasticity']  # Add your variable names here

    # Define the number of rows and columns for subplots based on the length of L
    n = len(L)
    ncols = 2  # Define number of columns per row
    nrows = n // ncols + (n % ncols > 0)  # Calculate required number of rows

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 5))
    fig.tight_layout(pad=5.0)

    # Flatten axes array if more than one row
    if nrows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Iterate over variables and create a boxplot for each
    for i, variable in enumerate(L):
        sns.boxplot(data=scenarios_complete, x=variable, y='Total costs', hue='biogas', ax=axes[i])
        axes[i].set_title(f'Boxplot of Total costs by {variable} and biogas')
        axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability

    # Hide any unused subplots if L does not fill up the entire grid
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    plt.show()