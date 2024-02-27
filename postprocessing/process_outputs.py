import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pickle import load
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import datetime
import sys

sys.path.append("../..")
from eoles.write_output import get_main_outputs, comparison_simulations_new, plot_typical_week, plot_typical_demand, plot_residual_demand, colormap_simulations
from pathlib import Path
from project.write_output import plot_compare_scenarios

from SALib.sample import saltelli
from SALib.analyze import sobol


def parse_outputs(folderpath):

    # Load scenarios names
    scenarios = pd.read_csv(folderpath / Path('scenarios.csv'), index_col=0)

    dict_output = {}
    # list all files in a folder with path folderpath
    for path in folderpath.iterdir():
        if path.is_dir():
            dict_output[path.name.split('_')[1]] = path

    # Process outputs
    output = get_main_outputs(dict_output)
    output['passed'] = output['passed'].to_frame().T.rename(index={0: 'passed'})
    output = pd.concat([output[k] for k in output.keys()], axis=0)
    new_index = ['passed', 'Total costs']
    # create a new index, with a reordering of output, with new_index as the first two index, and the rest following in the same order
    new_index.extend([k for k in output.index if k not in new_index])
    output = output.reindex(index=new_index)

    scenarios_complete = pd.concat([scenarios, output.T], axis=1)
    scenarios_complete.to_csv(folderpath / Path('scenarios_complete.csv'))

    difference_costs = scenarios_complete.copy()
    difference_costs['ban'] = difference_costs['ban'].fillna('No Ban')
    multi_index = pd.MultiIndex.from_arrays([difference_costs.index.to_series().str.replace('-ban', '', regex=False), difference_costs['ban']], names=('Scenario', 'Ban_Status'))
    difference_costs.index = multi_index
    difference_costs = difference_costs.drop(columns='ban')
    difference_costs = difference_costs.sort_index()
    # difference_costs['Total costs'] = np.random.randn(len(difference_costs))

    # groupby level 'Scenario' from index, and do the difference of the column 'Total costs' for the two lines involved in each groupby.
    difference_costs['Total costs'] = difference_costs['Total costs'].groupby(level='Scenario').diff()

    description_scenarios = difference_costs[scenarios.columns.drop('ban')]
    description_scenarios = description_scenarios[description_scenarios.index.get_level_values('Ban_Status') != 'Ban'].droplevel('Ban_Status')

    difference_costs =  - difference_costs[difference_costs.index.get_level_values('Ban_Status') != 'Ban']['Total costs'].droplevel('Ban_Status')

    difference_costs = pd.concat([description_scenarios, difference_costs], axis=1)

    return difference_costs, scenarios_complete


def salib_analysis(scenarios, list_features, y, num_samples=500):

    # transform categorical variables into numerical
    mapping_categorical = {}
    for col in list_features:
        unique_values = scenarios[col].unique()
        mapping = {unique_values[i]: i for i in range(len(unique_values))}
        mapping_categorical[col] = mapping
        scenarios[col] = scenarios[col].replace(mapping)

    # Create a SALib configuration
    bounds = [[0, max(mapping_categorical[col].values()) + 1] for col in list_features]
    problem = {
        'num_vars': len(list_features),
        'names': list_features,
        'bounds': bounds
    }

    # Generate samples (these will be continuous values within the bounds)
    param_values = saltelli.sample(problem, num_samples)

    param_values_categorical = np.floor(param_values).astype(int)
    param_values_categorical = pd.DataFrame(param_values_categorical, columns=list_features)

    # Sobol analysis with SALib
    df_tot = pd.merge(param_values_categorical, scenarios, how='left', on=list_features)
    Y = df_tot[y].values
    Si = sobol.analyze(problem, Y)
    first_order = pd.Series(Si['S1'])
    first_order.index = list_features
    total_order = pd.Series(Si['ST'])
    total_order.index = list_features

    second_order = pd.DataFrame(Si['S2'], index=list_features, columns=list_features    )

    sobol_salib_df = pd.DataFrame({'first_order': first_order, 'total_order': total_order})
    return sobol_salib_df, second_order


def manual_sobol_analysis(scenarios, list_features, y):
    """Computes manually the Sobol indices for a given set of scenarios and a given output variable y"""
    sobol_df = pd.DataFrame(index=list_features, columns=['first_order', 'total_order'])

    expectation, variance = scenarios[y].mean(), scenarios[y].var()

    for col in list_features:
        # first order
        conditional_means = scenarios.groupby(col)[y].mean()
        counts = scenarios.groupby(col).size() / len(scenarios)
        sobol_first_order = (counts * (conditional_means - expectation) ** 2).sum() / variance
        sobol_df.loc[col, 'first_order'] = sobol_first_order

        # total order
        list_features_minus_i = list_features.copy()
        list_features_minus_i.remove(col)
        conditional_means = scenarios.groupby(list_features_minus_i)[y].mean()
        counts = scenarios.groupby(list_features_minus_i).size() / len(scenarios)
        sobol_total_order = 1 - (counts * (conditional_means - expectation) ** 2).sum() / variance
        sobol_df.loc[col, 'total_order'] = sobol_total_order
    return sobol_df


def analysis_costs_regret(scenarios, list_features):
    """Calculates the difference of costs between Reference and Ban scenario."""
    ind = scenarios.groupby('Scenario')['passed'].sum()[scenarios.groupby('Scenario')['passed'].sum() == 2].index
    # select subset of scenarios_complete where first level of index is in ind
    tmp = scenarios[scenarios.index.get_level_values('Scenario').isin(ind)]

    tmp_costs = tmp.sort_index().groupby('Scenario')['Total costs'].diff()
    tmp_costs = -tmp_costs[tmp_costs.index.get_level_values('Ban_Status') != 'Ban'].droplevel('Ban_Status')
    tmp_costs = pd.concat([tmp[tmp.index.get_level_values('Ban_Status') != 'Ban'].droplevel('Ban_Status')[list_features], tmp_costs], axis=1)
    return tmp_costs

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

if __name__ == '__main__':
    folderpath = Path('simulations/exhaustive_20240223_184702')
    difference_costs, scenarios_complete = parse_outputs(folderpath)