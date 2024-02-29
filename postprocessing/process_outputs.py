import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pickle import load
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import datetime
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()

# Ensure the project root is in sys.path
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

sys.path.append("../..")
from eoles.write_output import get_main_outputs, comparison_simulations_new, plot_typical_week, plot_typical_demand, plot_residual_demand, colormap_simulations
from eoles.write_output import waterfall_chart, DICT_LEGEND_WATERFALL
from eoles.inputs.resources import resources_data

from project.write_output import plot_compare_scenarios

from SALib.sample import saltelli
from SALib.analyze import sobol

MAPPING = {'Learning+': 'High', 'Learning-': 'Low',
           'Elasticity+': 'High', 'Elasticity-': 'Low',
           'Biogas+': 'High', 'Biogas-': 'Low',
           'Capacity_ren+': 'High', 'Ren-': 'Low',
           'Demand+': 'High', 'Sufficiency': 'Low',
           'PriceGas+': 'High', 'PriceGas-': 'Low',
           'PriceWood+': 'High', 'PriceWood-': 'Low',
           'Policy_mix+': 'High', 'Policy_mix-': 'Low',
           'NoPolicyInsulation': 'Low',
          'NoPolicyHeater': 'Low',
           'CarbonBudget-': 'Low',
           'COP+': 'High',
           'reference': 'Reference'}

NAME_COLUMNS = {
    'policy_mix': 'Policy mix',
    'learning': 'Technical progress heat-pumps',
    'elasticity': 'Heat-pump price elasticity',
    'biogas': 'Biogas potential',
    'capacity_ren': 'Renewable capacity',
    'demand': 'Other electricity demand',
    'gasprices': 'Gas prices',
    'woodprices': 'Wood prices',
    'cop': 'COP heat pump',
    'policy_heater': 'Heater policy',
    'policy_insulation': 'Insulation policy',
    'carbon_budget': 'Carbon budget'

}


def parse_outputs(folderpath, features):
    """Parses the outputs of the simulations and creates a csv file with the results.
    Return scenarios_complete, and output which has been processed to only include information on the difference between the Ban and the reference scenario."""

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

    # transform how ban is handled in the dataframe
    multi_index = pd.MultiIndex.from_arrays([scenarios_complete.index.to_series().str.replace('-ban', '', regex=False), scenarios_complete['ban']],names=('Scenario', 'Ban_Status'))
    scenarios_complete.index = multi_index
    scenarios_complete = scenarios_complete.drop(columns='ban')
    scenarios_complete = scenarios_complete.sort_index()  # sort so that Ban and reference are always displayed in the same direction

    scenarios_complete.to_csv(folderpath / Path('scenarios_complete.csv'))

    def determine_value(group):
        # Extract 'passed' values for 'reference' and 'ban' within the group
        passed_reference = group.xs('reference', level='Ban_Status')['passed'].iloc[0]
        passed_ban = group.xs('Ban', level='Ban_Status')['passed'].iloc[0]

        # Determine the value based on the conditions
        if passed_reference == 1 and passed_ban == 0:
            return 1
        elif passed_reference == 0 and passed_ban == 1:
            return -1
        else:
            return 0

    output = scenarios_complete.groupby(level='Scenario').apply(determine_value)  # create output of interest, comparing reference and ban
    output = output.to_frame(name='passed')
    description_scenarios = scenarios_complete[features]
    output = pd.concat([description_scenarios[description_scenarios.index.get_level_values('Ban_Status') != 'Ban'].droplevel('Ban_Status'), output], axis=1)
    output.to_csv(folderpath / Path('scenarios_comparison.csv'))


    return scenarios_complete, output


def waterfall_analysis(scenarios_complete, reference='S0', save_path=None, wood=True):
    """Plots the waterfall chart to compare reference with Ban scenario."""
    list_costs = ['Investment heater costs', 'Investment insulation costs', 'Investment electricity costs','Functionment costs', 'Total costs']
    scenarios_complete = scenarios_complete.sort_index()  # order for estimating the difference
    costs_diff = - scenarios_complete.xs(reference, level='Scenario')[list_costs].diff()
    costs_diff = costs_diff.xs('reference')
    costs_diff['Total costs'] = costs_diff['Total costs'] * 25  # we had divided total costs by 25 to have the value per year, so here we need to multiply again
    if save_path is not None:
        save_path_costs = save_path / Path('waterfall_costs.png')
    else:
        save_path_costs = None
    waterfall_chart(costs_diff, colors=resources_data["colors_eoles"], rotation=0, save= save_path_costs, format_y=lambda y, _: '{:.0f} B€'.format(y),
                    title="Difference in total system costs", y_label=None, hline=True, dict_legend=DICT_LEGEND_WATERFALL)

    list_capacity = ['offshore', 'onshore', 'pv', 'battery', 'hydro', 'peaking plants', 'methanization', 'pyrogazification']
    capacity_diff = - scenarios_complete.xs(reference, level='Scenario')[list_capacity].diff()
    # drop values equal to 0
    capacity_diff = capacity_diff.xs('reference')
    capacity_diff = capacity_diff[abs(capacity_diff) > 0.1]
    if save_path is not None:
        save_path_capacity = save_path / Path('waterfall_capacity.png')
    else:
        save_path_capacity = None
    waterfall_chart(capacity_diff, colors=resources_data["new_colors_eoles"], rotation=0, save=save_path_capacity, format_y=lambda y, _: '{:.0f} GW'.format(y),
                    title="Difference in capacity installed (GW)", y_label=None, hline=True, total=False, unit='GW', float_precision=1, neg_offset=1.34, pos_offset=0.53)

    list_generation = ['Generation offshore (TWh)', 'Generation onshore (TWh)', 'Generation pv (TWh)',
                       'Generation hydro (TWh)', 'Generation battery (TWh)', 'Generation nuclear (TWh)', 'Generation natural gas (TWh)',
                       'Generation peaking plants (TWh)', 'Generation methanization (TWh)', 'Generation pyrogazification (TWh)', 'Consumption Oil (TWh)']
    if wood:  # we only include direct wood consumption
         list_generation = list_generation + ['Consumption Wood (TWh)']
    else:
        list_generation = list_generation + ['Consumption Wood (TWh)', 'Generation central wood boiler (TWh)']
    generation_diff = scenarios_complete.xs(reference, level='Scenario')[list_generation]
    if not wood:
        generation_diff['Consumption Wood (TWh)'] = generation_diff['Consumption Wood (TWh)'] + generation_diff['Generation central wood boiler (TWh)']  # we sum overall consumption
        generation_diff = generation_diff.drop(columns='Generation central wood boiler (TWh)')
    generation_diff = - generation_diff.diff()
    generation_diff = generation_diff.xs('reference')
    generation_diff = generation_diff[abs(generation_diff) > 0.1]
    if save_path is not None:
        save_path_generation = save_path / Path('waterfall_generation.png')
    else:
        save_path_generation = None
    waterfall_chart(generation_diff, colors=resources_data["new_colors_eoles"], rotation=0, save=save_path_generation, format_y=lambda y, _: '{:.0f} TWh'.format(y),
                    title="Difference in generation (TWh)", y_label=None, hline=True, total=False, unit='TWh', float_precision=1, dict_legend=DICT_LEGEND_WATERFALL, neg_offset=3, pos_offset=0.53)

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
    tmp_costs = - tmp_costs[tmp_costs.index.get_level_values('Ban_Status') != 'Ban'].droplevel('Ban_Status')
    tmp_costs = pd.concat([tmp[tmp.index.get_level_values('Ban_Status') != 'Ban'].droplevel('Ban_Status')[list_features], tmp_costs], axis=1)
    return tmp_costs


def create_frequency_dict(df):

    # Initialize an empty dictionary to store the frequency counts
    frequency_dict = {}

    # Total number of rows
    total = df.shape[0]
    # Iterate through each column in the DataFrame
    for column in df.columns:
        # Get the frequency of each value in the current column
        counts = df[column].value_counts()

        # For each value ('Low', 'Medium', 'High'), get the count, if not present, default to 0
        low_count = counts.get('Low', 0) / total
        medium_count = counts.get('Reference', 0) / total
        high_count = counts.get('High', 0) / total

        # Assign the counts to the corresponding column in the dictionary
        frequency_dict[column] = [low_count, medium_count, high_count]

    return frequency_dict


def make_frequency_chart(df, save_path=None):

    df = df.replace(MAPPING)
    df = create_frequency_dict(df)
    df = {NAME_COLUMNS[key]: value for key, value in df.items()}

    frequency_chart(df, save_path=save_path)


def frequency_chart(results, category_names=None, category_colors=None, save_path=None):
    """
    Parameters
    ----------
    results : dict
        A mapping from category labels to a list of variables per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    if category_names is None:
        category_names = ['Low', 'Reference', 'High']
        category_colors = ['#f6511d', '#ffb400', '#00a6ed']

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)

    if category_colors is None:
        category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())


    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts,
                        label=colname, color=color)

        text_color = 'white' if int(color[1:], 16) < 0x888888 else 'black'
        l = ['{:.0f}%'.format(val * 100) if val != 0 else '' for val in widths]

        ax.bar_label(rects, label_type='center', color=text_color, labels=l, fontsize='small')
    ax.legend(bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small', frameon=False, ncol=3)

    # remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig, ax


def frequency_chart_subplot(results1, results2, category_names=None, save_path=None, axis_titles=('Axis 1', 'Axis 2')):
    """
    Display survey results in two side-by-side stacked bar charts.

    Parameters
    ----------
    results1, results2 : dict
        Mappings from question labels to a list of answers per category for each of the two datasets.
    category_names : list of str, optional
        The category labels.
    save_path : str, optional
        Path where to save the figure.
    axis_titles : tuple of str, optional
        Titles for the left and right axes.
    """
    if category_names is None:
        category_names = ['Low', 'Reference', 'High']

    category_colors = ['#f6511d', '#ffb400', '#00a6ed']
    fig, axs = plt.subplots(1, 2, figsize=(18, 5), sharey=True)  # Share Y axis

    for ax_idx, (ax, results, axis_title) in enumerate(zip(axs, [results1, results2], axis_titles)):
        question_labels = list(results.keys())
        data = np.array(list(results.values()))
        data_cum = data.cumsum(axis=1)

        ax.invert_yaxis()
        if ax_idx == 0:  # Apply y-ticks only for the left subplot
            ax.set_yticks(range(len(question_labels)))
            ax.set_yticklabels(question_labels)
        ax.set_xlim(0, np.sum(data, axis=1).max())
        ax.set_title(axis_title, fontsize='small')

        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects = ax.barh(range(len(question_labels)), widths, left=starts, height=0.55, color=color)

            text_color = 'white' if int(color[1:], 16) < 0x888888 else 'black'
            bar_labels = ['{:.0f}%'.format(val * 100) if val != 0 else '' for val in widths]
            ax.bar_label(rects, labels=bar_labels, label_type='center', color=text_color, fontsize='small')

        # remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.xaxis.set_visible(False)


    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in zip(category_names, category_colors)]
    fig.legend(handles=legend_handles, bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)


    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    folderpath = Path('/mnt/beegfs/workdir/celia.escribe/eoles2/eoles/outputs/exhaustive_20240226_202408')  # for cluster use
    features = ['policy_heater', 'policy_insulation', 'learning', 'elasticity', 'cop', 'biogas', 'capacity_ren',
                'demand', 'carbon_budget', 'gasprices']
    scenarios_complete, output = parse_outputs(folderpath, features=features)