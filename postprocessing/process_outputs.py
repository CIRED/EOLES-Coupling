import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import product
import ast
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

from matplotlib.ticker import PercentFormatter

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

ORDER_COLUMNS = ['policy_mix', 'policy_heater', 'policy_insulation', 'learning', 'elasticity',
                 'cop', 'biogas', 'capacity_ren', 'demand', 'carbon_budget', 'gasprices']

LIST_FEATURES = ['policy_heater', 'policy_insulation', 'learning', 'elasticity', 'cop', 'biogas', 'capacity_ren',
            'demand', 'carbon_budget', 'gasprices']


def parse_outputs(folderpath, features, emissions=False):
    """Parses the outputs of the simulations and creates a csv file with the results.
    Return scenarios_complete, and output which has been processed to only include information on the difference between the Ban and the reference scenario."""

    # Load scenarios names
    scenarios = pd.read_csv(folderpath / Path('scenarios.csv'), index_col=0)
    scenarios = scenarios.fillna('reference')  # when using marginal method, some scenarios are NaNs, and need to be replaced with reference
    dict_output = {}
    # list all files in a folder with path folderpath
    for path in folderpath.iterdir():
        if path.is_dir():
            dict_output[path.name.split('_')[1]] = path

    # Process outputs
    output, hourly_generation = get_main_outputs(dict_output, emissions=emissions)
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
    # create a dataframe from hourly_generation, which includes a dataframe for each of its keys. Each dataframe has the same index and columns for each key.
    # you should take that into account when concatenating

    hourly_generation = pd.concat(hourly_generation.values(), axis=1, keys=hourly_generation.keys())
    hourly_generation.columns.names = ['Scenario', 'Hourly Data']
    hourly_generation.to_csv(folderpath / Path('hourly_generation.csv'))

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


    return scenarios_complete, output, hourly_generation


def make_table_summary(data):
    output_dict = {'Stock Heat pump (Million)': 'Number of heat pumps',
                   'Stock Direct electric (Million)': 'Number of direct electric',
                   'Stock Natural gas (Million)': 'Number of gas boilers',
                   'Stock Wood fuel (Million)': 'Number of wood boilers',
                   'Subsidies insulation (Billion euro)': 'Subsidies insulation',
                   'Subsidies heater (Billion euro)': 'Subsidies heater',
                   'Investment heater costs': 'Investment heating system',
                   'Investment insulation costs': 'Investment insulation',
                   'Consumption Electricity (TWh)': 'Consumption Electricity',
                   'Consumption Natural gas (TWh)': 'Consumption Gas',
                   'Consumption Wood fuel (TWh)': 'Consumption Wood',
                   'offshore': 'Offshore capacity',
                   'onshore': 'Onshore capacity',
                   'pv': 'Solar PV',
                   'battery': 'Battery',
                   'peaking plants': 'Peaking plants capacity',
                   'methanization': 'Methanization capacity',
                   'pyrogazification': 'Pyrogazification capacity',
                   'hydro': 'Hydroelectricity capacity',
                   'Generation offshore (TWh)': 'Offshore production',
                   'Generation onshore (TWh)': 'Onshore production',
                   'Generation pv (TWh)': 'Solar PV production',
                   'Generation battery (TWh)': 'Battery production',
                   'Generation hydro (TWh)': 'Hydroelectricity production',
                   'Generation peaking plants (TWh)': 'Peaking plants production',
                   'Generation nuclear (TWh)': 'Nuclear production',
                   'Generation methanization (TWh)': 'Methanization production',
                   'Generation pyrogazification (TWh)': 'Pyrogazification production'
                   }

    scenarios = {('S0', 'reference'): 'Counterfactual', ('S0', 'Ban'): 'Ban'}
    df = data.loc[scenarios.keys(), output_dict.keys()].T
    df.rename(index=output_dict, inplace=True)
    df.columns = scenarios.values()
    df = df.astype('int')
    return df


def get_distributional_data(df):
    """Extracts detailed distributional data, using the knowledge that those distributional data are tuples."""

    # Update column names to correctly display tuples
    new_column_names = []
    for name in df.columns:
        try:
            # Try to evaluate the name as a tuple
            evaluated_name = ast.literal_eval(name)
            if isinstance(evaluated_name, tuple):
                new_column_names.append(evaluated_name)
            else:
                new_column_names.append(name)
        except:
            # If it's not a tuple, just use the name as is
            new_column_names.append(name)

    df.columns = new_column_names

    tmp = df[df.columns[df.columns.to_series().apply(lambda x: isinstance(x, tuple))]]
    tmp.columns = pd.MultiIndex.from_tuples(tmp.columns)
    return tmp

def waterfall_analysis(scenarios_complete, reference='S0', save_path=None, wood=True, neg_offset_dict=None, pos_offset_dict=None):
    """Plots the waterfall chart to compare reference with Ban scenario."""
    list_costs = ['Investment heater costs', 'Investment insulation costs', 'Investment electricity costs','Functionment costs', 'Total costs']
    if len(scenarios_complete) == 2:
        costs_diff = scenarios_complete[list_costs].diff()
        costs_diff = costs_diff.iloc[-1]
    else:
        scenarios_complete = scenarios_complete.sort_index()  # order for estimating the difference
        costs_diff = - scenarios_complete.xs(reference, level='Scenario')[list_costs].diff()
        costs_diff = costs_diff.xs('reference')
    costs_diff['Total costs'] = costs_diff['Total costs'] * 25  # we had divided total costs by 25 to have the value per year, so here we need to multiply again
    if save_path is not None:
        save_path_costs = save_path / Path('waterfall_costs.pdf')
    else:
        save_path_costs = None

    neg_offset, pos_offset = None, None
    if (neg_offset_dict is not None) and ('costs' in neg_offset_dict.keys()):
        neg_offset = neg_offset_dict['costs']
    if (pos_offset_dict is not None) and ('costs' in pos_offset_dict.keys()):
        pos_offset = pos_offset_dict['costs']
    waterfall_chart(costs_diff, colors=resources_data["colors_eoles"], rotation=0, save= save_path_costs, format_y=lambda y, _: '{:.0f} B€'.format(y),
                    title="Additional system costs when Ban is implemented (B€)", y_label=None, hline=True, dict_legend=DICT_LEGEND_WATERFALL,
                    df_max=None, df_min=None, neg_offset=neg_offset, pos_offset=pos_offset)

    list_capacity = ['offshore', 'onshore', 'pv', 'battery', 'hydro', 'peaking plants', 'methanization', 'pyrogazification']
    if len(scenarios_complete) == 2:
        capacity_diff = scenarios_complete[list_capacity].diff()
        capacity_diff = capacity_diff.iloc[-1]
    else:
        capacity_diff = - scenarios_complete.xs(reference, level='Scenario')[list_capacity].diff()
        capacity_diff = capacity_diff.xs('reference')
    # drop values equal to 0
    capacity_diff = capacity_diff[abs(capacity_diff) > 0.1]
    if save_path is not None:
        save_path_capacity = save_path / Path('waterfall_capacity.pdf')
    else:
        save_path_capacity = None
    waterfall_chart(capacity_diff, colors=resources_data["new_colors_eoles"], rotation=0, save=save_path_capacity, format_y=lambda y, _: '{:.0f} GW'.format(y),
                    title="Additional capacity installed when Ban is implemented (GW)", y_label=None, hline=True, total=False, unit='GW', float_precision=1, dict_legend=DICT_LEGEND_WATERFALL, neg_offset=1.5, pos_offset=0.53)

    list_generation = ['Generation offshore (TWh)', 'Generation onshore (TWh)', 'Generation pv (TWh)',
                       'Generation hydro (TWh)', 'Generation battery (TWh)', 'Generation nuclear (TWh)', 'Generation natural gas (TWh)',
                       'Generation peaking plants (TWh)', 'Generation methanization (TWh)', 'Generation pyrogazification (TWh)', 'Consumption Oil (TWh)']
    if wood:  # we only include direct wood consumption
         list_generation = list_generation + ['Consumption Wood (TWh)']
    else:
        list_generation = list_generation + ['Consumption Wood (TWh)', 'Generation central wood boiler (TWh)']

    if len(scenarios_complete) == 2:
        generation_diff = scenarios_complete[list_generation]
    else:
        generation_diff = scenarios_complete.xs(reference, level='Scenario')[list_generation]
    if not wood:
        generation_diff['Consumption Wood (TWh)'] = generation_diff['Consumption Wood (TWh)'] + generation_diff['Generation central wood boiler (TWh)']  # we sum overall consumption
        generation_diff = generation_diff.drop(columns='Generation central wood boiler (TWh)')

    if len(scenarios_complete) == 2:
        generation_diff = generation_diff.diff()
        generation_diff = generation_diff.iloc[-1]
    else:
        generation_diff = - generation_diff.diff()
        generation_diff = generation_diff.xs('reference')
    generation_diff = generation_diff[abs(generation_diff) > 0.1]
    if save_path is not None:
        save_path_generation = save_path / Path('waterfall_generation.pdf')
    else:
        save_path_generation = None
    waterfall_chart(generation_diff, colors=resources_data["new_colors_eoles"], rotation=0, save=save_path_generation, format_y=lambda y, _: '{:.0f} TWh'.format(y),
                    title="Additional generation when Ban is implemented (TWh)", y_label=None, hline=True, total=False, unit='TWh',
                    float_precision=1, dict_legend=DICT_LEGEND_WATERFALL, neg_offset=3, pos_offset=0.53)


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
    sobol_df = pd.DataFrame(index=list_features, columns=['First order', 'Total order'])

    expectation, variance = scenarios[y].mean(), scenarios[y].var()

    for col in list_features:
        # first order
        conditional_means = scenarios.groupby(col)[y].mean()
        counts = scenarios.groupby(col).size() / len(scenarios)
        sobol_first_order = (counts * (conditional_means - expectation) ** 2).sum() / variance
        sobol_df.loc[col, 'First order'] = sobol_first_order

        # total order
        list_features_minus_i = list_features.copy()
        list_features_minus_i.remove(col)
        conditional_means = scenarios.groupby(list_features_minus_i)[y].mean()
        counts = scenarios.groupby(list_features_minus_i).size() / len(scenarios)
        sobol_total_order = 1 - (counts * (conditional_means - expectation) ** 2).sum() / variance
        sobol_df.loc[col, 'Total order'] = sobol_total_order
    return sobol_df


def analysis_regret(data, list_features, variable='Total costs'):
    """Calculates the difference of costs between Reference and Ban scenario."""
    ind = data.groupby('Scenario')['passed'].sum()[data.groupby('Scenario')['passed'].sum() == 2].index
    # select subset of scenarios_complete where ban and reference both passed carbon constraint
    tmp = data[data.index.get_level_values('Scenario').isin(ind)]

    tmp_diff = tmp.sort_index().groupby('Scenario')[variable].diff()
    tmp_diff = - tmp_diff[tmp_diff.index.get_level_values('Ban_Status') != 'Ban'].droplevel('Ban_Status')
    tmp_diff = pd.concat([tmp[tmp.index.get_level_values('Ban_Status') != 'Ban'].droplevel('Ban_Status')[list_features], tmp_diff], axis=1)
    return tmp_diff


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
    # order the columns with ORDER_COLUMNS
    df = {key: df[key] for key in ORDER_COLUMNS if key in df.keys()}
    df = {NAME_COLUMNS[key]: value for key, value in df.items()}

    frequency_chart(df, save_path=save_path)


def make_frequency_chart_subplots(df1, df2, folder_name):
    """
    Create a side-by-side comparison of the frequency of each parameter in two datasets.

    Examples:
        df1 = costs_regret[costs_regret["Total costs"] < 0].drop(columns=["Total costs"])
        df2 = costs_regret[costs_regret["Total costs"] > 0].drop(columns=["Total costs"])
        make_frequency_chart_subplots(df1, df2, folder_name)

    Parameters
    ----------
    df1
    df2
    folder_name

    Returns
    -------

    """

    df1 = df1.replace(MAPPING)
    df1 = create_frequency_dict(df1)
    df1 = {NAME_COLUMNS[key]: value for key, value in df1.items()}

    df2 = df2.replace(MAPPING)
    df2 = create_frequency_dict(df2)
    df2 = {NAME_COLUMNS[key]: value for key, value in df2.items()}

    frequency_chart_subplot(df1, df2, save_path=folder_name / Path('total_cost_parameters.pdf'),
                            axis_titles=('Total system cost with ban is lower', 'Total system cost without ban is lower'))


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

    fig, ax = plt.subplots(figsize=(14, 9.6))
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

        ax.bar_label(rects, label_type='center', color=text_color, labels=l, fontsize=18)
    ax.legend(bbox_to_anchor=(0, 1), loc='lower left', fontsize=18, frameon=False, ncol=3)

    # remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)


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
    fig, axs = plt.subplots(1, 2, figsize=(14, 9.6), sharey=True)  # Share Y axis

    for ax_idx, (ax, results, axis_title) in enumerate(zip(axs, [results1, results2], axis_titles)):
        question_labels = list(results.keys())
        data = np.array(list(results.values()))
        data_cum = data.cumsum(axis=1)

        ax.invert_yaxis()
        if ax_idx == 0:  # Apply y-ticks only for the left subplot
            ax.set_yticks(range(len(question_labels)))
            ax.set_yticklabels(question_labels)
        ax.set_xlim(0, np.sum(data, axis=1).max())
        ax.set_title(axis_title, fontsize=20)

        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects = ax.barh(range(len(question_labels)), widths, left=starts, height=0.55, color=color)

            text_color = 'white' if int(color[1:], 16) < 0x888888 else 'black'
            bar_labels = ['{:.0f}%'.format(val * 100) if val != 0 else '' for val in widths]
            ax.bar_label(rects, labels=bar_labels, label_type='center', color=text_color, fontsize=18)

        # remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.xaxis.set_visible(False)

    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in zip(category_names, category_colors)]
    fig.legend(handles=legend_handles, bbox_to_anchor=(1, 0.5), loc='center left', frameon=False, fontsize=18)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.tight_layout()

def horizontal_stack_bar_plot(df, columns=None, title=None, order=None, save_path=None):
    """
    Create a horizontal stacked bar plot from a DataFrame.

    Examples: horizontal_stack_bar_plot(sobol_df.rename(index=NAME_COLUMNS), columns=['First order', 'Total order'],
        title='Influence of parameters that the ban i', order='Total order',
        save_path=folder_name / Path('sobol_ban.png'))

    Parameters
    ----------
    df
    columns
    title
    order
    save_path

    Returns
    -------

    """
    # If no specific columns are provided, use all columns in the DataFrame
    if columns is None:
        columns = df.columns

    if order is not None:
        df = df.sort_values(by=order, ascending=True)

    # Number of rows and bars to plot
    n_rows = len(df)
    n_cols = len(columns)
    bar_width = 0.8 / n_cols  # Adjust bar width based on number of columns
    y_positions = np.arange(n_rows)
    fig, ax = plt.subplots(1, 1, figsize=(14, 9.6))

    # Plot each column
    for i, col in enumerate(columns):
        ax.barh(y_positions - 0.4 + (i + 0.5) * bar_width, df[col], height=bar_width, label=col,
                )

    # Set the y-ticks to use the index of the DataFrame
    ax.set_yticks(y_positions, df.index)
    #ax.yticks(y_positions, df.index)

    # Hide the top, right, and left spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # size of x-axis and y-axis ticks
    ax.tick_params(axis='both', which='major', labelsize=18)
    # size of title

    # Remove the x-axis and y-axis titles
    ax.set_xlabel('')
    ax.set_ylabel('')


    # Set title if provided align on the left
    if title:
        ax.set_title(title, fontsize=20, fontweight='bold', loc='left')

    # Place legend to the right of the figure, without frame
    ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=18)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)


def histogram_plot(df, variable, binrange=None, title=None, save_path=None, xlabel=None, bins=5):
    """
    Plots a histogram of the 'total_costs' column in 'df', adjusting values outside a specified range.

    Examples: plot_adjusted_histogram(costs_regret, 'Total costs', binrange = (-5, 2),
        title='Distribution of additional system costs when Ban is implemented (B€/year)',
        xlabel='Additional system costs (B€/year)', save_path=folder_name / Path('histogram_cost_regret.png'))

    Parameters:
    - df: DataFrame containing the data.
    - total_costs: The name of the column in 'df' to plot.
    - title: Optional; the title of the plot.
    """
    # Copy the DataFrame to avoid modifying the original
    temp = df.copy()

    if binrange is not None:
        # Adjust 'total_costs' values outside the specified range
        temp.loc[temp[variable] < binrange[0], variable] = binrange[0]
        temp.loc[temp[variable] > binrange[1], variable] = binrange[1]

    fig, ax = plt.subplots(1, 1, figsize=(14, 9.6), sharey=True)  # Share Y axis

    # Plotting
    sns.histplot(data=temp, x=variable, stat="proportion", binrange=binrange, ax=ax, color='grey', bins=bins)

    # Set title if provided
    if title:
        ax.set_title(title, fontsize=20, fontweight='bold', loc='left')

    # Customize as needed (e.g., labels)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=18)
    else:
        ax.set_xlabel('')
    ax.set_ylabel('')

    # Hide the top, right, and left spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.set_major_formatter(PercentFormatter(1, 0))

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)


def distributional_plot(df, folder_name=None):
    """
    Plots a bar plot of the difference of distributional index.

    Examples:
        distributional_data = get_distributional_data(scenarios_complete)
        df = distributional_data.loc[ [('S0', 'Ban'), ('S0', 'reference')]].diff().iloc[-1]
        df = df.unstack(-1)
        distributional_plot(df)
    """

    level0_values = df.index.get_level_values(0).unique()
    level1_values = df.index.get_level_values(1).unique()

    # Create the figure and axes for the subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharey=True)

    # Flatten the axes for easy iteration
    axes_flat = axes.flatten()

    # Iterate over the combinations of the first two levels
    for idx, (level0, level1) in enumerate(product(level0_values, level1_values)):
        ax = axes_flat[idx]
        # Select the data for the current combination of level 0 and level 1
        data = df.loc[(level0, level1)]
        data.plot(kind='bar', ax=ax)
        ax.set_title(f'{level0} | {level1}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.axhline(color='black')
        ax.set_xticklabels(data.index, rotation=0, ha='right')

    # Adjust the layout
    plt.tight_layout()
    plt.axhline(color='black')

    if folder_name is not None:
        fig.savefig(folder_name / Path('distributional_plot.png'), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()



if __name__ == '__main__':
    folderpath = Path('/mnt/beegfs/workdir/celia.escribe/eoles2/eoles/outputs/exhaustive_20240226_202408')  # for cluster use
    features = ['policy_heater', 'policy_insulation', 'learning', 'elasticity', 'cop', 'biogas', 'capacity_ren', 'demand', 'carbon_budget', 'gasprices']
    scenarios_complete, output, hourly_generation = parse_outputs(folderpath, features=features)