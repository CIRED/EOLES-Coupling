import json

import pandas as pd
import matplotlib.pyplot as plt
import os
from os import listdir
import seaborn as sns
import numpy as np
from pickle import load
import datetime
from PIL import Image

from eoles.inputs.resources import resources_data
from project.utils import save_fig

sns.set_theme(context="talk", style="white")
# sns.set_theme(context="notebook", style="white")

COLORS_SCENARIOS = {}  # a ajouter des couleurs par simulation
STYLES = ['-', '--', ':', "-.", '*-', 's-', 'o-', '^-', 's-', 'o-', '^-', '*-']

DICT_TRANSFORM_LEGEND = {
    "Annualized electricity system costs": "investment power mix",
    "Annualized investment heater costs": "investment heater switch",
    "Annualized investment insulation costs": "investment insulation",
    "Annualized health costs": "health costs",
    "Annualized total costs HC excluded": "total costs",
    "Annualized total costs": "total costs",
    "Investment electricity costs": "Investment energy mix",
    "Functionment costs": "Energy operational costs",
    "Investment heater costs": "Investment heater switch",
    "Investment insulation costs": "Investment insulation",
    "Carbon cost": "Carbon cost",
    "Health costs": "Health costs",
    "Total costs HC excluded": "Total system costs",
    "Total costs": "Total system costs (Billion EUR)",
    "Consumption saving insulation (TWh/year)": "insulation",
    "Consumption saving heater (TWh/year)": "heater",
    "Consumption saving insulation (TWh)": "insulation",
    "Consumption saving heater (TWh)": "heater",
    "Investment heater (Billion euro)": "investment heater",
    "Investment insulation (Billion euro)": "investment insulation",
    "Subsidies heater (Billion euro)": "subsidies heat pump",
    "Subsidies insulation (Billion euro)": "subsidies insulation",
    "offshore_f": "offshore floating wind",
    "offshore_g": "offshore ground wind",
    "onshore": "onshore wind",
    "pv_g": "pv ground",
    "pv_c": "pv large roof",
    "nuclear": "nuclear",
    "river": "river",
    "lake": "lake",
    "methanization": "methanization",
    "pyrogazification": "pyrogazification",
    "natural_gas": "natural gas"
}

DICT_XLABELS = {
    "Uniform": "Uniform \n Uniform insulation \n ad valorem subsidies \n Heat pump subsidy",
    "No subsidy insulation": "No subsidy insulation \n No subsidy insulation \n Heat pump subsidy",
    # "Global renovation": "Global renovation \n Global renovation \n insulation subsidiy \n Heat pump subsidy",
    "Global renovation no MF": "Global renovation no MF \n Heat pump subsidy",
    "Global renovation FGE": "Global renovation FGE \n Heat pump subsidy",
    "Centralized": "Centralized \n Technical cost-optimal \n insulation measures \n Heat pump subsidy",
    "Centralized GR": "Centralized GR \n Heat pump subsidy",
    "Centralized social ": "Centralized social \n Heat pump subsidy",
    "Efficiency100": "Efficiency100 \n Heat pump subsidy",
    'Global renovation': r'\fontsize{20pt}{3em}\selectfont{}{Global renovation \n}{\fontsize{18pt}{3em}\selectfont{}(GR)}'
}


def plot_comparison(output1, output2, name1, name2):
    # capacity_df_1 = output1["Capacities (GW)"].T.rename(index={2025: name1})
    resirf_subsidies_1 = output1["Subsidies (%)"].rename(index={2025: name1})
    resirf_costs_df_1 = output1["ResIRF costs (Billion euro)"].rename(index={2025: name1})
    resirf_consumption_df_1 = output1["ResIRF consumption (TWh)"].rename(index={2025: name1})
    resirf_replacement_heater_1 = output1["ResIRF replacement heater (Thousand)"].T.rename(index={2025: name1})
    resirf_stock_heater_1 = output1["ResIRF stock heater (Thousand)"].T.rename(index={2029: name1})
    annualized_system_costs_1 = output1["Annualized system costs (Billion euro / year)"].rename(index={2025: name1})

    # capacity_df_2 = output2["Capacities (GW)"]
    resirf_subsidies_2 = output2["Subsidies (%)"].rename(index={2025: name2})
    resirf_costs_df_2 = output2["ResIRF costs (Billion euro)"].rename(index={2025: name2})
    resirf_consumption_df_2 = output2["ResIRF consumption (TWh)"].rename(index={2025: name2})
    resirf_replacement_heater_2 = output2["ResIRF replacement heater (Thousand)"].T.rename(index={2025: name2})
    resirf_stock_heater_2 = output2["ResIRF stock heater (Thousand)"].T.rename(index={2029: name2})
    annualized_system_costs_2 = output2["Annualized system costs (Billion euro / year)"].rename(index={2025: name2})

    resirf_subsidies = pd.concat([resirf_subsidies_1, resirf_subsidies_2], axis=0)
    resirf_costs_df = pd.concat([resirf_costs_df_1, resirf_costs_df_2], axis=0)
    resirf_consumption_df = pd.concat([resirf_consumption_df_1, resirf_consumption_df_2], axis=0)
    resirf_replacement_heater = pd.concat([resirf_replacement_heater_1, resirf_replacement_heater_2], axis=0)
    resirf_stock_heater = pd.concat([resirf_stock_heater_1, resirf_stock_heater_2], axis=0)
    annualized_system_costs = pd.concat([annualized_system_costs_1, annualized_system_costs_2], axis=0)

    make_line_plot(resirf_subsidies, y_label="Subsidies (%)", save=None,
                   format_y=lambda y, _: '{:.0f}'.format(y), index_int=False, str=True)

    make_line_plot(resirf_costs_df, y_label="Costs (Billion euro)", subset=["Heater", "Insulation", "Health cost"],
                   save=None, format_y=lambda y, _: '{:.0f}'.format(y), index_int=False, str=True)

    make_line_plot(resirf_consumption_df, subset=["Electricity", "Natural gas", "Oil fuel", "Wood fuel"],
                   y_label="Heating consumption (TWh)", colors=resources_data["colors_resirf"],
                   save=None, format_y=lambda y, _: '{:.0f}'.format(y), index_int=False, str=True)

    make_line_plot(resirf_consumption_df, subset=['Saving heater', "Saving insulation"],
                   y_label="Savings (TWh)",
                   save=None, format_y=lambda y, _: '{:.0f}'.format(y), index_int=False, str=True)

    make_line_plot(annualized_system_costs, y_label="Annualized system costs",
                   subset=["Annualized electricity system costs"],
                   save=None, index_int=False, str=True)

    make_line_plot(resirf_replacement_heater, y_label="Replacement heater (Thousand households)",
                   save=None, format_y=lambda y, _: '{:.0f}'.format(y), index_int=False, str=True)

    make_line_plot(resirf_stock_heater, y_label="Stock heater (Thousand households)",
                   save=None, format_y=lambda y, _: '{:.0f}'.format(y), index_int=False, str=True)


def plot_comparison_savings_move(df1, df2, x, y, col_for_size, smallest_size=100, biggest_size=300, fontsize=10, y_min=0,
                                 y_max=None, x_min=0, x_max=None, unit='TWh', save=None, coordinates = None, df3=None):
    """Same as next graph, but includes two different scenario"""
    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    # x = "Consumption saving insulation (TWh/year)"
    # y = "Consumption saving heater (TWh/year)"
    relative_size1 = list(df1[col_for_size])
    relative_size2 = list(df2[col_for_size])
    s_min1, s_max1 = min(relative_size1), max(relative_size1)
    s_min2, s_max2 = min(relative_size2), max(relative_size2)
    s_max = max(s_max1, s_max2)
    s_min = min(s_min1, s_min2)
    size1 = [smallest_size + (biggest_size - smallest_size)/(s_max - s_min) * (s - s_min) for s in relative_size1]
    size2 = [smallest_size + (biggest_size - smallest_size) / (s_max - s_min) * (s - s_min) for s in relative_size2]

    if df3 is not None:
        relative_size3 = list(df3[col_for_size])
        s_min3, s_max3 = min(relative_size3), max(relative_size3)
        s_max = max(s_max3, s_max)
        s_min = min(s_min3, s_min)
        size3 = [smallest_size + (biggest_size - smallest_size) / (s_max - s_min) * (s - s_min) for s in relative_size3]

    # colors = dict(zip(df1.index, sns.color_palette(n_colors=len(df1.index))))
    # colors.update({'Historic': (0, 0, 0)})
    # TODO: ajouter une couleur fixe par scénario
    scatter1 = ax.scatter(x=df1[x], y=df1[y], s=size1, c=sns.color_palette(n_colors=len(df1.index)))
    scatter2 = ax.scatter(x=df2[x], y=df2[y], s=size2, c=sns.color_palette(n_colors=len(df1.index)), hatch='/////')
    if df3 is not None:
        scatter3 = ax.scatter(x=df3[x], y=df3[y], s=size3, c=sns.color_palette(n_colors=len(df1.index)), hatch='....')
    if coordinates is None:
        for scenario, v in df1.iterrows():
            ax.annotate(scenario, xy=(v[x], v[y]), xytext=(-20, -25), textcoords="offset points", fontsize=fontsize)
    else:
        for scenario, v in df1.iterrows():
            ax.annotate(scenario, xy=(v[x], v[y]), xytext=coordinates[scenario], textcoords="offset points", fontsize=fontsize)

    # merge = df1.merge(df2, left_index=True, right_index=True)

    # OLD VERSION: fleche automatique avec python
    # for scenario in merge.index.get_level_values(0):
    # # for _, row in merge.iterrows():
    #     x_start = merge.loc[scenario, f'{x}_x']
    #     y_start = merge.loc[scenario, f'{y}_x']
    #     x_end = merge.loc[scenario, f'{x}_y']
    #     y_end = merge.loc[scenario, f'{y}_y']
    #     ax.arrow(x_start, y_start, x_end - x_start, y_end - y_start,
    #              head_width=1, head_length=2, fc='k', ec='k')
    #     midpoint_x = (x_start + x_end) / 2
    #     midpoint_y = (y_start + y_end) / 2
    #     ax.annotate(scenario, xy=(midpoint_x, midpoint_y), xytext=(20, -5),
    #                 textcoords="offset points", fontsize=fontsize)
        # ax.arrow(row[f'{x}_x'], row[f'{y}_x'], row[f'{x}_y'] - row[f'{x}_x'], row[f'{y}_y'] - row[f'{y}_x'],
        #          head_width=1, head_length=2, fc='k', ec='k')
    # for scenario, v in df1.iterrows():
    #     ax.annotate(scenario, xy=(v[x], v[y]), xytext=(20, -5), textcoords="offset points", fontsize=fontsize)

    if y == "Stock Heat pump (Million)":
        title = "Stock Heat pump (Million) \n"
    else:
        title = f"Energy savings through switch to heat pumps {unit} \n"

    ax = format_ax(ax,
                   # title="Comparison savings (TWh)",
                   title=title,
                   # y_label="Savings heater (TWh)",
                   x_label=f"Energy savings through home renovation {unit}",
                   format_y=lambda y, _: '{:.0f}'.format(y), format_x=lambda x, _: '{:.0f}'.format(x),
                   y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max,
                   loc_title="left", c_title="black", loc_xlabel="right")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    kw = dict(prop="sizes", num=2, func=lambda s: s_min + (s - smallest_size) * (s_max - s_min) / (biggest_size - smallest_size))
    # handles, labels = scatter.legend_elements(prop="sizes")
    # legend2 = ax.legend(handles, labels, loc="upper right", title="Sizes")
    if col_for_size == "Total costs":
        title = "Total system costs (Billion EUR)"
    else:
        title = col_for_size
    legend2 = ax.legend(*scatter1.legend_elements(**kw), title=title, loc='upper left', bbox_to_anchor=(1, 0.5), frameon=False)

    # legend_style = plt.legend(dummy_lines_style, labels_style, loc='upper left', bbox_to_anchor=(1, 0.5), frameon=False)
    # legend_color = plt.legend(dummy_lines_color, labels_color, loc='lower left', bbox_to_anchor=(1, 0.5), frameon=False)
    # ax.add_artist(legend_style)
    # ax.add_artist(legend_color)

    save_fig(fig, save=save)


def plot_comparison_savings(df, x, y, save, col_for_size, smallest_size=100, biggest_size=300, fontsize=10, y_min=0, y_max=None, x_min=0, x_max=None,
                            unit="TWh", coordinates=None):
    """

    :param unit: string
        Whether we display savings in absolute or relative terms.
    :return:
    """
    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    # x = "Consumption saving insulation (TWh/year)"
    # y = "Consumption saving heater (TWh/year)"
    relative_size = list(df[col_for_size])
    s_min, s_max = min(relative_size), max(relative_size)
    size = [smallest_size + (biggest_size - smallest_size)/(s_max - s_min) * (s - s_min) for s in relative_size]

    scatter = ax.scatter(x=df[x], y=df[y], s=size, c=sns.color_palette(n_colors=len(df.index)))
    # for scenario, v in df.iterrows():
    #     ax.annotate(scenario, xy=(v[x], v[y]), xytext=(20, -5), textcoords="offset points", fontsize=fontsize)

    if coordinates is None:
        for scenario, v in df.iterrows():
            ax.annotate(scenario, xy=(v[x], v[y]), xytext=(20, -5), textcoords="offset points", fontsize=fontsize)
    else:
        for scenario, v in df.iterrows():
            if scenario in coordinates.keys():
                ax.annotate(scenario, xy=(v[x], v[y]), xytext=coordinates[scenario], textcoords="offset points", fontsize=fontsize)
            else:
                ax.annotate(scenario, xy=(v[x], v[y]), xytext=(20, -5), textcoords="offset points",
                            fontsize=fontsize)

    if y == "Stock Heat pump (Million)":
        title = "Stock Heat pump (Million) \n"
    else:
        title = f"Energy savings through switch to heat pumps {unit} \n"

    ax = format_ax(ax,
                   # title="Comparison savings (TWh)",
                   title=title,
                   # y_label="Savings heater (TWh)",
                   x_label=f"Energy savings through home renovation {unit}",
                   format_y=lambda y, _: '{:.0f}'.format(y), format_x=lambda x, _: '{:.0f}'.format(x),
                   y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max,
                   loc_title="left", c_title="black", loc_xlabel="right")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    kw = dict(prop="sizes", num=2, func=lambda s: s_min + (s - smallest_size) * (s_max - s_min) / (biggest_size - smallest_size))
    # handles, labels = scatter.legend_elements(prop="sizes")
    # legend2 = ax.legend(handles, labels, loc="upper right", title="Sizes")
    if col_for_size == "Total costs":
        title = "Total system costs (Billion EUR)"
    else:
        title = col_for_size
    legend2 = ax.legend(*scatter.legend_elements(**kw), title=title, loc='upper left', bbox_to_anchor=(1, 0.5), frameon=False)

    save_fig(fig, save=save)


def comparison_simulations_scenarios(dict_output1, dict_output2, x_min=-5, x_max=None, y_min=-5, y_max=None,
                                     save_path=None, pdf=False, carbon_constraint=True, percent=False, eoles=True,
                                     coordinates=None, dict_output3=None, smallest_size=100, biggest_size=400, fontsize=18):
    if pdf:
        extension = "pdf"
    else:
        extension = "png"
    # annualized_system_costs_df = pd.DataFrame(dtype=float)
    total_system_costs_df1, total_system_costs_df2, total_system_costs_df3 = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)
    consumption_savings_tot_df1, consumption_savings_tot_df2, consumption_savings_tot_df3 = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)

    if save_path is not None:
        if not os.path.isdir(save_path):  # create directory
            os.mkdir(save_path)

    for path, name_config in zip(dict_output1.values(), [n for n in dict_output1.keys()]):
        with open(os.path.join(path, 'coupling_results.pkl'), "rb") as file:
            output = load(file)

            annualized_new_investment_df = output["Annualized new investments (1e9€/yr)"]
            annualized_new_energy_capacity_df = output["Annualized costs new energy capacity (1e9€/yr)"]
            functionment_costs_df = output["System functionment (1e9€/yr)"]
            total_system_costs = process_total_costs(annualized_new_investment_df, annualized_new_energy_capacity_df,
                                                     functionment_costs_df, carbon_constraint=carbon_constraint, eoles=eoles)
            total_system_costs = total_system_costs.to_frame().rename(columns={0: name_config})
            total_system_costs_df1 = pd.concat([total_system_costs_df1, total_system_costs], axis=1)

            try:
                consumption_savings = output["ResIRF consumption savings (TWh/year)"]
            except:
                consumption_savings = output["ResIRF consumption savings (TWh)"]
                consumption_savings = consumption_savings.rename(
                    columns={"Consumption saving heater (TWh)": "Consumption saving heater (TWh/year)",
                             "Consumption saving insulation (TWh)": "Consumption saving insulation (TWh/year)"})
            consumption = output["Output global ResIRF ()"].loc[["Consumption Electricity (TWh)", "Consumption Natural gas (TWh)",
                                                                 "Consumption Oil fuel (TWh)", "Consumption Wood fuel (TWh)"]]
            consumption_ini = consumption.sum(axis=0).iloc[0]
            consumption_savings_tot = consumption_savings.sum(axis=0).to_frame().rename(columns={0: name_config})
            if percent:
                consumption_savings_tot = consumption_savings_tot / consumption_ini * 100
            consumption_savings_tot_df1 = pd.concat([consumption_savings_tot_df1, consumption_savings_tot], axis=1)

    for path, name_config in zip(dict_output2.values(), [n for n in dict_output2.keys()]):
        with open(os.path.join(path, 'coupling_results.pkl'), "rb") as file:
            output = load(file)

            annualized_new_investment_df = output["Annualized new investments (1e9€/yr)"]
            annualized_new_energy_capacity_df = output["Annualized costs new energy capacity (1e9€/yr)"]
            functionment_costs_df = output["System functionment (1e9€/yr)"]
            total_system_costs = process_total_costs(annualized_new_investment_df, annualized_new_energy_capacity_df,
                                                     functionment_costs_df, carbon_constraint=carbon_constraint, eoles=eoles)
            total_system_costs = total_system_costs.to_frame().rename(columns={0: name_config})
            total_system_costs_df2 = pd.concat([total_system_costs_df2, total_system_costs], axis=1)

            try:
                consumption_savings = output["ResIRF consumption savings (TWh/year)"]
            except:
                consumption_savings = output["ResIRF consumption savings (TWh)"]
                consumption_savings = consumption_savings.rename(
                    columns={"Consumption saving heater (TWh)": "Consumption saving heater (TWh/year)",
                             "Consumption saving insulation (TWh)": "Consumption saving insulation (TWh/year)"})
            consumption = output["Output global ResIRF ()"].loc[["Consumption Electricity (TWh)", "Consumption Natural gas (TWh)",
                                                                 "Consumption Oil fuel (TWh)", "Consumption Wood fuel (TWh)"]]
            consumption_ini = consumption.sum(axis=0).iloc[0]
            consumption_savings_tot = consumption_savings.sum(axis=0).to_frame().rename(columns={0: name_config})
            if percent:
                consumption_savings_tot = consumption_savings_tot / consumption_ini * 100
            consumption_savings_tot_df2 = pd.concat([consumption_savings_tot_df2, consumption_savings_tot], axis=1)

    if dict_output3 is not None:
        for path, name_config in zip(dict_output3.values(), [n for n in dict_output3.keys()]):
            with open(os.path.join(path, 'coupling_results.pkl'), "rb") as file:
                output = load(file)

                annualized_new_investment_df = output["Annualized new investments (1e9€/yr)"]
                annualized_new_energy_capacity_df = output["Annualized costs new energy capacity (1e9€/yr)"]
                functionment_costs_df = output["System functionment (1e9€/yr)"]
                total_system_costs = process_total_costs(annualized_new_investment_df,
                                                         annualized_new_energy_capacity_df,
                                                         functionment_costs_df, carbon_constraint=carbon_constraint,
                                                         eoles=eoles)
                total_system_costs = total_system_costs.to_frame().rename(columns={0: name_config})
                total_system_costs_df3 = pd.concat([total_system_costs_df3, total_system_costs], axis=1)

                try:
                    consumption_savings = output["ResIRF consumption savings (TWh/year)"]
                except:
                    consumption_savings = output["ResIRF consumption savings (TWh)"]
                    consumption_savings = consumption_savings.rename(
                        columns={"Consumption saving heater (TWh)": "Consumption saving heater (TWh/year)",
                                 "Consumption saving insulation (TWh)": "Consumption saving insulation (TWh/year)"})
                consumption = output["Output global ResIRF ()"].loc[
                    ["Consumption Electricity (TWh)", "Consumption Natural gas (TWh)",
                     "Consumption Oil fuel (TWh)", "Consumption Wood fuel (TWh)"]]
                consumption_ini = consumption.sum(axis=0).iloc[0]
                consumption_savings_tot = consumption_savings.sum(axis=0).to_frame().rename(columns={0: name_config})
                if percent:
                    consumption_savings_tot = consumption_savings_tot / consumption_ini * 100
                consumption_savings_tot_df3 = pd.concat([consumption_savings_tot_df3, consumption_savings_tot], axis=1)

    if percent:
        unit = "(%)"
    else:
        unit = "(TWh)"

    savings_and_costs_df1 = pd.concat([consumption_savings_tot_df1, total_system_costs_df1], axis=0)
    savings_and_costs_df1 = savings_and_costs_df1.T
    savings_and_costs_df2 = pd.concat([consumption_savings_tot_df2, total_system_costs_df2], axis=0)
    savings_and_costs_df2 = savings_and_costs_df2.T

    savings_and_costs_df3 = None
    if dict_output3 is not None:
        savings_and_costs_df3 = pd.concat([consumption_savings_tot_df3, total_system_costs_df3], axis=0)
        savings_and_costs_df3 = savings_and_costs_df3.T

    if save_path is None:
        save_path_plot = None
    else:
        save_path_plot = os.path.join(save_path, f"consumption_savings_comparison.{extension}")

    plot_comparison_savings_move(savings_and_costs_df1, savings_and_costs_df2, x="Consumption saving insulation (TWh/year)",
                                 y="Consumption saving heater (TWh/year)",
                                 col_for_size="Total costs", smallest_size=smallest_size, biggest_size=biggest_size,
                                 fontsize=fontsize,
                                 y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max, unit=unit,
                                 save=os.path.join(save_path, f"savings_and_costs.{extension}"),
                                 coordinates=coordinates, df3=savings_and_costs_df3)


def plot_capacities_barplot_2(capacity_df, save):
    # TODO: graphe en cours, il faut ajouter du white space entre les barres et ajouter les noms des configurations.
    list_configs = capacity_df['Configuration'].unique()
    list_years = capacity_df.index.unique()

    fig, ax = fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    df = capacity_df.loc[capacity_df.Configuration == list_configs[0]]
    df.index.name = None
    df.plot(kind='bar', stacked=True, ax=ax, linewidth=0, position=1.5, width=0.1)

    df = capacity_df.loc[capacity_df.Configuration == list_configs[1]]
    df.index.name = None
    df.plot(kind='bar', stacked=True, ax=ax, linewidth=0, position=-0.5, width=0.1)

    df = capacity_df.loc[capacity_df.Configuration == list_configs[2]]
    df.index.name = None
    df.plot(kind='bar', stacked=True, ax=ax, linewidth=0, position=0.5, width=0.1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    current_labels = ax.get_legend_handles_labels()[1]
    current_labels = current_labels[0:int(len(current_labels) / len(list_configs))]
    new_labels = [DICT_TRANSFORM_LEGEND[e] if e in DICT_TRANSFORM_LEGEND.keys() else e for e in current_labels]
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), labels=new_labels, frameon=False)

    # # Calculate positions and width
    # positions = np.array([0, 0.5, 1])  # You can adjust these values as needed
    # width = 0.3  # You can adjust the width of the bars as needed
    #
    # xticks = positions
    # xticks_labels = list(capacity_df['Configuration'].unique())
    #
    # df = capacity_df.loc[2030].set_index("Configuration")
    # df.index.name = None
    # df.plot(kind='bar', stacked=True, ax=ax, linewidth=0, position=positions, width=width)
    #
    # # Calculate positions and width
    # positions = np.array([2.5, 3, 3.5])  # You can adjust these values as needed
    # width = 0.3  # You can adjust the width of the bars as needed
    #
    # df = capacity_df.loc[2035].set_index("Configuration")
    # df.index.name = None
    # df.plot(kind='bar', stacked=True, ax=ax, linewidth=0, position=positions, width=width)
    #
    # xticks = np.concatenate((xticks, positions))
    # xticks_labels +=  list(capacity_df['Configuration'].unique())
    #
    # # Customize the x-axis labels if needed
    # ax.set_xticks(positions)
    # # ax.set_xticklabels(xticks_labels)

    # Show the plot
    plt.show()


def plot_capacities_barplot(capacity_df, save, rotation):
    # capacity_df = capacity_df.reset_index().rename(columns={'index': 'Year'})
    # years = capacity_df['Year'].unique()
    # configurations = capacity_df['Configuration'].unique()
    #
    # # Set up the subplots
    # fig, axs = plt.subplots(1, len(years), figsize=(15, 5), sharey=True)
    #
    # # Loop through years and create stacked bar plots
    # for i, year in enumerate(years):
    #     ax = axs[i]
    #     year_data = capacity_df[capacity_df['Year'] == year]
    #
    #     bottom = np.zeros(len(configurations))  # Initialize bottom values for stacking
    #     year_data.set_index("Configuration").plot(kind='bar', stacked=True, ax=ax, linewidth=0)
    #
    #     # for j, config in enumerate(configurations):
    #     #     config_data = year_data[year_data['Configuration'] == config]
    #     #     ax.bar(config_data.index, config_data['Technology1'], label=config, bottom=bottom)
    #     #     bottom += config_data['ocgt']  # Update bottom values for stacking
    #
    #     ax.set_xlabel('Configuration')
    #     ax.set_ylabel('Technology1 Capacity')
    #     ax.set_title(f'Stacked Bar Plot for {year}')
    #     ax.set_xticks(year_data.index)
    #     ax.set_xticklabels(year_data['Configuration'])
    #     ax.legend()
    #
    # plt.tight_layout()
    # plt.show()

    list_configs = capacity_df['Configuration'].unique()
    list_years = capacity_df.index.unique()

    fig, axs = plt.subplots(1, len(list_years), figsize=(15, 5), sharey=True)
    for i, year in enumerate(list_years):
        ax = axs[i]
        df = capacity_df.loc[year].set_index("Configuration")
        df.index.name = None
        df.plot(kind='bar', stacked=True, ax=ax, linewidth=0)
        ax = format_ax_string(ax, title="", x_ticks_labels=df.index, format_y=lambda y, _: '{:.0f}'.format(y), rotation=rotation)
        if i == len(list_years) - 1:
            # format_legend(ax, dict_legend=DICT_TRANSFORM_LEGEND)

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            current_labels = ax.get_legend_handles_labels()[1]
            new_labels = [DICT_TRANSFORM_LEGEND[e] if e in DICT_TRANSFORM_LEGEND.keys() else e for e in current_labels]
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), labels=new_labels, frameon=False)

        else:
            ax.get_legend().remove()

        plt.axhline(y=0)

    save_fig(fig, save=save)

        # make_stacked_bar_plot(capacity_df.loc[year].set_index("Configuration"), y_label="Flexible capacity (GW)",
        #                   format_y=lambda y, _: '{:.0f}'.format(y), index_int=False, rotation=90, dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot)

    # fig, axs = plt.subplots(len(list_years), figsize=(10, 8))  # TODO: a changer
    #
    # for i, year in enumerate(list_years):
    #     data = capacity_df.loc[year]
    #     width = 0.35
    #     x = np.arange(len(list_configs))
    #     ax = axs[i]
    #
    #     # for j, config_name in enumerate(list_configs):
    #     #     data.drop(columns=["Configuration"]).iloc[j].plot(kind='bar', stacked=True, ax=ax, linewidth=0)  # TODO: il faut changer l'endroit où plotter, là ils s'entassent.
    #
    #     for j, config_name in enumerate(list_configs):
    #         # Add an offset to x for each configuration
    #         ax.bar(x + j * width, data.drop(columns=["Configuration"]).iloc[j], width, label=config_name)
    #
    #
    #     ax.set_xlabel('Configuration')
    #     ax.set_ylabel('Capacities')
    #     ax.set_title(f'Stacked Bar Plot for {year}')
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(list_configs)
    #     ax.legend()
    #
    # save_fig(fig, save=save)

def comparison_simulations(dict_output: dict, ref, greenfield=False, health=False, x_min=0, x_max=None, y_min=0, y_max=None,
                           rotation=90, save_path=None, pdf=False, carbon_constraint=True, percent=False, eoles=True,
                           coordinates=None, secondary_y=None, secondary_axis_spec=None, smallest_size=100, biggest_size=400,
                           fontsize=18):
    if pdf:
        extension = "pdf"
    else:
        extension = "png"
    # annualized_system_costs_df = pd.DataFrame(dtype=float)
    total_system_costs_df = pd.DataFrame(dtype=float)
    complete_system_costs_2050_df = pd.DataFrame(dtype=float)
    consumption_savings_tot_df, consumption_resirf_df = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)
    stock_heat_pump_df = pd.DataFrame(dtype=float)
    generation_df, conversion_generation_df = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)
    peak_electricity_load_dict = {}
    consumption_saving_evolution_dict = {}
    emissions_dict = {}
    # annualized_system_costs_dict = {}
    subsidies_insulation_dict = {}
    subsidies_heater_dict = {}
    capacities_vre_dict, capacities_peaking_dict = {}, {}
    capacities_flex_df = pd.DataFrame(dtype=float)

    if save_path is not None:
        if not os.path.isdir(save_path):  # create directory
            os.mkdir(save_path)

    for path, name_config in zip(dict_output.values(), [n for n in dict_output.keys()]):
        with open(os.path.join(path, 'coupling_results.pkl'), "rb") as file:
            output = load(file)
            # annualized_system_costs = output["Annualized system costs (Billion euro / year)"]
            # annualized_system_costs_dict[name_config] = annualized_system_costs.sum(axis=0)
            # annualized_system_costs = annualized_system_costs.sum(axis=0).to_frame().rename(columns={0: name_config})
            # annualized_system_costs_df = pd.concat([annualized_system_costs_df, annualized_system_costs], axis=1)

            annualized_new_investment_df = output["Annualized new investments (1e9€/yr)"]
            annualized_new_energy_capacity_df = output["Annualized costs new energy capacity (1e9€/yr)"]
            functionment_costs_df = output["System functionment (1e9€/yr)"]
            total_system_costs = process_total_costs(annualized_new_investment_df, annualized_new_energy_capacity_df,
                                                     functionment_costs_df, carbon_constraint=carbon_constraint, eoles=eoles)
            total_system_costs = total_system_costs.to_frame().rename(columns={0: name_config})
            total_system_costs_df = pd.concat([total_system_costs_df, total_system_costs], axis=1)

            annualized_new_investment_df = output["Annualized new investments (1e9€/yr)"]
            annualized_new_energy_capacity_df = output["Annualized costs new energy capacity (1e9€/yr)"]
            functionment_costs_df = output["System functionment (1e9€/yr)"]
            complete_system_costs_2050 = process_complete_system_cost_2050(annualized_new_investment_df,
                                                                           annualized_new_energy_capacity_df,
                                                                           functionment_costs_df, carbon_constraint=carbon_constraint, eoles=eoles)
            complete_system_costs_2050 = complete_system_costs_2050.to_frame().rename(columns={0: name_config})
            complete_system_costs_2050_df = pd.concat([complete_system_costs_2050_df, complete_system_costs_2050],
                                                      axis=1)

            try:
                consumption_savings = output["ResIRF consumption savings (TWh/year)"]
            except:
                consumption_savings = output["ResIRF consumption savings (TWh)"]
                consumption_savings = consumption_savings.rename(
                    columns={"Consumption saving heater (TWh)": "Consumption saving heater (TWh/year)",
                             "Consumption saving insulation (TWh)": "Consumption saving insulation (TWh/year)"})
            consumption = output["Output global ResIRF ()"].loc[["Consumption Electricity (TWh)", "Consumption Natural gas (TWh)",
                                                                 "Consumption Oil fuel (TWh)", "Consumption Wood fuel (TWh)"]]

            consumption_ini = consumption.sum(axis=0).iloc[0]
            consumption_savings_tot = consumption_savings.sum(axis=0).to_frame().rename(columns={0: name_config})
            if percent:
                consumption_savings_tot = consumption_savings_tot / consumption_ini * 100
            consumption_savings_tot_df = pd.concat([consumption_savings_tot_df, consumption_savings_tot], axis=1)

            consumption_savings_evolution = consumption_savings.reset_index().rename(columns={'index': 'year'})
            consumption_savings_evolution["period"] = consumption_savings_evolution.apply(
                lambda row: (row["year"] - 2025) // 5, axis=1)
            consumption_savings_evolution = consumption_savings_evolution.groupby("period").agg(
                {"year": np.min, "Consumption saving heater (TWh/year)": np.sum,
                 "Consumption saving insulation (TWh/year)": np.sum}).set_index("year")
            consumption_savings_evolution.index.name = None
            consumption_saving_evolution_dict[name_config] = consumption_savings_evolution

            if eoles:
                consumption_resirf = output['ResIRF consumption (TWh)'].T[2050]
                if isinstance(consumption_resirf, pd.Series):  # in the case of greenfield
                    consumption_resirf = consumption_resirf.to_frame()
                consumption_resirf = consumption_resirf.rename(columns={2050: name_config})
                consumption_resirf_df = pd.concat([consumption_resirf_df, consumption_resirf], axis=1)

            output_resirf = output["Output global ResIRF ()"]
            stock_heat_pump = pd.Series(output_resirf.loc["Stock Heat pump (Million)"][2049], index=["Stock Heat pump (Million)"]).to_frame().rename(columns={0: name_config})
            stock_heat_pump_df = pd.concat([stock_heat_pump_df, stock_heat_pump], axis=1)

            try:
                peak_electricity_load_info_df = output["Peak electricity load"]
                peak_electricity_load_info_df = peak_electricity_load_info_df[["peak_electricity_load", "year"]].groupby(
                    ["year"]).mean().squeeze()
                peak_electricity_load_dict[name_config] = peak_electricity_load_info_df

                emissions = output["Emissions (MtCO2)"]
                if greenfield:
                    emissions = pd.Series(emissions.squeeze(), index=emissions.index)
                else:
                    emissions = emissions.squeeze()
                emissions_dict[name_config] = emissions

                subsidies = output["Subsidies (%)"] * 100
                dataframe_subsidy_list = [subsidies]
                for i in range(4):
                    tmp = subsidies.copy()
                    tmp.index += i + 1
                    dataframe_subsidy_list.append(tmp)
                dataframe_subsidy = pd.concat(dataframe_subsidy_list, axis=0).sort_index(ascending=True)

                subsidies_insulation_dict[name_config] = dataframe_subsidy[["Insulation"]].squeeze()
                subsidies_heater_dict[name_config] = dataframe_subsidy[["Heater"]].squeeze()
                if secondary_y is not None and name_config == secondary_y:
                    with open(os.path.join(path, 'config', 'config_coupling.json')) as file:
                        config_coupling = json.load(file)
                    price_cap = config_coupling['subsidy']['insulation']['cap']
                    subsidies_insulation_dict[name_config] = subsidies_insulation_dict[name_config]*price_cap/100
            except:
                pass

            if eoles:
                capacities_df = output["Capacities (GW)"].T
                selected_capacities = ["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "battery1", "battery4"]
                capacities_df = capacities_df[selected_capacities]
                capacities_df["offshore"] = capacities_df["offshore_f"] + capacities_df["offshore_g"]
                capacities_df["pv"] = capacities_df["pv_g"] + capacities_df["pv_c"]
                capacities_df["battery"] = capacities_df["battery1"] + capacities_df["battery4"]
                capacities_vre_dict[name_config] = capacities_df[["offshore", "onshore", "pv", "battery"]]

                capacities_df = output["Capacities (GW)"].T
                tec_flexibility = ['nuclear', "phs", 'ocgt', 'ccgt', 'h2_ccgt', 'battery1', 'battery4']
                capacities_flex = capacities_df[tec_flexibility].copy()
                capacities_flex['Configuration'] = name_config
                capacities_flex_df = pd.concat([capacities_flex_df, capacities_flex])  # TODO: prendre l'année 2050, ou bien montrer l'évolution des capacités au cours du temps

                capacities_df = output["Capacities (GW)"].T
                capacities_df = capacities_df[['ocgt', 'ccgt', 'h2_ccgt']]
                capacities_df['methane_plants'] = capacities_df['ocgt'] + capacities_df['ccgt']
                capacities_peaking_dict[name_config] = capacities_df[['methane_plants', 'h2_ccgt']]

                generation = output["Generation (TWh)"]
                generation_2050 = generation[[2050]]
                generation_2050 = generation_2050.rename(columns={2050: name_config}).T
                generation_2050['offshore'] = generation_2050['offshore_f'] + generation_2050['offshore_g']
                generation_2050['pv'] = generation_2050['pv_g'] + generation_2050['pv_c']
                generation_2050['peaking plants'] = generation_2050['ocgt'] + generation_2050['ccgt'] + generation_2050['h2_ccgt']
                generation_2050['hydro'] = generation_2050['river'] + generation_2050['lake']
                generation_2050['biogas'] = generation_2050['methanization'] + generation_2050['pyrogazification']
                generation_2050 = pd.concat([generation_2050.T, consumption_resirf.rename(index={'Electricity': 'Electricity for heating',
                                                                                                 'Natural gas': 'Gas for heating'})], axis=0)
                generation_df = pd.concat([generation_df, generation_2050], axis=1)

                conversion_generation = output['Conversion generation (TWh)']
                conversion_generation = conversion_generation[[2050]]
                conversion_generation = conversion_generation.rename(columns={2050: name_config}).T
                # conversion_generation['peaking plants'] = conversion_generation['ocgt'] + conversion_generation['ccgt'] + conversion_generation['h2_ccgt']
                conversion_generation['peaking plants'] = conversion_generation['ocgt'] + conversion_generation['ccgt']  # TODO: à modifier pour les prochains runs
                conversion_generation_df = pd.concat([conversion_generation_df, conversion_generation.T], axis=1)

    if eoles:
        generation_df, conversion_generation_df = generation_df.T, conversion_generation_df.T

        if save_path is None:
            save_path_plot = None
        else:
            save_path_plot = os.path.join(save_path, f"gas_demand_balance.{extension}")
        supply_gas = ['biogas', 'methanation', 'electrolysis']
        demand_gas = ['peaking plants', 'Gas for heating']

        gas_generation_demand = pd.concat([generation_df[['biogas', 'methanation', 'electrolysis', 'Gas for heating']], conversion_generation_df[['peaking plants']]], axis=1)
        gas_generation_demand[demand_gas] = - gas_generation_demand[demand_gas]
        # TODO: ajouter la demande en hydrogène
        # TODO: ajouter une option en différence par rapport à une ref
        # gas_generation_demand = gas_generation_demand.T
        # for col in gas_generation_demand.columns:
        #     if col != ref:
        #         gas_generation_demand[col] = gas_generation_demand[col] - gas_generation_demand[ref]
        make_stacked_bar_plot(gas_generation_demand, subset=supply_gas + demand_gas, y_label="Gas demand balance (TWh)",
                              colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                              index_int=False, rotation=rotation, dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot, hline=True)

        if save_path is None:
            save_path_plot = None
        else:
            save_path_plot = os.path.join(save_path, f"elec_demand_balance.{extension}")
        supply_elec = ['onshore', 'offshore', 'pv', 'hydro', 'peaking plants']
        demand_elec = ['electrolysis', 'methanation', 'Electricity for heating']
        elec_generation_demand = pd.concat([generation_df[supply_elec + ['Electricity for heating']], conversion_generation_df[['electrolysis', 'methanation']]], axis=1)
        elec_generation_demand[demand_elec] = - elec_generation_demand[demand_elec]
        # TODO: ajouter la demande en elec
        make_stacked_bar_plot(elec_generation_demand, subset=supply_elec + demand_elec, y_label="Electricity demand balance (TWh)",
                              colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                              index_int=False, rotation=rotation, dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot, hline=True)

    # Total system costs
    if carbon_constraint:
        subset_annualized_costs = ["Investment electricity costs", "Investment heater costs",
                                   "Investment insulation costs", "Functionment costs", "Health costs"]
    else:
        subset_annualized_costs = ["Investment electricity costs", "Investment heater costs",
                                   "Investment insulation costs", "Functionment costs", "Carbon cost", "Health costs"]
    if not eoles:  # only includes ResIRF for CBA
        subset_annualized_costs.remove("Investment electricity costs")
    if save_path is None:
        save_path_plot = None
    else:
        save_path_plot = os.path.join(save_path, f"total_system_costs.{extension}")
    make_stacked_bar_plot(total_system_costs_df.T, subset=subset_annualized_costs, y_label="Total system costs (Md€)",
                          colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                          index_int=False,
                          rotation=90, dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot)

    # Total system costs
    total_system_costs_diff_df = total_system_costs_df.T.copy()
    total_system_costs_diff_df["Total costs HC excluded"] = total_system_costs_diff_df["Total costs"] - \
                                                       total_system_costs_diff_df["Health costs"]
    total_system_costs_diff_df = total_system_costs_diff_df.T

    for col in total_system_costs_diff_df.columns:
        if col != ref:
            total_system_costs_diff_df[col] = total_system_costs_diff_df[col] - total_system_costs_diff_df[ref]
    if health:
        if carbon_constraint:
            subset_costs = ["Investment electricity costs", "Investment heater costs",
                            "Investment insulation costs", "Functionment costs", "Health costs"]
        else:
            subset_costs = ["Investment electricity costs", "Investment heater costs",
                            "Investment insulation costs", "Functionment costs", "Carbon cost", "Health costs"]
    else:
        if carbon_constraint:
            subset_costs = ["Investment electricity costs", "Investment heater costs",
                            "Investment insulation costs", "Functionment costs"]
        else:
            subset_costs = ["Investment electricity costs", "Investment heater costs",
                            "Investment insulation costs", "Functionment costs", "Carbon cost"]

    if not eoles:
        subset_costs.remove('Investment electricity costs')

    if save_path is None:
        save_path_plot = None
    else:
        save_path_plot = os.path.join(save_path, f"difference_total_system_costs.{extension}")
    if len(total_system_costs_diff_df.columns) >= 3:  # ie, at least two scenarios to compare to the ref
        if health:
            make_stacked_investment_plot(df=total_system_costs_diff_df.drop(columns=[ref]).T,
                                         # y_label="Difference of total system costs over 2025-2050 (Billion €)",
                                         y_label="Total costs (Billion EUR)",
                                         subset=subset_costs,
                                         scatter=total_system_costs_diff_df.drop(columns=[ref]).T[
                                             ["Total costs"]].squeeze(),
                                         save=save_path_plot, colors=resources_data["colors_eoles"],
                                         format_y=lambda y, _: '{:.0f}'.format(y), rotation=rotation,
                                         dict_legend=DICT_TRANSFORM_LEGEND, dict_xlabels=None)
        else:
            make_stacked_investment_plot(df=total_system_costs_diff_df.drop(columns=[ref]).T,
                                         y_label="Difference of total system costs over 2025-2050 (Billion €)",
                                         subset=subset_costs,
                                         scatter=total_system_costs_diff_df.drop(columns=[ref]).T[
                                             ["Total costs HC excluded"]].squeeze(),
                                         save=save_path_plot, colors=resources_data["colors_eoles"],
                                         format_y=lambda y, _: '{:.0f}'.format(y), rotation=rotation,
                                         dict_legend=DICT_TRANSFORM_LEGEND, dict_xlabels=None)

    else:
        make_stacked_investment_plot(df=total_system_costs_diff_df.drop(columns=[ref]).T,
                                     y_label="Difference of total system costs over 2025-2050 (Billion €)",
                                     subset=subset_costs,
                                     scatter=total_system_costs_diff_df.drop(columns=[ref]).T["Total costs"],
                                     save=save_path_plot, colors=resources_data["colors_eoles"],
                                     format_y=lambda y, _: '{:.0f}'.format(y), rotation=rotation,
                                     dict_legend=DICT_TRANSFORM_LEGEND, dict_xlabels=None)

    # Complete system costs in 2050
    if carbon_constraint:
        subset_complete_costs = ["Investment electricity costs", "Investment heater costs",
                                 "Investment insulation costs", "Functionment costs", "Health costs"]
    else:
        subset_complete_costs = ["Investment electricity costs", "Investment heater costs",
                                 "Investment insulation costs", "Functionment costs", "Carbon cost", "Health costs"]

    if not eoles:
        subset_complete_costs.remove('Investment electricity costs')

    if save_path is None:
        save_path_plot = None
    else:
        save_path_plot = os.path.join(save_path, f"complete_system_costs_2050.{extension}")
    make_stacked_bar_plot(complete_system_costs_2050_df.T, subset=subset_complete_costs,
                          y_label="Complete system costs in 2050 (Md€/year)",
                          colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.01f}'.format(y),
                          index_int=False,
                          rotation=90, dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot)

    complete_system_costs_2050_diff_df = complete_system_costs_2050_df.T.copy()
    complete_system_costs_2050_diff_df["Total costs HC excluded"] = complete_system_costs_2050_diff_df["Total costs"] - \
                                                       complete_system_costs_2050_diff_df["Health costs"]
    complete_system_costs_2050_diff_df = complete_system_costs_2050_diff_df.T

    if health:
        if carbon_constraint:
            subset_complete_costs = ["Investment electricity costs", "Investment heater costs",
                            "Investment insulation costs", "Functionment costs", "Health costs"]
        else:
            subset_complete_costs = ["Investment electricity costs", "Investment heater costs",
                            "Investment insulation costs", "Functionment costs", "Carbon cost", "Health costs"]
    else:
        if carbon_constraint:
            subset_complete_costs = ["Investment electricity costs", "Investment heater costs",
                            "Investment insulation costs", "Functionment costs"]
        else:
            subset_complete_costs = ["Investment electricity costs", "Investment heater costs",
                                     "Investment insulation costs", "Functionment costs", "Carbon cost"]

    if not eoles:
        subset_complete_costs.remove('Investment electricity costs')

    for col in complete_system_costs_2050_diff_df.columns:
        if col != ref:
            complete_system_costs_2050_diff_df[col] = complete_system_costs_2050_diff_df[col] - complete_system_costs_2050_diff_df[ref]

    if save_path is None:
        save_path_plot = None
    else:
        save_path_plot = os.path.join(save_path, f"difference_complete_system_costs_2050.{extension}")
    if len(complete_system_costs_2050_diff_df.columns) >= 3:  # ie, at least two scenarios to compare to the ref
        if health:
            make_stacked_investment_plot(df=complete_system_costs_2050_diff_df.drop(columns=[ref]).T,
                                         y_label="Difference of complete system costs in 2050 (Billion € / year)",
                                         subset=subset_complete_costs,
                                         scatter=complete_system_costs_2050_diff_df.drop(columns=[ref]).T[
                                             ["Total costs"]].squeeze(),
                                         save=save_path_plot, colors=resources_data["colors_eoles"],
                                         format_y=lambda y, _: '{:.1f}'.format(y), rotation=rotation,
                                         dict_legend=DICT_TRANSFORM_LEGEND, dict_xlabels=None)
        else:
            make_stacked_investment_plot(df=complete_system_costs_2050_diff_df.drop(columns=[ref]).T,
                                         y_label="Difference of complete system costs in 2050 (Billion € / year)",
                                         subset=subset_complete_costs,
                                         scatter=complete_system_costs_2050_diff_df.drop(columns=[ref]).T[
                                             ["Total costs HC excluded"]].squeeze(),
                                         save=save_path_plot, colors=resources_data["colors_eoles"],
                                         format_y=lambda y, _: '{:.1f}'.format(y), rotation=rotation,
                                         dict_legend=DICT_TRANSFORM_LEGEND, dict_xlabels=None)
    else:
        make_stacked_investment_plot(df=complete_system_costs_2050_diff_df.drop(columns=[ref]).T,
                                     y_label="Difference of complete system costs in 2050 (Billion € / year)",
                                     subset=subset_complete_costs,
                                     scatter=complete_system_costs_2050_diff_df.drop(columns=[ref]).T["Total costs"],
                                     save=save_path_plot, colors=resources_data["colors_eoles"],
                                     format_y=lambda y, _: '{:.1f}'.format(y), rotation=rotation,
                                     dict_legend=DICT_TRANSFORM_LEGEND, dict_xlabels=None)

    # Total consumption savings
    if save_path is None:
        save_path_plot = None
    else:
        save_path_plot = os.path.join(save_path, f"consumption_savings.{extension}")
    if percent:
        unit = "(%)"
    else:
        unit = "(TWh)"
    make_stacked_bar_plot(consumption_savings_tot_df.T, y_label=f"Total consumption savings {unit}",
                          colors=resources_data["colors_resirf"], format_y=lambda y, _: '{:.0f}'.format(y),
                          index_int=False,
                          rotation=90, dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot)

    savings_and_costs_df = pd.concat([consumption_savings_tot_df, total_system_costs_df], axis=0)
    savings_and_costs_df = savings_and_costs_df.T
    plot_comparison_savings(savings_and_costs_df, x="Consumption saving insulation (TWh/year)",
                            y="Consumption saving heater (TWh/year)", save=os.path.join(save_path, f"savings_and_costs.{extension}"),
                            col_for_size="Total costs", smallest_size=smallest_size, biggest_size=biggest_size, fontsize=fontsize,
                            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, unit=unit, coordinates=coordinates)

    savings_and_costs_hp = pd.concat([consumption_savings_tot_df, stock_heat_pump_df, total_system_costs_df], axis=0)
    savings_and_costs_hp = savings_and_costs_hp.T
    plot_comparison_savings(savings_and_costs_hp, x="Consumption saving insulation (TWh/year)",
                            y="Stock Heat pump (Million)", save=os.path.join(save_path, f"savings_and_costs_hp.{extension}"),
                            col_for_size="Total costs", smallest_size=smallest_size, biggest_size=biggest_size, fontsize=fontsize,
                            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, unit=unit, coordinates=coordinates)

    if eoles:
        emissions_tot = pd.concat([emissions_dict[key].rename(key).to_frame() for key in emissions_dict.keys()], axis=1).loc[2050].rename("Emissions (MtCO2)").to_frame().T
        savings_and_emissions_df = pd.concat([consumption_savings_tot_df, emissions_tot], axis=0)
        savings_and_emissions_df = savings_and_emissions_df.T
        plot_comparison_savings(savings_and_emissions_df, x="Consumption saving insulation (TWh/year)",
                                y="Consumption saving heater (TWh/year)", save=os.path.join(save_path, f"savings_and_emissions.{extension}"),
                                col_for_size="Emissions (MtCO2)", smallest_size=100,
                                biggest_size=400, fontsize=18, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, unit=unit, coordinates=coordinates)

    try:  # TODO: a modifier pour les prochains graphes
        if not greenfield:
            # Evolution of peak load
            if save_path is None:
                save_path_plot = None
            else:
                save_path_plot = os.path.join(save_path, f"electricity_peak_load.{extension}")
            make_line_plots(peak_electricity_load_dict, y_label="Electricity peak load (GW)",
                            format_y=lambda y, _: '{:.0f}'.format(y),
                            index_int=True, save=save_path_plot)

        # Evolution of emissions
        if save_path is None:
            save_path_plot = None
        else:
            save_path_plot = os.path.join(save_path, f"CO2_emissions.{extension}")
        make_line_plots(emissions_dict, y_label="Emissions (MtCO2)", format_y=lambda y, _: '{:.0f}'.format(y),
                        index_int=True, save=save_path_plot, y_min=0)

        # Evolution of insulation subsidies
        if save_path is None:
            save_path_plot = None
        else:
            save_path_plot = os.path.join(save_path, f"subsidies_insulation.{extension}")
        make_line_plots(subsidies_insulation_dict, y_label="Subsidies insulation (%)",
                        format_y=lambda y, _: '{:.0f}'.format(y),
                        index_int=True, save=save_path_plot, rotation=45, x_ticks=dataframe_subsidy.index[::2], y_min=0,
                        y_max=100, secondary_y=secondary_y, secondary_axis_spec=secondary_axis_spec)

        # Evolution of heater subsidies
        if save_path is None:
            save_path_plot = None
        else:
            save_path_plot = os.path.join(save_path, f"subsidies_heater.{extension}")
        make_line_plots(subsidies_heater_dict, y_label="Subsidies heater (%)", format_y=lambda y, _: '{:.2f}'.format(y),
                        index_int=True, save=save_path_plot, rotation=45, x_ticks=dataframe_subsidy.index[::2], y_min=0,
                        y_max=100)

        if len(dict_output.keys()) <= 3:
            # Evolution of consumption savings
            if save_path is None:
                save_path_plot = None
            else:
                save_path_plot = os.path.join(save_path, f"evolution_consumption_savings.{extension}")
            stacked_bars(consumption_saving_evolution_dict, y_label="Consumption savings (TWh)",
                         format_y=lambda y, _: '{:.0f}'.format(y),
                         colors=None, x_ticks=None, index_int=True, save=save_path_plot, rotation=0,
                         n=len(consumption_saving_evolution_dict.keys()),
                         dict_legend=DICT_TRANSFORM_LEGEND)

    except:
        pass

    if eoles:
        # Evolution of electricity capacities
        if save_path is None:
            save_path_plot = None
        else:
            save_path_plot = os.path.join(save_path, f"electricity_capacities.{extension}")
        make_line_plots(capacities_vre_dict, y_label="Capacities (GW)", format_y=lambda y, _: '{:.0f}'.format(y),
                        index_int=True, colors=resources_data["colors_eoles"], multiple_legend=True, save=save_path_plot)

        if save_path is None:
            save_path_plot = None
        else:
            save_path_plot = os.path.join(save_path, f"flex_capacities_2050.{extension}")
        make_stacked_bar_plot(capacities_flex_df.loc[2050].set_index("Configuration"), y_label="Flexible capacity (GW)",
                          format_y=lambda y, _: '{:.0f}'.format(y), index_int=False, rotation=90, save=save_path_plot)  # TODO: ajouter les bonnes couleurs ici
        # plot_capacities_barplot_2(capacities_flex_df, save=save_path_plot)  # TODO: a modifier pour que cela fonctionne, ce n'est pas encore bien.

    try:  # TODO: a modifier proprement
        if not pdf:  # only save summary without pdf option
            images_to_save = [os.path.join(save_path, f"total_system_costs.{extension}"),
                              os.path.join(save_path, f"difference_total_system_costs.{extension}"),
                              os.path.join(save_path, f"complete_system_costs_2050.{extension}"),
                              os.path.join(save_path, f"difference_complete_system_costs_2050.{extension}"),
                              os.path.join(save_path, f"consumption_savings.{extension}"),
                              os.path.join(save_path, f"savings_and_costs.{extension}"),
                              os.path.join(save_path, f"savings_and_emissions.{extension}"),
                              os.path.join(save_path, f"electricity_peak_load.{extension}"),
                              os.path.join(save_path, f"CO2_emissions.{extension}"),
                              os.path.join(save_path, f"subsidies_insulation.{extension}"),
                              os.path.join(save_path, f"subsidies_heater.{extension}"),
                              os.path.join(save_path, f"electricity_capacities.{extension}"),
                              ]
            if len(dict_output.keys()) <= 3:
                images_to_save.append(os.path.join(save_path, f"evolution_consumption_savings.{extension}"))

            if greenfield:
                images_to_save = [os.path.join(save_path, f"total_system_costs.{extension}"),
                                  os.path.join(save_path, f"difference_total_system_costs.{extension}"),
                                  os.path.join(save_path, f"complete_system_costs_2050.{extension}"),
                                  os.path.join(save_path, f"difference_complete_system_costs_2050.{extension}"),
                                  os.path.join(save_path, f"consumption_savings.{extension}"),
                                  os.path.join(save_path, f"savings_and_costs.{extension}"),
                                  os.path.join(save_path, f"savings_and_emissions.{extension}")
                                  ]

            images = [Image.open(img) for img in images_to_save]
            new_images = []
            for png in images:
                png.load()
                background = Image.new("RGB", png.size, (255, 255, 255))
                background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
                new_images.append(background)

            pdf_path = os.path.join(save_path, "summary_comparison.pdf")

            new_images[0].save(
                pdf_path, "PDF", resolution=100.0, save_all=True, append_images=new_images[1:]
            )
    except:
        pass

    return total_system_costs_df, consumption_savings_tot_df, complete_system_costs_2050_df


def save_summary_pdf(path):
    """Saves a summary of files as pdf"""
    images_to_save = [os.path.join(path, "plots", "img", "consumption_energy.png"),
                      os.path.join(path, "plots", "img", "consumption_heater.png"),
                      os.path.join(path, "plots", "img", "investment.png"),
                      os.path.join(path, "plots", "img", "renovation_decision_maker.png"),
                      os.path.join(path, "plots", "img", "replacement_insulation.png"),
                      os.path.join(path, "plots", "img", "retrofit_measures.png"),
                      os.path.join(path, "plots", "img", "stock_heater.png"),
                      os.path.join(path, "plots", "img", "stock_performance.png"),
                      os.path.join(path, "plots", "img", "switch_heater.png"),
                      os.path.join(path, "plots", "annualized_system_costs_evolution.png"),
                      os.path.join(path, "plots", "emissions.png"),
                      os.path.join(path, "plots", "prices.png"),
                      os.path.join(path, "plots", "primary_generation.png"),
                      os.path.join(path, "plots", "resirf_subsidies.png"),
                      ]
    images = [Image.open(img) for img in images_to_save]
    new_images = []
    for png in images:
        png.load()
        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
        new_images.append(background)

    pdf_path = os.path.join(path, "summary_pdf.pdf")

    new_images[0].save(
        pdf_path, "PDF", resolution=100.0, save_all=True, append_images=new_images[1:]
    )


def save_subsidies_again(dict_output, save_path):
    for path, name_config in zip(dict_output.values(), [n for n in dict_output.keys()]):
        with open(os.path.join(path, 'coupling_results.pkl'), "rb") as file:
            output = load(file)
            subsidies = output["Subsidies (%)"]
            subsidies = output["Subsidies (%)"]
            dataframe_subsidy_list = [subsidies]
            for i in range(4):
                tmp = subsidies.copy()
                tmp.index += i + 1
                dataframe_subsidy_list.append(tmp)
            dataframe_subsidy = pd.concat(dataframe_subsidy_list, axis=0).sort_index(ascending=True)
            new_subsidies_2020_2024 = pd.concat(
                [subsidies.iloc[0].to_frame().T.rename(index={2025: 2020 + i}) for i in range(5)], axis=0)
            dataframe_subsidy = pd.concat([new_subsidies_2020_2024, dataframe_subsidy], axis=0)
            dataframe_subsidy = pd.concat(
                [dataframe_subsidy, dataframe_subsidy.iloc[-1].to_frame().T.rename(index={2049: 2050})], axis=0)
            # subsidies = pd.concat([subsidies, subsidies.iloc[0].to_frame().T.rename(index={2025: 2020})], axis=0).sort_index()
            dataframe_subsidy["Heater"].to_csv(os.path.join(save_path, f"subsidies_heater_{name_config}.csv"),
                                               header=None)
            dataframe_subsidy["Insulation"].to_csv(os.path.join(save_path, f"subsidies_insulation_{name_config}.csv"),
                                                   header=None)


def process_total_costs(annualized_new_investment_df, annualized_new_energy_capacity_df, functionment_costs_df,
                        carbon_constraint=True, eoles=True):
    """
    Calculates total system (new) costs over the considered time period. This adds investment annuity for each year when the investment was present.
    Remark: we only include new costs, not costs of historical capacities.
    :param annualized_new_investment_df: pd.DataFrame
        Annualized costs of new investment done during the considered year.
    :param annualized_new_energy_capacity_df: pd.DataFrame
        Annualized costs of new energy capacity investment done during the considered year.
    :param functionment_costs_df: pd.DataFrame
        Functionment cost of the system for one year.
    :return:
    """
    annualized_new_investment_df_copy = annualized_new_investment_df.copy()
    annualized_new_energy_capacity_df_copy = annualized_new_energy_capacity_df.copy()
    functionment_costs_df_copy = functionment_costs_df.copy()
    dict_count = {2030: 5 * 5, 2035: 4 * 5, 2040: 3 * 5, 2045: 2 * 5, 2050: 5}
    for col in annualized_new_investment_df_copy.columns:  # attention à vérifier que les colonnes sont des int
        annualized_new_investment_df_copy[col] = annualized_new_investment_df_copy[col] * dict_count[col]
        if not annualized_new_energy_capacity_df_copy.empty:
            annualized_new_energy_capacity_df_copy[col] = annualized_new_energy_capacity_df_copy[col] * dict_count[col]
    functionment_costs_df_copy = functionment_costs_df_copy * 5  # we count each functionment costs 5 times

    if carbon_constraint:
        elec_inv = annualized_new_investment_df_copy.drop(index=["investment_heater",
                                                                 "investment_insulation"]).sum().sum()
        if not annualized_new_energy_capacity_df_copy.empty:
            elec_inv = elec_inv + annualized_new_energy_capacity_df_copy.sum().sum()
        heater_inv = annualized_new_investment_df_copy.T[["investment_heater"]].sum().sum()
        insulation_inv = annualized_new_investment_df_copy.T[["investment_insulation"]].sum().sum()
        functionment_cost = functionment_costs_df_copy.drop(index=["health_costs"]).sum().sum()
        health_costs = functionment_costs_df_copy.T[["health_costs"]].sum().sum()
        if eoles:
            total_costs = elec_inv + heater_inv + insulation_inv + functionment_cost + health_costs
            total_system_costs = pd.Series(
                index=["Investment electricity costs", "Investment heater costs", "Investment insulation costs",
                       "Functionment costs", "Health costs", "Total costs"],
                data=[elec_inv, heater_inv, insulation_inv, functionment_cost, health_costs, total_costs])
        else:  # we only include ResIRF costs
            total_costs = heater_inv + insulation_inv + functionment_cost + health_costs
            total_system_costs = pd.Series(
                index=["Investment heater costs", "Investment insulation costs",
                       "Functionment costs", "Health costs", "Total costs"],
                data=[heater_inv, insulation_inv, functionment_cost, health_costs, total_costs])
    else:  # we have to consider carbon cost in addition to functionment cost
        elec_inv = annualized_new_investment_df_copy.drop(index=["investment_heater",
                                                                 "investment_insulation"]).sum().sum()
        if not annualized_new_energy_capacity_df_copy.empty:
            elec_inv = elec_inv + + annualized_new_energy_capacity_df_copy.sum().sum()
        heater_inv = annualized_new_investment_df_copy.T[["investment_heater"]].sum().sum()
        insulation_inv = annualized_new_investment_df_copy.T[["investment_insulation"]].sum().sum()
        functionment_cost = functionment_costs_df_copy.drop(index=["health_costs", "carbon_cost"]).sum().sum()
        health_costs = functionment_costs_df_copy.T[["health_costs"]].sum().sum()
        carbon_cost = functionment_costs_df_copy.T[["carbon_cost"]].sum().sum()
        if eoles:
            total_costs = elec_inv + heater_inv + insulation_inv + functionment_cost + health_costs + carbon_cost
            total_system_costs = pd.Series(
                index=["Investment electricity costs", "Investment heater costs", "Investment insulation costs",
                       "Functionment costs", "Health costs", "Carbon cost", "Total costs"],
                data=[elec_inv, heater_inv, insulation_inv, functionment_cost, health_costs, carbon_cost, total_costs])
        else:
            total_costs = heater_inv + insulation_inv + functionment_cost + health_costs + carbon_cost
            total_system_costs = pd.Series(
                index=["Investment heater costs", "Investment insulation costs",
                       "Functionment costs", "Health costs", "Carbon cost", "Total costs"],
                data=[heater_inv, insulation_inv, functionment_cost, health_costs, carbon_cost, total_costs])
    return total_system_costs


def process_evolution_annualized_energy_system_cost(annualized_new_investment_df, annualized_new_energy_capacity_df,
                                                    functionment_costs_df,
                                                    historical_capacity_df, historical_energy_capacity_df,
                                                    transport_distribution_costs):
    """Process the evolution of complete energy annualized system costs, in 1e9 €/yr. This includes in particular
    historical costs, and transport and distribution costs. Be careful: cannot be compared directly to RTE outputs, since this includes
    as well the functionment of wood and oil boilers. Otherwise, we do look like RTE. DOES NOT INCLUDE HEATER AND INSULATION INVESTMENT."""
    total_cost = annualized_new_investment_df.drop(
        index=["investment_heater", "investment_insulation"])  # we are only interested in the energy system cost
    total_cost = total_cost.add(annualized_new_energy_capacity_df, fill_value=0)  # we add the value of investments
    for i in range(1, annualized_new_investment_df.shape[
        1]):  # we estimate cumulated costs from new investments which are still active in following years
        total_cost[total_cost.columns[i]] = total_cost[total_cost.columns[i - 1]] + total_cost[total_cost.columns[i]]

    total_cost = total_cost.add(functionment_costs_df.drop(index=["health_costs"]),
                                fill_value=0)  # add functionment cost for each year, and not interested in health costs
    total_cost = total_cost.add(historical_capacity_df,
                                fill_value=0)  # add historical capacity cost present during the considered year
    total_cost = total_cost.add(historical_energy_capacity_df,
                                fill_value=0)  # add historical energy capacity cost present during the considered year
    total_cost = total_cost.add(transport_distribution_costs, fill_value=0)  # add transport and distribution costs
    total_cost = total_cost.sum(axis=0)
    return total_cost.T.squeeze()


def process_evolution_annualized_costs(annualized_new_investment_df, annualized_new_energy_capacity_df,
                                       functionment_costs_df):
    """Evolution of annualized investment costs for the social planner, in 1e9 €/yr."""
    annualized_new_investment_df_copy = annualized_new_investment_df.copy()
    annualized_new_energy_capacity_df_copy = annualized_new_energy_capacity_df.copy()
    functionment_costs_df_copy = functionment_costs_df.copy()
    investment_costs = annualized_new_investment_df_copy
    investment_costs = investment_costs.add(annualized_new_energy_capacity_df_copy,
                                            fill_value=0)  # we add the value of investments

    elec_inv = investment_costs.drop(index=["investment_heater", "investment_insulation"]).sum(axis=0)
    heater_inv = investment_costs.T[["investment_heater"]].squeeze()
    insulation_inv = investment_costs.T[["investment_insulation"]].squeeze()
    functionment_cost = functionment_costs_df_copy.drop(index=["health_costs"]).sum(axis=0)
    health_costs = functionment_costs_df_copy.T[["health_costs"]].squeeze()
    total_cost = investment_costs.add(functionment_costs_df_copy,
                                      fill_value=0)  # add functionment cost for each year, and not interested in health costs
    total_cost = total_cost.sum(axis=0)

    return pd.concat([elec_inv.to_frame().rename(columns={0: "Investment electricity costs"}),
                      heater_inv.to_frame().rename(columns={'investment_heater': "Investment heater costs"}),
                      insulation_inv.to_frame().rename(columns={'investment_insulation': "Investment insulation costs"}),
                      functionment_cost.to_frame().rename(columns={0: "Functionment costs"}),
                      health_costs.to_frame().rename(columns={'health_costs': "Health costs"}),
                      total_cost.to_frame().rename(columns={0: "Total costs"})], axis=1)


def process_complete_system_cost_2050(annualized_new_investment_df, annualized_new_energy_capacity_df,
                                      functionment_costs_df, carbon_constraint=True, eoles=True):
    """

    :param eoles: bool
        If true, classic CBA analysis. If False, we only rely on ResIRF costs, so we get rid of electricity investment costs.
    :return:
    """
    annualized_new_investment_df_copy = annualized_new_investment_df.copy()
    annualized_new_energy_capacity_df_copy = annualized_new_energy_capacity_df.copy()
    functionment_costs_df_copy = functionment_costs_df.copy()
    investment_costs = annualized_new_investment_df_copy
    if not annualized_new_energy_capacity_df.empty:
        investment_costs = investment_costs.add(annualized_new_energy_capacity_df_copy,
                                                fill_value=0)  # we add the value of investments
    for i in range(1, annualized_new_investment_df_copy.shape[1]):  # we estimate cumulated costs from new investments which are still active in following years
        investment_costs[investment_costs.columns[i]] = investment_costs[investment_costs.columns[i - 1]] + \
                                                        investment_costs[investment_costs.columns[i]]

    if carbon_constraint:
        elec_inv = investment_costs.drop(index=["investment_heater", "investment_insulation"]).sum(axis=0)
        heater_inv = investment_costs.T[["investment_heater"]].squeeze()
        insulation_inv = investment_costs.T[["investment_insulation"]].squeeze()
        functionment_cost = functionment_costs_df_copy.drop(index=["health_costs"]).sum(axis=0)
        health_costs = functionment_costs_df_copy.T[["health_costs"]].squeeze()
    else:
        elec_inv = investment_costs.drop(index=["investment_heater", "investment_insulation"]).sum(axis=0)
        heater_inv = investment_costs.T[["investment_heater"]].squeeze()
        insulation_inv = investment_costs.T[["investment_insulation"]].squeeze()
        functionment_cost = functionment_costs_df_copy.drop(index=["carbon_cost", "health_costs"]).sum(axis=0)
        health_costs = functionment_costs_df_copy.T[["health_costs"]].squeeze()
        carbon_cost = functionment_costs_df_copy.T[["carbon_cost"]].squeeze()

    if not isinstance(heater_inv, pd.Series):
        heater_inv = pd.Series(heater_inv, index=list(elec_inv.index))
    if not isinstance(insulation_inv, pd.Series):
        insulation_inv = pd.Series(insulation_inv, index=list(elec_inv.index))
    if not isinstance(health_costs, pd.Series):
        health_costs = pd.Series(health_costs, index=list(elec_inv.index))
    total_cost = investment_costs.add(functionment_costs_df_copy,
                                      fill_value=0)  # sum investment costs and functionment costs for each year
    total_cost = total_cost.sum(axis=0)
    if carbon_constraint:
        if eoles:
            complete_system_costs_df = pd.Series(
            data=[elec_inv.loc[2050], heater_inv.loc[2050], insulation_inv.loc[2050], functionment_cost.loc[2050],
                  health_costs.loc[2050], total_cost.loc[2050]],
            index=["Investment electricity costs", "Investment heater costs", "Investment insulation costs",
                   "Functionment costs", "Health costs", "Total costs"])
        else:
            complete_system_costs_df = pd.Series(
                data=[heater_inv.loc[2050], insulation_inv.loc[2050], functionment_cost.loc[2050],
                      health_costs.loc[2050], total_cost.loc[2050]],
                index=["Investment heater costs", "Investment insulation costs",
                       "Functionment costs", "Health costs", "Total costs"])
    else:
        if eoles:
            complete_system_costs_df = pd.Series(
            data=[elec_inv.loc[2050], heater_inv.loc[2050], insulation_inv.loc[2050], functionment_cost.loc[2050],
                  health_costs.loc[2050], carbon_cost.loc[2050], total_cost.loc[2050]],
            index=["Investment electricity costs", "Investment heater costs", "Investment insulation costs",
                   "Functionment costs", "Health costs", "Carbon cost", "Total costs"])
        else:
            complete_system_costs_df = pd.Series(
            data=[heater_inv.loc[2050], insulation_inv.loc[2050], functionment_cost.loc[2050],
                  health_costs.loc[2050], carbon_cost.loc[2050], total_cost.loc[2050]],
            index=["Investment heater costs", "Investment insulation costs",
                   "Functionment costs", "Health costs", "Carbon cost", "Total costs"])
    return complete_system_costs_df


def plot_simulation(output, save_path):
    capacity_df = output["Capacities (GW)"]
    generation_df = output["Generation (TWh)"]
    primary_generation_df = output["Primary generation (TWh)"]
    conversion_generation_df = output["Conversion generation (TWh)"]
    prices_df = output["Prices (€/MWh)"]
    resirf_subsidies = output["Subsidies (%)"]
    resirf_costs_df = output["ResIRF costs (Billion euro)"]
    resirf_costs_eff_df = output["ResIRF costs eff (euro/kWh)"]
    resirf_consumption_df = output["ResIRF consumption (TWh)"]
    try:
        resirf_consumption_saving_df = output["ResIRF consumption savings (TWh/year)"]
    except:  # old name
        resirf_consumption_saving_df = output["ResIRF consumption savings (TWh)"]
        resirf_consumption_saving_df = resirf_consumption_saving_df.rename(
            columns={"Consumption saving heater (TWh)": "Consumption saving heater (TWh/year)",
                     "Consumption saving insulation (TWh)": "Consumption saving insulation (TWh/year)"})
    resirf_replacement_heater = output["ResIRF replacement heater (Thousand)"]
    resirf_stock_heater = output["ResIRF stock heater (Thousand)"]
    annualized_system_costs = output["Annualized system costs (Billion euro / year)"]
    peak_electricity_load_df = output["Peak electricity load"]
    peak_heat_load_df = output["Peak heat load"]
    emissions = output["Emissions (MtCO2)"]
    annualized_new_investment_df = output["Annualized new investments (1e9€/yr)"]
    annualized_new_energy_capacity_df = output["Annualized costs new energy capacity (1e9€/yr)"]
    functionment_costs_df = output["System functionment (1e9€/yr)"]

    # Plot capacities
    # # Electricity generation capacities
    elec_generation = ["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "nuclear", "ocgt",
                       "ccgt"]
    make_line_plot(capacity_df.T, subset=elec_generation, y_label="Capacity power (GWh)",
                   colors=resources_data["colors_eoles"],
                   save=os.path.join(save_path, "capacities_electricity_generation.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    # # CH4 generation capacities
    CH4_generation = ["methanization", "pyrogazification", "methanation"]
    make_line_plot(capacity_df.T, subset=CH4_generation, y_label="Capacity CH4 (GWh)",
                   colors=resources_data["colors_eoles"],
                   save=os.path.join(save_path, "capacities_CH4_generation.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    # Plot generation
    # # Primary generation
    make_area_plot(primary_generation_df.T, y_label="Primary generation (TWh)", colors=resources_data["colors_eoles"],
                   save=os.path.join(save_path, "primary_generation.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    # # CH4 generation
    # TODO: ajouter une selection
    CH4_generation = ["methanation", "methanization", "pyrogazification", "natural_gas"]
    make_area_plot(generation_df.T, subset=CH4_generation, y_label="CH4 generation (TWh)",
                   colors=resources_data["colors_eoles"],
                   save=os.path.join(save_path, "primary_CH4_generation.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    # # CH4 génération with demand
    df = pd.concat([generation_df.T[CH4_generation], conversion_generation_df.T[["ocgt", "ccgt"]],
                    resirf_consumption_df[["Natural gas"]].rename(columns={'Natural gas': 'Gas for heating'})], axis=1)
    df["ocgt"] = - df["ocgt"]
    df["ccgt"] = - df["ccgt"]
    df['Gas for heating'] = - df['Gas for heating']
    make_area_plot(df, y_label="Gas generation and conversion (TWh)", colors=resources_data["colors_eoles"],
                   save=os.path.join(save_path, "CH4_generation_and_conversion.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    # # Power génération with demand
    df = pd.concat([generation_df.T[elec_generation], conversion_generation_df.T[["electrolysis", "methanation"]],
                    resirf_consumption_df[["Electricity"]].rename(columns={'Electricity': 'Electricity for heating'})],
                   axis=1)
    df["electrolysis"] = - df["electrolysis"]
    df["methanation"] = - df["methanation"]
    df['Electricity for heating'] = - df['Electricity for heating']
    make_area_plot(df, y_label="Power generation and conversion (TWh)", colors=resources_data["colors_eoles"],
                   save=os.path.join(save_path, "power_generation_and_conversion.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    # Plot LCOE and price
    subset_lcoe = ["LCOE electricity", "LCOE electricity volume", "LCOE electricity value", "LCOE CH4", "LCOE CH4 volume",
              "LCOE CH4 value", "LCOE CH4 volume noSCC", "LCOE CH4 noSCC"]
    make_line_plot(prices_df, subset=subset_lcoe, y_label="Prices (€/MWh)", colors=resources_data["colors_eoles"],
                   save=os.path.join(save_path, "prices.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    # Plot subsidies ResIRF
    dataframe_subsidy_list = [resirf_subsidies]
    for i in range(4):
        tmp = resirf_subsidies.copy()
        tmp.index += i + 1
        dataframe_subsidy_list.append(tmp)
    dataframe_subsidy = pd.concat(dataframe_subsidy_list, axis=0).sort_index(ascending=True)
    dataframe_subsidy *= 100  # we write in percentage
    make_line_plot(dataframe_subsidy, y_label="Subsidies (%)", save=os.path.join(save_path, "resirf_subsidies.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y),
                   rotation=45, x_ticks=dataframe_subsidy.index[::2])

    # make_line_plot(dataframe_subsidy, y_label="Subsidies (%)", save=os.path.join(save_path, "resirf_subsidies.png"))

    # Plots no longer needed (use of ResIRF plots)
    # # Plot consumption ResIRF
    # make_area_plot(resirf_consumption_df, subset=["Electricity", "Natural gas", "Oil fuel", "Wood fuel"],
    #                y_label="Heating consumption (TWh)", colors=resources_data["colors_resirf"],
    #                save=os.path.join(save_path, "resirf_consumption.png"), format_y=lambda y, _: '{:.0f}'.format(y))
    #
    # try:  # TODO: ajout temporaire pour gérer un problème de savings négatif
    #     # Plot savings ResIRF
    #     make_area_plot(resirf_consumption_saving_df, y_label="Consumption savings (TWh)",
    #                    save=os.path.join(save_path, "resirf_savings.png"), format_y=lambda y, _: '{:.0f}'.format(y),
    #                    rotation=45, x_ticks=resirf_consumption_saving_df.index[::2])
    #
    #     # Unique plot consumption + savings
    #     resirf_consumption_saving_df = resirf_consumption_saving_df.reset_index().rename(columns={'index': 'year'})
    #     resirf_consumption_saving_df["Consumption saving heater cumulated (TWh)"] = resirf_consumption_saving_df["Consumption saving heater (TWh/year)"].cumsum()
    #     resirf_consumption_saving_df["Consumption saving insulation cumulated (TWh)"] = resirf_consumption_saving_df[
    #         "Consumption saving insulation (TWh/year)"].cumsum()
    #     resirf_consumption_saving_df = resirf_consumption_saving_df.loc[resirf_consumption_saving_df["year"] % 5 == 4]  # we only keep rows like 2029, 2034, etc...
    #     resirf_consumption_saving_df["year"] = resirf_consumption_saving_df["year"] + 1  # we modify the year
    #     resirf_consumption_saving_df = resirf_consumption_saving_df.set_index("year")
    #     # resirf_consumption_saving_df["period"] = resirf_consumption_saving_df.apply(lambda row: (row["year"] - 2025) // 5, axis=1)
    #     # resirf_consumption_saving_df = resirf_consumption_saving_df.groupby("period").agg(
    #     #     {"year": np.min, "Consumption saving heater (TWh)": np.sum, "Consumption saving insulation (TWh)": np.sum}).set_index("year")
    #
    #     resirf_consumption_total_df = pd.concat([resirf_consumption_df, resirf_consumption_saving_df[["Consumption saving heater cumulated (TWh)", "Consumption saving insulation cumulated (TWh)"]]], axis=1)
    #     make_area_plot(resirf_consumption_total_df, y_label="Consumption and savings (TWh)", colors=resources_data["colors_resirf"],
    #                    save=os.path.join(save_path, "resirf_consumption_savings.png"), format_y=lambda y, _: '{:.0f}'.format(y))
    # except:
    #     pass
    #
    # # Plot costs ResIRF
    # make_line_plot(resirf_costs_df, y_label="Costs (Billion euro)", save=os.path.join(save_path, "resirf_costs.png"),
    #                colors=resources_data["colors_eoles"],
    #                format_y=lambda y, _: '{:.0f}'.format(y), rotation=45, x_ticks=resirf_costs_df.index[::2])
    #
    # try:
    #     if "ResIRF costs eff (euro/kWh)" in output.keys():
    #         resirf_costs_eff_df = output["ResIRF costs eff (euro/kWh)"]
    #         make_line_plot(resirf_costs_eff_df, y_label="Costs per saving (euro/kWh)", save=os.path.join(save_path, "resirf_costs_eff.png"),
    #                        format_y=lambda y, _: '{:.2f}'.format(y), rotation=45, x_ticks=resirf_costs_eff_df.index[::2])
    # except:
    #     pass
    #
    # # Plot stock and replacement ResIRF
    # make_area_plot(resirf_replacement_heater, y_label="Replacement heater (Thousand households)",
    #                save=os.path.join(save_path, "resirf_replacement.png"), format_y=lambda y, _: '{:.0f}'.format(y),
    #                rotation=45, x_ticks=resirf_replacement_heater.index[::2])
    #
    # resirf_stock_heater = resirf_stock_heater.T
    # resirf_stock_heater["Heat pump"] = resirf_stock_heater["Stock Heat pump (Million)"]
    # resirf_stock_heater["Electric heating"] = resirf_stock_heater["Stock Direct electric (Million)"]
    # resirf_stock_heater["Natural gas"] = resirf_stock_heater["Stock Natural gas (Million)"]
    # resirf_stock_heater["Oil fuel"] = resirf_stock_heater[ "Stock Oil fuel (Million)"]
    # resirf_stock_heater["Wood fuel"] = resirf_stock_heater["Stock Wood fuel (Million)"]
    # resirf_stock_heater.index += 1  # we increase the index to get the correct years
    # make_area_plot(resirf_stock_heater[["Heat pump", "Electric heating", "Natural gas", "Oil fuel", "Wood fuel"]], y_label="Stock heater (Million)",
    #                save=os.path.join(save_path, "resirf_stock.png"), format_y=lambda y, _: '{:.0f}'.format(y), colors=resources_data["colors_eoles"])

    # Plot annualized system costs
    make_line_plot(annualized_system_costs, y_label="Annualized system costs (Md€ / year)",
                   save=os.path.join(save_path, "annualized_system_costs.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    subset_annualized_costs = ["Annualized electricity system costs", "Annualized investment heater costs",
                               "Annualized investment insulation costs", "Annualized health costs"]
    make_area_plot(annualized_system_costs, subset=subset_annualized_costs,
                   y_label="Annualized system costs (Md€ / year)",
                   save=os.path.join(save_path, "annualized_system_costs_area.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y),
                   colors=resources_data["colors_eoles"])

    # Plot stacked bar plot costs
    evolution_annualized_system_costs = process_evolution_annualized_costs(annualized_new_investment_df, annualized_new_energy_capacity_df, functionment_costs_df)
    subset_complete_costs = ["Investment electricity costs", "Investment heater costs",
                             "Investment insulation costs", "Functionment costs", "Health costs"]
    make_stacked_bar_plot(evolution_annualized_system_costs, subset=subset_complete_costs,
                          y_label="Evolution annualized system costs in 2050 (Md€/year)",
                          colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.01f}'.format(y),
                          index_int=True, rotation=0, dict_legend=DICT_TRANSFORM_LEGEND, save=os.path.join(save_path, "annualized_system_costs_evolution.png"))

    # Peak load
    peak_electricity_load = peak_electricity_load_df[["peak_electricity_load", "year"]].groupby(["year"]).mean()
    make_line_plot(peak_electricity_load, y_label="Electricity peak load (GW)",
                   save=os.path.join(save_path, "peak_load_electricity.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    peak_heat_load = peak_heat_load_df[["heat_elec", "year"]].groupby(["year"]).mean()
    make_line_plot(peak_heat_load, y_label="Heat peak load (GW)", save=os.path.join(save_path, "peak_load_heat.png"),
                   format_y=lambda y, _: '{:.2f}'.format(y))

    # Emissions
    make_line_plot(emissions, y_label="Emissions (MtCO2)", save=os.path.join(save_path, "emissions.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y), y_min=0)


def plot_investment_trajectory(resirf_costs_df, save=None):
    if save is None:
        fig, ax = plt.subplots(1, 1)
        save_path = None
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
        save_path = os.path.join(save, "investment_subsidies_heater.png")
    resirf_costs_df[["Investment heater (Billion euro)"]].plot.line(ax=ax, color=resources_data["colors_eoles"])
    resirf_costs_df[["Subsidies heater (Billion euro)"]].plot.line(ax=ax, color=resources_data["colors_eoles"],
                                                                   style="--")

    ax = format_ax(ax, title="Investments heater (Billion euro)", x_ticks=resirf_costs_df.index[::2],
                   format_y=lambda y, _: '{:.0f}'.format(y), rotation=45)
    format_legend(ax, dict_legend=DICT_TRANSFORM_LEGEND)
    save_fig(fig, save=save_path)

    if save is None:
        fig, ax = plt.subplots(1, 1)
        save_path = None
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
        save_path = os.path.join(save, "investment_subsidies_insulation.png")
    resirf_costs_df[["Investment insulation (Billion euro)"]].plot.line(ax=ax, color=resources_data["colors_eoles"])
    resirf_costs_df[["Subsidies insulation (Billion euro)"]].plot.line(ax=ax, color=resources_data["colors_eoles"],
                                                                       style="--")
    ax = format_ax(ax, title="Investments insulation (Billion euro)", x_ticks=resirf_costs_df.index[::2],
                   format_y=lambda y, _: '{:.0f}'.format(y), rotation=45)
    format_legend(ax, dict_legend=DICT_TRANSFORM_LEGEND)
    save_fig(fig, save=save_path)


def plot_residual_demand(hourly_generation, date_start, date_end, climate=2006, save_path=None,
                      y_min=None, y_max=None, x_min=None, x_max=None):
    """This plot allows to compare electricity demand, including electrolysis and methanation, and residual demand, where
    fatal production is substracted from overall demand."""
    # TODO: il faut modifier le graphe pour obtenir un graphe comme celui p.103 du chapitre 3 de RTE (il faut a priori juste décaler)
    hourly_generation_subset = hourly_generation.copy()
    hourly_generation_subset["date"] = hourly_generation_subset.apply(
        lambda row: datetime.datetime(climate, 1, 1, 0) + datetime.timedelta(hours=row["hour"]),
        axis=1)
    hourly_generation_subset = hourly_generation_subset.set_index("date")

    hourly_generation_subset = hourly_generation_subset.loc[date_start: date_end, :]  # select week of interest

    hourly_generation_subset["pv"] = hourly_generation_subset["pv_g"] + hourly_generation_subset["pv_c"]
    hourly_generation_subset["wind"] = hourly_generation_subset["onshore"] + hourly_generation_subset["offshore_f"] + \
                                       hourly_generation_subset["offshore_g"]
    hourly_generation_subset["hydro"] = hourly_generation_subset["river"] + hourly_generation_subset["lake"]

    # TODO: a updater quand je ne travaillerai qu'avec les nouvelles versions de code
    if "electrolysis_elec" in hourly_generation_subset.columns:  # we make two subcases, as the code changed, and i still want to process older versions of dataframes
        hourly_generation_subset["electrolysis"] = hourly_generation_subset["electrolysis_elec"]  # we consider the electricity used by electrolysis !
    if "methanation_elec" in hourly_generation_subset.columns:
        hourly_generation_subset["methanation"] = hourly_generation_subset["methanation_elec"]  # similarly for methanation

    hourly_generation_subset["total_electricity_demand"] = hourly_generation_subset["elec_demand"] + hourly_generation_subset["electrolysis"] + hourly_generation_subset["methanation"]

    hourly_generation_subset["residual_demand"] = hourly_generation_subset["elec_demand"] - \
                                (hourly_generation_subset["pv"] + hourly_generation_subset["wind"] + hourly_generation_subset["hydro"])

    prod = hourly_generation_subset[["pv", "wind", "hydro"]]
    elec_demand = hourly_generation_subset[["elec_demand"]].squeeze()
    total_elec_demand = hourly_generation_subset[["total_electricity_demand"]].squeeze()
    residual_demand = hourly_generation_subset[["residual_demand"]].squeeze()
    if save_path is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))

    prod.plot.area(color=resources_data["colors_eoles"], ax=ax, linewidth=0)
    elec_demand.plot(ax=ax, style='-', c='red')
    total_elec_demand.plot(ax=ax, style='-', c='black')
    residual_demand.plot(ax=ax, style='--', c='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_title("Residual demand (GW)", loc='left', color='black')
    ax.set_xlabel('')
    if y_min is not None:
        ax.set_ylim(ymin=y_min)
    if y_max is not None:
        ax.set_ylim(ymax=y_max)
    if x_min is not None:
        ax.set_xlim(xmin=x_min)
    if x_max is not None:
        ax.set_xlim(xmax=x_max)
    format_legend(ax)
    plt.axhline(y=0)

    save_fig(fig, save=save_path)

def plot_typical_demand(hourly_generation, date_start, date_end, climate=2006, save_path=None,
                      y_min=None, y_max=None, x_min=None, x_max=None):
    hourly_generation_subset = hourly_generation.copy()
    hourly_generation_subset["date"] = hourly_generation_subset.apply(
        lambda row: datetime.datetime(climate, 1, 1, 0) + datetime.timedelta(hours=row["hour"]),
        axis=1)
    hourly_generation_subset = hourly_generation_subset.set_index("date")

    hourly_generation_subset = hourly_generation_subset.loc[date_start: date_end, :]  # select week of interest
    hourly_generation_subset["electricity demand"] = hourly_generation_subset["elec_demand"]

    # TODO: a updater quand je ne travaillerai qu'avec les nouvelles versions de code
    if "electrolysis_elec" in hourly_generation_subset.columns:  # we make two subcases, as the code changed, and i still want to process older versions of dataframes
        hourly_generation_subset["electrolysis"] = hourly_generation_subset["electrolysis_elec"]  # we consider the electricity used by electrolysis !
    if "methanation_elec" in hourly_generation_subset.columns:
        hourly_generation_subset["methanation"] = hourly_generation_subset["methanation_elec"]  # similarly for methanation

    demand = hourly_generation_subset[["electricity demand", "electrolysis", "methanation"]]

    if save_path is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))

    demand.plot.area(color=resources_data["colors_eoles"], ax=ax, linewidth=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_title("Hourly demand (GW)", loc='left', color='black')
    ax.set_xlabel('')
    if y_min is not None:
        ax.set_ylim(ymin=y_min)
    if y_max is not None:
        ax.set_ylim(ymax=y_max)
    if x_min is not None:
        ax.set_xlim(xmin=x_min)
    if x_max is not None:
        ax.set_xlim(xmax=x_max)
    format_legend(ax)
    plt.axhline(y=0)

    save_fig(fig, save=save_path)

def plot_typical_week(hourly_generation, date_start, date_end, climate=2006, methane=True, save_path=None,
                      y_min=None, y_max=None, x_min=None, x_max=None):
    hourly_generation_subset = hourly_generation.copy()
    hourly_generation_subset["date"] = hourly_generation_subset.apply(
        lambda row: datetime.datetime(climate, 1, 1, 0) + datetime.timedelta(hours=row["hour"]),
        axis=1)
    hourly_generation_subset = hourly_generation_subset.set_index("date")

    hourly_generation_subset = hourly_generation_subset.loc[date_start: date_end, :]  # select week of interest

    hourly_generation_subset["pv"] = hourly_generation_subset["pv_g"] + hourly_generation_subset["pv_c"]
    hourly_generation_subset["wind"] = hourly_generation_subset["onshore"] + hourly_generation_subset["offshore_f"] + \
                                       hourly_generation_subset["offshore_g"]
    hourly_generation_subset["hydro"] = hourly_generation_subset["river"] + hourly_generation_subset["lake"]
    hourly_generation_subset["battery charging"] = - hourly_generation_subset["battery1_in"] - hourly_generation_subset[
        "battery4_in"]
    hourly_generation_subset["battery discharging"] = hourly_generation_subset["battery1"] + hourly_generation_subset[
        "battery4"]
    hourly_generation_subset["phs charging"] = - hourly_generation_subset["phs_in"]
    hourly_generation_subset["phs discharging"] = hourly_generation_subset["phs"]
    if "electrolysis_elec" in hourly_generation_subset.columns:  # we make two subcases, as the code changed, and i still want to process older versions of dataframes
        hourly_generation_subset["electrolysis"] = - hourly_generation_subset["electrolysis_elec"]  # we consider the electricity used by electrolysis !
    else:
        hourly_generation_subset["electrolysis"] = - hourly_generation_subset["electrolysis"]
    if "methanation_elec" in hourly_generation_subset.columns:
        hourly_generation_subset["methanation"] = - hourly_generation_subset["methanation_elec"]  # similarly for methanation
    else:
        hourly_generation_subset["methanation"] = - hourly_generation_subset["methanation"]
    hourly_generation_subset["peaking plants"] = hourly_generation_subset["ocgt"] + hourly_generation_subset["ccgt"] + \
                                                 hourly_generation_subset["h2_ccgt"]
    if methane:
        prod = hourly_generation_subset[
            ["nuclear", "wind", "pv", "hydro", "battery charging", "battery discharging", "phs discharging", "phs charging", "peaking plants",
             "electrolysis", "methanation", "methane"]]
    else:
        prod = hourly_generation_subset[
            ["nuclear", "wind", "pv", "hydro", "battery charging", "battery discharging", "phs discharging", "phs charging", "peaking plants",
             "electrolysis", "methanation"]]
    elec_demand = hourly_generation_subset[["elec_demand"]].squeeze()

    if save_path is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))

    prod.plot.area(color=resources_data["colors_eoles"], ax=ax, linewidth=0)
    elec_demand.plot(ax=ax, style='-', c='red')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_title("Hourly production and demand (GW)", loc='left', color='black')
    ax.set_xlabel('')
    if y_min is not None:
        ax.set_ylim(ymin=y_min)
    if y_max is not None:
        ax.set_ylim(ymax=y_max)
    if x_min is not None:
        ax.set_xlim(xmin=x_min)
    if x_max is not None:
        ax.set_xlim(xmax=x_max)
    # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    # plt.gcf().autofmt_xdate()
    # ax = format_ax(ax, title="Hourly demand and production (GWh)", format_y=lambda y, _: '{:.0f}'.format(y),
    #               x_ticks=prod.index[::12])
    format_legend(ax)
    plt.axhline(y=0)

    save_fig(fig, save=save_path)


def format_legend(ax, dict_legend=None):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    if dict_legend is not None:
        current_labels = ax.get_legend_handles_labels()[1]
        new_labels = [dict_legend[e] if e in dict_legend.keys() else e for e in current_labels]
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), labels=new_labels, frameon=False)
    else:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)


def format_legend_multiple(ax, d, n_style, n_color):
    """Format legend when we want multiple legends for color and linestyle."""
    dummy_lines_color = []
    labels_color = []
    labels_style = []
    dummy_lines_style = []
    for i, l in enumerate(ax.get_lines()):
        if i % n_color == 1:
            dummy_lines_style.append(l)
            labels_style.append(list(d.keys())[i // n_color])
        if i // n_color == 0:
            dummy_lines_color.append(l)
            labels_color.append(l._label)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    legend_style = plt.legend(dummy_lines_style, labels_style, loc='upper left', bbox_to_anchor=(1, 0.5), frameon=False)
    legend_color = plt.legend(dummy_lines_color, labels_color, loc='lower left', bbox_to_anchor=(1, 0.5), frameon=False)
    ax.add_artist(legend_style)
    ax.add_artist(legend_color)


def format_y_ax(ax, title=None, format_y=lambda y, _: y, y_min=None, y_max=None, loc_title=None, c_title=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
    if y_min is not None:
        ax.set_ylim(ymin=y_min)
    if y_max is not None:
        ax.set_ylim(ymax=y_max)
    if title is not None:
        if loc_title is not None:
            ax.set_title(title, loc=loc_title, color=c_title)
        else:
            ax.set_title(title)

    return ax


def format_ax(ax: plt.Axes, title=None, y_label=None, x_label=None, x_ticks=None, format_y=lambda y, _: y, format_x=lambda x, _: x,
              rotation=None, y_min=None, y_max=None, x_min=None, x_max=None, loc_title=None, loc_xlabel=None, c_title=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_x))
    if x_ticks is not None:
        ax.set_xticks(ticks=x_ticks, labels=x_ticks)
    if rotation is not None:
        ax.set_xticklabels(ax.get_xticks(), rotation=rotation)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if x_label is not None:
        if loc_xlabel is not None:
            ax.set_xlabel(x_label, loc=loc_xlabel)
        else:
            ax.set_xlabel(x_label)

    if y_min is not None:
        ax.set_ylim(ymin=y_min)
    if y_max is not None:
        ax.set_ylim(ymax=y_max)
    if x_min is not None:
        ax.set_xlim(xmin=x_min)
    if x_max is not None:
        ax.set_xlim(xmax=x_max)

    if title is not None:
        if loc_title is not None:
            ax.set_title(title, loc=loc_title, color=c_title)
        else:
            ax.set_title(title)

    return ax


def format_ax_string(ax: plt.Axes, title=None, y_label=None, x_label=None, x_ticks_labels=None, format_y=lambda y, _: y,
                     rotation=None, dict_labels=None, loc_title=None, c_title=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
    if dict_labels is not None:
        x_ticks_labels = [dict_labels[e] if e in dict_labels.keys() else e for e in x_ticks_labels]

    ax.set_xticks(ticks=range(len(x_ticks_labels)), labels=x_ticks_labels, rotation=rotation)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if title is not None:
        if loc_title is not None:
            ax.set_title(title, loc=loc_title, color=c_title)
        else:
            ax.set_title(title)

    return ax


def make_area_plot(df, subset=None, y_label=None, colors=None, format_y=lambda y, _: y, save=None, rotation=None,
                   x_ticks=None, dict_legend=None):
    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    df.index = df.index.astype(int)
    if subset is None:
        if colors is None:
            df.plot.area(ax=ax)
        else:
            df.plot.area(ax=ax, color=colors)
    else:
        if colors is None:
            df[subset].plot.area(ax=ax)
        else:
            df[subset].plot.area(ax=ax, color=colors)

    if x_ticks is None:
        ax = format_ax(ax, title=y_label, x_ticks=df.index, format_y=format_y, rotation=rotation)
    else:
        ax = format_ax(ax, title=y_label, x_ticks=x_ticks, format_y=format_y, rotation=rotation)

    format_legend(ax, dict_legend=dict_legend)

    save_fig(fig, save=save)


def make_stacked_investment_plot(df, y_label, subset, scatter, save, colors, format_y, rotation, dict_legend, dict_xlabels):
    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    df[subset].plot(kind='bar', stacked=True, color=colors, ax=ax)
    scatter.plot(ax=ax, style='.', c='black', ms=20)
    ax = format_ax_string(ax, title=y_label, x_ticks_labels=df.index, format_y=format_y, rotation=rotation, dict_labels=dict_xlabels,
                          loc_title='left', c_title='black')
    ax.spines['bottom'].set_visible(False)
    format_legend(ax, dict_legend=dict_legend)
    plt.axhline(y=0, c='black')
    save_fig(fig, save=save)


def make_stacked_bar_plot(df, y_label=None, subset=None, colors=None, format_y=lambda y, _: y, save=None, rotation=None,
                          index_int=True, dict_legend=None, hline=False):
    """The index of the dataframe should correspond to the variable which will be displayed on the x axis of the plot."""
    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    if index_int:
        df.index = df.index.astype(int)

    if subset is not None:
        if colors is None:
            df[subset].plot(kind='bar', stacked=True, ax=ax, linewidth=0)
        else:
            df[subset].plot(kind='bar', stacked=True, color=colors, ax=ax, linewidth=0)
    else:
        if colors is None:
            df.plot(kind='bar', stacked=True, ax=ax, linewidth=0)
        else:
            df.plot(kind='bar', stacked=True, color=colors, ax=ax, linewidth=0)

    if index_int:
        ax = format_ax(ax, title=y_label, format_y=format_y)
        ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=rotation)
    else:  # index is not an int
        ax = format_ax_string(ax, title=y_label, x_ticks_labels=df.index, format_y=format_y, rotation=rotation)

    format_legend(ax, dict_legend=dict_legend)

    if hline:
        plt.axhline(y=0)

    save_fig(fig, save=save)


def stacked_bars(dict_df, y_label, format_y=lambda y, _: y, colors=None, x_ticks=None, index_int=True, save=None,
                 rotation=None,
                 dict_legend=None, n=2):
    """Plots stacked bars for different simulations. Allowed number of simulations to be compared: 2 or 3"""
    if n == 2:
        # list_position = [1.1, 0]
        list_position = [0, 1]
        list_width = [0.3, 0.3]
        list_hatch = [None, ".."]
    else:  # n=3
        # list_position = [1.6, 0.5, -0.6]
        list_position = [0, 0.5, 1]
        list_width = [0.1, 0.1, 0.1]
        list_hatch = [None, "//", ".."]

    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    for i, (key, df) in enumerate(dict_df.items()):
        # df = df.rename(key)
        if index_int:
            df.index = df.index.astype(int)
        if colors is None:
            if i == 0:
                df.plot(kind='bar', stacked=True, ax=ax, position=list_position[i], width=list_width[i],
                        hatch=list_hatch[i], align="center")
            else:
                df.plot(kind='bar', stacked=True, ax=ax, position=list_position[i], width=list_width[i],
                        hatch=list_hatch[i], legend=False, align="center")
        else:
            df.plot(kind='bar', stacked=True, ax=ax, color=colors, position=list_position[i], width=list_width[i],
                    hatch=list_hatch[i], align="center")

        # if i == 0:
    ax = format_ax(ax, title=y_label, format_y=format_y)
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=rotation)
    format_legend(ax, dict_legend=dict_legend)

    save_fig(fig, save=save)


def make_line_plot(df, subset=None, y_label=None, colors=None, format_y=lambda y, _: y, save=None, rotation=None,
                   x_ticks=None, index_int=True, str=False, dict_legend=None, y_min=None):
    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    if index_int:
        df.index = df.index.astype(int)
    if subset is None:
        if colors is None:
            df.plot.line(ax=ax)
        else:
            df.plot.line(ax=ax, color=colors)
    else:
        if colors is None:
            df[subset].plot.line(ax=ax)
        else:
            df[subset].plot.line(ax=ax, color=colors)

    if not str:  # index is an int, not a string
        if x_ticks is None:
            ax = format_ax(ax, title=y_label, x_ticks=df.index, format_y=format_y, rotation=rotation, y_min=y_min)
        else:
            ax = format_ax(ax, title=y_label, x_ticks=x_ticks, format_y=format_y, rotation=rotation, y_min=y_min)
    else:
        ax = format_ax_string(ax, title=y_label, x_ticks_labels=df.index, format_y=format_y, rotation=rotation)
    format_legend(ax, dict_legend=dict_legend)

    save_fig(fig, save=save)


def make_line_plots(dict_df, y_label, format_y=lambda y, _: y, colors=None, x_ticks=None, index_int=True, save=None,
                    rotation=None, multiple_legend=False, y_min=None, y_max=None, secondary_y=None, secondary_axis_spec=None):
    """Make line plot by combining different scenarios."""
    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))

    secondary_ax = None
    for i, (key, df) in enumerate(dict_df.items()):
        if isinstance(df, pd.Series):
            df = df.rename(key)
        if index_int:
            df.index = df.index.astype(int)

        if secondary_y is not None and key == secondary_y:
            secondary_ax = ax.twinx()
            if colors is None:
                df.plot.line(ax=secondary_ax, style=STYLES[i])
            else:
                df.plot.line(ax=secondary_ax, color=colors, style=STYLES[i])
        else:
            if colors is None:
                df.plot.line(ax=ax, style=STYLES[i])
            else:
                df.plot.line(ax=ax, color=colors, style=STYLES[i])

    if x_ticks is None:
        ax = format_ax(ax, title=y_label, x_ticks=df.index, format_y=format_y, rotation=rotation, y_min=y_min,
                       y_max=y_max, loc_title='left', c_title='black')
    else:
        ax = format_ax(ax, title=y_label, x_ticks=x_ticks, format_y=format_y, rotation=rotation, y_min=y_min,
                       y_max=y_max, loc_title='left', c_title='black')

    if secondary_y is not None and secondary_ax is not None:
        y_min, y_max, title = None, None, None
        if secondary_axis_spec is not None:
            if 'y_min' in secondary_axis_spec.keys():
                y_min = secondary_axis_spec['y_min']
            if 'y_max' in secondary_axis_spec.keys():
                y_max = secondary_axis_spec['y_max']
            if 'title' in secondary_axis_spec.keys():
                title = secondary_axis_spec['title']
        secondary_ax = format_y_ax(secondary_ax, title=title, format_y=format_y,
                                 y_min=y_min, y_max=y_max, loc_title='right', c_title='black')

    if not multiple_legend:
        format_legend(ax)
    else:
        format_legend_multiple(ax, dict_df, n_style=len(list(dict_df.keys())), n_color=df.shape[1])

    if secondary_y is not None and secondary_ax is not None:
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = secondary_ax.get_legend_handles_labels()
        merged_handles = handles1 + handles2
        merged_labels = labels1 + labels2
        ax.legend(merged_handles, merged_labels, loc='center left', bbox_to_anchor=(1.2, 0.5), frameon=False)

    save_fig(fig, save=save)


def make_area_plot_multiple(df1, df2, y_label=None, colors=None, format_y=lambda y, _: y, save=None, rotation=None):
    """In progress: pour ajouter des """
    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    df1.index = df1.index.astype(int)
    df2.index = df2.index.astype(int)
    if colors is None:
        df1.plot.area(ax=ax)
    else:
        df1.plot.area(ax=ax, color=colors)

    ax = format_ax(ax, title=y_label, x_ticks=df1.index, format_y=format_y, rotation=rotation)

    format_legend(ax)

    save_fig(fig, save=save)


def plot_blackbox_optimization(dict_optimizer, save_path, two_stage_optim=False):
    if not two_stage_optim:
        for key in dict_optimizer.keys():  # save evaluations
            optimizer = dict_optimizer[key]
            optimizer.save_evaluations(os.path.join(save_path, f'evaluations_optimizer_{key}.csv'))
            optimizer.save_report(os.path.join(save_path, f'report_optimizer_{key}.txt'))
        for key in dict_optimizer.keys():  # save plots
            optimizer = dict_optimizer[key]
            optimizer.plot_convergence(filename=os.path.join(save_path, "plots", f"optimizer_{key}_convergence.png"))
            optimizer.plot_acquisition(filename=os.path.join(save_path, "plots", f"optimizer_{key}_acquisition.png"))
    else:  # we have two successive optimizer to refine the optimum
        for key in dict_optimizer.keys():  # save evaluations
            for stage in dict_optimizer[key].keys():
                optimizer = dict_optimizer[key][stage]
                optimizer.save_evaluations(os.path.join(save_path, f'evaluations_optimizer_{key}_{stage}.csv'))
                optimizer.save_report(os.path.join(save_path, f'report_optimizer_{key}_{stage}.txt'))
        for key in dict_optimizer.keys():  # save plots
            for stage in dict_optimizer[key].keys():
                optimizer = dict_optimizer[key][stage]
                optimizer.plot_convergence(filename=os.path.join(save_path, "plots", f"optimizer_{key}_{stage}_convergence.png"))
                optimizer.plot_acquisition(filename=os.path.join(save_path, "plots", f"optimizer_{key}_{stage}_acquisition.png"))
