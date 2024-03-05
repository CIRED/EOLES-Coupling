import json

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from  matplotlib.colors import LinearSegmentedColormap
import os
from os import listdir
import seaborn as sns
import numpy as np
from pickle import load
import datetime
from PIL import Image
from pathlib import Path
from itertools import product

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
    "Investment heater costs": "Investment switch heater",
    "Investment insulation costs": "Investment insulation",
    "Carbon cost": "Carbon cost",
    "Health costs": "Health costs",
    "Total costs HC excluded": "Total system costs (Billion €)",
    "Total costs": "Total system costs (Billion €)",
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
    "onshore": "Onshore",
    "offshore": "Offshore",
    "pv": "Solar PV",
    "nuclear": "Nuclear",
    "hydro": "Hydroelectricity",
    "battery": "Battery",
    "peaking plants": "Peaking plants",
    "pv_g": "pv ground",
    "pv_c": "pv large roof",
    "river": "river",
    "lake": "lake",
    "methanization": "methanization",
    "pyrogazification": "pyrogazification",
    "natural_gas": "natural gas"
}

DICT_LEGEND_WATERFALL = {
    "Investment electricity costs": "Investment \nenergy mix",
    "Functionment costs": "Energy \noperational costs",
    "Investment heater costs": "Investment \nheating system",
    "Investment insulation costs": "Investment \ninsulation",
    "Health costs": "Health costs",
    'Total costs': 'Total costs',
    'offshore': 'Offshore',
    'onshore': 'Onshore',
    'pv': 'Solar PV',
    'battery': 'Battery',
    'hydro': 'Hydroelectricity',
    'peaking plants': 'Peaking Plants',
    'Generation offshore (TWh)': 'Offshore',
    'Generation onshore (TWh)': 'Onshore',
    'Generation pv (TWh)': 'Solar PV',
    'Generation battery (TWh)': 'Battery',
    'Generation hydro (TWh)': 'Hydroelectricity',
    'Generation nuclear (TWh)': 'Nuclear',
    'Generation natural gas (TWh)': 'Natural Gas',
    'Generation peaking plants (TWh)': 'Peaking Plants',
    'Generation methanization (TWh)': 'Methanization',
    'Generation pyrogazification (TWh)': 'Pyrogazification',
    'Consumption Oil (TWh)': 'Oil Fuel',
    'Consumption Wood (TWh)': 'Wood Fuel'
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


def colormap_simulations(overall_folder, config_ref, save_path=None, pdf=False, carbon_constraint=True, eoles=True,
                         subset_configs=None, percent=False, reorder=None, dict_scenario=None, dict_config_demandsupply=None):
    """
    Processes the total system costs for the different simulations and saves the colormap figure
    :param overall_folder: str
        Path to the folder containing the different subfolders corresponding to the different simulations
    :param config_ref: dict
        Dictionary providing the default values for the configuration
    :param save_path: str
        Path to the folder where the figure will be saved
    :param pdf: bool
        If True, the figure is saved in pdf format, otherwise in png format
    :param carbon_constraint: bool
        If True, the carbon constraint is taken into account
    :param eoles: bool
        If True, EOLES model was used in the simulations and the results are processed accordingly
    :return:
    """
    if pdf:
        extension = "pdf"
    else:
        extension = "png"
    # if subset_configs is None:
    #     subset_configs = []
    total_system_costs_2050_df, complete_system_costs_2050_df = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)
    if save_path is not None:
        save_path = Path(save_path)
        if not save_path.is_dir():  # create directory
            os.mkdir(save_path)

    overall_folder = Path(overall_folder)
    for subfolder in os.listdir(overall_folder):
        if (Path(overall_folder) / Path(subfolder)).is_dir():
            subfolder_names = subfolder.split('_')
            date, scenario, configuration = subfolder_names[0], subfolder_names[-1][6:], '_'.join(subfolder_names[1:-1])

            if (subset_configs is not None) and (scenario not in subset_configs):
                pass
            else:
                # change name for Reference configuration
                tmp = configuration.split('_')  # get values for different parameters
                config = {}
                new_configuration = ''
                for e in tmp:
                    if config_ref is not None:
                        for k in config_ref.keys():
                            if k in e:
                                config[k] = e[len(k):]
                                if e[len(k):] != config_ref[k]:  # we only keep the parameters that are different from the reference configuration
                                    new_configuration += e[len(k):] + ' '
                                break
                    else:
                        new_configuration = configuration

                if config == config_ref:  # in this case, this is the reference configuration without any variant
                    configuration = 'Reference'
                else:
                    configuration =  new_configuration

                with open(Path(overall_folder) /Path(subfolder) / Path('coupling_results.pkl'), "rb") as file:
                    output = load(file)
                    # Total system costs
                    annualized_new_investment_df = output["Annualized new investments (1e9€/yr)"]
                    if 2020 in annualized_new_investment_df.columns:
                        annualized_new_investment_df = annualized_new_investment_df.drop(columns=[2020])
                    annualized_new_energy_capacity_df = output["Annualized costs new energy capacity (1e9€/yr)"]
                    if 2020 in annualized_new_energy_capacity_df.columns:
                        annualized_new_energy_capacity_df = annualized_new_energy_capacity_df.drop(columns=[2020])
                    functionment_costs_df = output["System functionment (1e9€/yr)"]
                    if 2020 in functionment_costs_df.columns:
                        functionment_costs_df = functionment_costs_df.drop(columns=[2020])

                    passed_cc = True  # check if all time steps were passed
                    if 2050 not in annualized_new_investment_df.columns:
                        passed_cc = False
                    total_system_costs_2050, total_operational_costs = process_total_costs(annualized_new_investment_df,
                                                                  annualized_new_energy_capacity_df,
                                                                  functionment_costs_df, carbon_constraint=carbon_constraint,
                                                                  eoles=eoles, year=2050)
                    total_system_costs_2050 = total_system_costs_2050.to_frame().rename(columns={0: scenario})
                    total_system_costs_2050.index.name = 'Costs'
                    second_level_index = pd.MultiIndex.from_product([total_system_costs_2050.columns, [configuration]],
                                                                    names=["Gas scenario", "Policy scenario"])
                    total_system_costs_2050.columns = second_level_index

                    if passed_cc:
                        capacities_df = output["Capacities (GW)"]
                        capacities_battery = max(capacities_df.loc['battery1', 2050], capacities_df.loc['battery4', 2050])
                        if capacities_battery > 100:  # we consider that the amount of batteries is not realistic.
                            passed_cc = False

                    if not passed_cc:
                        total_system_costs_2050 = total_system_costs_2050.applymap(lambda x: np.nan)
                    total_system_costs_2050_df = pd.concat([total_system_costs_2050_df, total_system_costs_2050], axis=1)

    total_system_costs_2050_df = total_system_costs_2050_df.T['Total costs'].reset_index().pivot(columns='Gas scenario', index='Policy scenario',
                                                                    values='Total costs')

    if save_path is None:
        save_path_plot = None
    else:
        save_path_plot = Path(save_path) / Path(f"total_system_costs.{extension}")

    # sort again dataframe to have Reference as first row
    if 'reference' in total_system_costs_2050_df.index:
        total_system_costs_2050_df = total_system_costs_2050_df.rename(index={'reference': 'Reference'})
    if 'Reference' in total_system_costs_2050_df.index:
        total_system_costs_2050_df = total_system_costs_2050_df.reindex(['Reference'] + [i for i in total_system_costs_2050_df.index if i != 'Reference'])
    if 'reference' in total_system_costs_2050_df.columns:
        total_system_costs_2050_df = total_system_costs_2050_df.reindex(columns=
            ['reference'] + [i for i in total_system_costs_2050_df.columns if i != 'reference'])
    # total_system_costs_2050_df = total_system_costs_2050_df.iloc[:,0:1]
    if percent:  # in this case, we want to do a comparison of the second column compared to the first
        for i in range(1, len(total_system_costs_2050_df.columns)):
            total_system_costs_2050_df.iloc[:,i] = (total_system_costs_2050_df.iloc[:,i] - total_system_costs_2050_df.iloc[:,0])/total_system_costs_2050_df.iloc[:,0] * 100
        total_system_costs_2050_df = total_system_costs_2050_df.drop(columns=total_system_costs_2050_df.columns[0])

    if reorder is not None:
        assert isinstance(reorder, list), 'Reorder parameter should be a list'
        total_system_costs_2050_df = total_system_costs_2050_df.reindex(columns=reorder)
    if percent:
        custom_cmap = LinearSegmentedColormap.from_list('rg',["g", "w", "r"], N=256)
    else:
        custom_cmap = None

    if dict_scenario is not None:  # we rename the scenarios
        dict_rename = {e: dict_scenario[e] if e in dict_scenario.keys() else e for e in total_system_costs_2050_df.columns}
        total_system_costs_2050_df = total_system_costs_2050_df.rename(columns=dict_rename)

    if dict_config_demandsupply is not None:  # we rename the configurations
        total_system_costs_2050_df = total_system_costs_2050_df.drop(index=[e for e in total_system_costs_2050_df.index if e not in dict_config_demandsupply.keys()])
        dict_rename = {e: dict_config_demandsupply[e] for e in total_system_costs_2050_df.index}
        total_system_costs_2050_df = total_system_costs_2050_df.rename(index=dict_rename)
    colormap(total_system_costs_2050_df, save=save_path_plot, percent=percent, custom_cmap=custom_cmap)
    return total_system_costs_2050_df


def get_main_outputs(dict_output, carbon_constraint=True, eoles=True, health=False, emissions=False):
    total_system_costs_2050_df, total_system_costs_2030_df, total_operational_costs_2050_df = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)
    stock_df, consumption_df = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)
    capacities_df, generation_df = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)
    passed = pd.Series(dtype=float)
    hourly_generation = dict()  # to save hourly generation for the two reference scenarios
    distributional_df, distributional_income_df = pd.DataFrame(), pd.DataFrame()
    emissions_df = pd.DataFrame()
    for path, name_config in zip(dict_output.values(), [n for n in dict_output.keys()]):
        with open(os.path.join(path, 'coupling_results.pkl'), "rb") as file:
            output = load(file)

            # Total system costs
            annualized_new_investment_df = output["Annualized new investments (1e9€/yr)"]
            if 2020 in annualized_new_investment_df.columns:
                annualized_new_investment_df = annualized_new_investment_df.drop(columns=[2020])
            annualized_new_energy_capacity_df = output["Annualized costs new energy capacity (1e9€/yr)"]
            if 2020 in annualized_new_energy_capacity_df.columns:
                annualized_new_energy_capacity_df = annualized_new_energy_capacity_df.drop(columns=[2020])
            functionment_costs_df = output["System functionment (1e9€/yr)"]
            if 2020 in functionment_costs_df.columns:
                functionment_costs_df = functionment_costs_df.drop(columns=[2020])

            if 2050 in annualized_new_investment_df.columns:  # scenario passed the constraint

                if name_config in ['S0', 'S0-ban']:
                    hourly_generation[name_config] = output["Hourly generation 2050 (GWh)"]

                passed = pd.concat([passed, pd.Series(1, index=[name_config])])
                total_system_costs_2050, total_operational_costs_2050 = process_total_costs(annualized_new_investment_df,
                                                                                            annualized_new_energy_capacity_df,
                                                                                            functionment_costs_df,
                                                                                            carbon_constraint=carbon_constraint,
                                                                                            eoles=eoles, year=2050, health=health)

                total_system_costs_2050 = total_system_costs_2050.to_frame().rename(columns={0: name_config})
                total_system_costs_2050 = total_system_costs_2050.drop(index='Health costs')
                total_system_costs_2050.loc['Total costs'] = total_system_costs_2050.loc['Total costs'] / 25  # convert to annualized billion € /yr
                total_system_costs_2050_df = pd.concat([total_system_costs_2050_df, total_system_costs_2050], axis=1)

                output_resirf = output["Output global ResIRF ()"]
                stock = pd.Series(output_resirf.loc[['Stock Heat pump (Million)', 'Stock Direct electric (Million)', 'Stock Natural gas (Million)', 'Stock Wood fuel (Million)']][2049]).to_frame().rename(columns={2049: name_config})
                stock_df = pd.concat([stock_df, stock], axis=1)

                consumption = pd.Series(output_resirf.loc[["Consumption Electricity (TWh)", "Consumption Natural gas (TWh)", "Consumption Wood fuel (TWh)"]][2049]).to_frame().rename(columns={2049: name_config})
                consumption_df = pd.concat([consumption_df, consumption], axis=1)

                l = list(
                    product(['Single-family', 'Multi-family'], ['Owner-occupied', 'Privately rented', 'Social-housing'],
                            ['C1', 'C2', 'C3', 'C4', 'C5']))
                temp = {}
                stock_temp = {}
                annuities_temp = {}
                for i in l:
                    t = (output_resirf.loc['Annuities {} - {} - {} (euro)'.format(*i), :] + output_resirf.loc[
                                                                                     'Energy expenditures {} - {} - {} (euro)'.format(
                                                                                         *i), :])
                    stock = output_resirf.loc['Stock {} - {} - {}'.format(*i), :]
                    t = t.loc[2025:2050]
                    stock = stock.loc[2025:2050]
                    temp.update({i: (t.sum() / stock.sum())})  # we average the cost per household over the period, and we compute the weighted mean based on the stock
                    stock_temp.update({i: stock})
                    annuities_temp.update({i: t})
                temp = pd.Series(temp)
                temp.index.names = ['Type', 'Status', 'Income']
                distributional_df = pd.concat([distributional_df, temp.rename(name_config)], axis=1)  # distributional index by type of household

                stock_temp = pd.DataFrame(stock_temp)
                annuities_temp = pd.DataFrame(annuities_temp)
                stock_temp.columns.names = ['Type', 'Status', 'Income']
                annuities_temp.columns.names = ['Type', 'Status', 'Income']
                temp_income = annuities_temp.groupby('Income', axis=1).sum().sum(axis=0) / stock_temp.groupby('Income', axis=1).sum().sum(axis=0)  # distributional index by income

                distributional_income_df = pd.concat([distributional_income_df,temp_income.rename(name_config)], axis=1)

                capacities = output["Capacities (GW)"]
                capacities.loc["offshore"] = capacities.loc["offshore_f"] + capacities.loc["offshore_g"]
                capacities.loc["pv"] = capacities.loc["pv_g"] + capacities.loc["pv_c"]
                capacities.loc["battery"] = capacities.loc["battery1"] + capacities.loc["battery4"]
                capacities.loc['hydro'] = capacities.loc['river'] + capacities.loc['lake']
                capacities.loc['peaking plants'] = capacities.loc['ocgt'] + capacities.loc['ccgt'] + capacities.loc['h2_ccgt']

                capacities = capacities.loc[['offshore', 'onshore', 'pv', 'battery', 'hydro', 'peaking plants', 'methanization', 'pyrogazification']]
                capacities = pd.Series(capacities[2050]).to_frame().rename(columns={2050: name_config})
                capacities_df = pd.concat([capacities_df, capacities], axis=1)

                generation, resirf_consumption = output["Generation (TWh)"], output['ResIRF consumption (TWh)'].T
                generation.loc['offshore'] = generation.loc['offshore_f'] + generation.loc['offshore_g']
                generation.loc['pv'] = generation.loc['pv_g'] + generation.loc['pv_c']
                generation.loc['peaking plants'] = generation.loc['ocgt'] + generation.loc['ccgt'] + generation.loc['h2_ccgt']
                generation.loc['hydro'] = generation.loc['river'] + generation.loc['lake']
                generation.loc["battery"] = generation.loc["battery1"] + generation.loc["battery4"]
                generation = generation.loc[['offshore', 'onshore', 'pv', 'battery', 'hydro', 'peaking plants', 'nuclear', 'natural_gas', 'central_wood_boiler', 'methanization', 'pyrogazification']]
                consumption = resirf_consumption.loc[['Oil fuel', 'Wood fuel']]
                generation = pd.concat([generation, consumption], axis=0)
                generation = pd.Series(generation[2050]).to_frame().rename(columns={2050: name_config})
                generation = generation.rename(index={
                    'offshore': 'Generation offshore (TWh)',
                    'onshore': 'Generation onshore (TWh)',
                    'pv': 'Generation pv (TWh)',
                    'battery': 'Generation battery (TWh)',
                    'hydro': 'Generation hydro (TWh)',
                    'peaking plants': 'Generation peaking plants (TWh)',
                    'nuclear': 'Generation nuclear (TWh)',
                    'natural_gas': 'Generation natural gas (TWh)',
                    'methanization': 'Generation methanization (TWh)',
                    'pyrogazification': 'Generation pyrogazification (TWh)',
                    'central_wood_boiler': 'Generation central wood boiler (TWh)',
                    'Oil fuel': 'Consumption Oil (TWh)',
                    'Wood fuel': 'Consumption Wood (TWh)'
                })
                generation_df = pd.concat([generation_df, generation], axis=1)

                if emissions:
                    emissionsC02 = output["Emissions (MtCO2)"]
                    emissions_df = pd.concat([emissions_df, emissionsC02.sum().rename(name_config)], axis=1)
            else:
                passed = pd.concat([passed, pd.Series(0, index=[name_config])])
                pass
                # index = ['Investment electricity costs', 'Investment heater costs', 'Investment insulation costs', 'Functionment costs', 'Health costs', 'Total costs']
                # # create a series with this index and only np.nan values
                # total_system_costs_2050 = pd.Series(index=index, data=[np.nan]*len(index)).to_frame().rename(columns={0: name_config})

    o = {
        'costs': total_system_costs_2050_df,
        'stock': stock_df,
        'consumption': consumption_df,
        'distributional': distributional_df,
        'distributional_income': distributional_income_df,
        'capacity': capacities_df,
        'generation': generation_df,
        'passed': passed
    }
    if emissions:
        o['emissions'] = emissions_df
    return o, hourly_generation


def comparison_simulations_new(dict_output: dict, ref, greenfield=False, health=False, x_min=0, x_max=None, y_min=0, y_max=None,
                           rotation=90, save_path=None, pdf=False, carbon_constraint=True, percent=False, eoles=True,
                           coordinates=None, secondary_y=None, secondary_axis_spec=None, smallest_size=100, biggest_size=400,
                           fontsize=18, remove_legend=False, s_min=None, s_max=None, waterfall=False, ref_waterfall='Package 2024 + Ban',
                               plot=True):
    if pdf:
        extension = "pdf"
    else:
        extension = "png"
    total_system_costs_2050_df, total_system_costs_2030_df, total_operational_costs_2050_df, complete_system_costs_2050_df = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float), pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)
    consumption_savings_tot_df, consumption_resirf_df = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)
    stock_heat_pump_df = pd.DataFrame(dtype=float)
    generation_df, generation_evolution_df, conversion_generation_df = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)
    peak_electricity_load_dict, emissions_dict = {}, {}
    consumption_saving_evolution_dict = {}
    subsidies_insulation_dict, subsidies_heater_dict = {}, {}
    capacities_peaking_dict, capacities_flex_df, capacities_evolution_df = {}, pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)

    if save_path is not None:
        if not os.path.isdir(save_path):  # create directory
            os.mkdir(save_path)

    for scenario in dict_output.keys():
        for path, name_config in zip(dict_output[scenario].values(), [n for n in dict_output[scenario].keys()]):
            with open(os.path.join(path, 'coupling_results.pkl'), "rb") as file:
                # if scenario == '':
                #     name_config_tot = name_config
                # else:
                name_config_tot = scenario + ' ' + name_config
                output = load(file)

                # Total system costs
                annualized_new_investment_df = output["Annualized new investments (1e9€/yr)"]
                if 2020 in annualized_new_investment_df.columns:
                    annualized_new_investment_df = annualized_new_investment_df.drop(columns=[2020])
                annualized_new_energy_capacity_df = output["Annualized costs new energy capacity (1e9€/yr)"]
                if 2020 in annualized_new_energy_capacity_df.columns:
                    annualized_new_energy_capacity_df = annualized_new_energy_capacity_df.drop(columns=[2020])
                functionment_costs_df = output["System functionment (1e9€/yr)"]
                if 2020 in functionment_costs_df.columns:
                    functionment_costs_df = functionment_costs_df.drop(columns=[2020])
                total_system_costs_2050, total_operational_costs_2050 = process_total_costs(annualized_new_investment_df, annualized_new_energy_capacity_df,
                                                         functionment_costs_df, carbon_constraint=carbon_constraint,
                                                         eoles=eoles, year=2050)
                total_system_costs_2030, total_operational_costs_2030 = process_total_costs(annualized_new_investment_df, annualized_new_energy_capacity_df,
                                                         functionment_costs_df, carbon_constraint=carbon_constraint,
                                                         eoles=eoles, year=2030)
                total_system_costs_2050 = total_system_costs_2050.to_frame().rename(columns={0: scenario})
                total_system_costs_2050.index.name = 'Costs'
                second_level_index = pd.MultiIndex.from_product([total_system_costs_2050.columns, [name_config]], names=["Gas scenario", "Policy scenario"])
                total_system_costs_2050.columns = second_level_index
                total_system_costs_2050_df = pd.concat([total_system_costs_2050_df, total_system_costs_2050], axis=1)

                total_operational_costs_2050 = total_operational_costs_2050.to_frame()
                total_operational_costs_2050 = aggregate_capacities(total_operational_costs_2050)
                total_operational_costs_2050 = total_operational_costs_2050.rename(columns={0: scenario})
                total_operational_costs_2050.index.name = 'Costs'
                second_level_index = pd.MultiIndex.from_product([total_operational_costs_2050.columns, [name_config]], names=["Gas scenario", "Policy scenario"])
                total_operational_costs_2050.columns = second_level_index
                total_operational_costs_2050_df = pd.concat([total_operational_costs_2050_df, total_operational_costs_2050], axis=1)

                total_system_costs_2030 = total_system_costs_2030.to_frame().rename(columns={0: scenario})
                total_system_costs_2030.index.name = 'Costs'
                second_level_index = pd.MultiIndex.from_product([total_system_costs_2030.columns, [name_config]], names=["Gas scenario", "Policy scenario"])
                total_system_costs_2030.columns = second_level_index
                total_system_costs_2030_df = pd.concat([total_system_costs_2030_df, total_system_costs_2030], axis=1)

                # Complete system costs
                annualized_new_investment_df = output["Annualized new investments (1e9€/yr)"]
                if 2020 in annualized_new_investment_df.columns:
                    annualized_new_investment_df = annualized_new_investment_df.drop(columns=[2020])
                annualized_new_energy_capacity_df = output["Annualized costs new energy capacity (1e9€/yr)"]
                if 2020 in annualized_new_energy_capacity_df.columns:
                    annualized_new_energy_capacity_df = annualized_new_energy_capacity_df.drop(columns=[2020])
                functionment_costs_df = output["System functionment (1e9€/yr)"]
                if 2020 in functionment_costs_df.columns:
                    functionment_costs_df = functionment_costs_df.drop(columns=[2020])
                complete_system_costs_2050 = process_complete_system_cost_2050(annualized_new_investment_df,
                                                                               annualized_new_energy_capacity_df,
                                                                               functionment_costs_df,
                                                                               carbon_constraint=carbon_constraint,
                                                                               eoles=eoles)

                complete_system_costs_2050 = complete_system_costs_2050.to_frame().rename(columns={0: scenario})
                complete_system_costs_2050.index.name = 'Costs'
                second_level_index = pd.MultiIndex.from_product([complete_system_costs_2050.columns, [name_config]], names=["Gas scenario", "Policy scenario"])
                complete_system_costs_2050.columns = second_level_index
                complete_system_costs_2050_df = pd.concat([complete_system_costs_2050_df, complete_system_costs_2050], axis=1)

                consumption_savings = output["ResIRF consumption savings (TWh)"]
                consumption_savings = consumption_savings.rename(
                    columns={"Consumption saving heater (TWh)": "Consumption saving heater (TWh/year)",
                             "Consumption saving insulation (TWh)": "Consumption saving insulation (TWh/year)"})
                consumption = output["Output global ResIRF ()"].loc[["Consumption Electricity (TWh)", "Consumption Natural gas (TWh)",
                     "Consumption Oil fuel (TWh)", "Consumption Wood fuel (TWh)", "Consumption Heating (TWh)"]]

                consumption_ini = consumption.sum(axis=0).iloc[0]
                consumption_savings_tot = consumption_savings.sum(axis=0).to_frame().rename(columns={0: name_config_tot})
                if percent:
                    consumption_savings_tot = consumption_savings_tot / consumption_ini * 100
                consumption_savings_tot_df = pd.concat([consumption_savings_tot_df, consumption_savings_tot], axis=1)

                consumption_savings_evolution = consumption_savings.reset_index().rename(columns={'index': 'year'})
                consumption_savings_evolution["period"] = consumption_savings_evolution.apply(lambda row: (row["year"] - 2025) // 5, axis=1)
                consumption_savings_evolution = consumption_savings_evolution.groupby("period").agg(
                    {"year": np.min, "Consumption saving heater (TWh/year)": np.sum, "Consumption saving insulation (TWh/year)": np.sum}).set_index("year")
                consumption_savings_evolution.index.name = None
                consumption_saving_evolution_dict[name_config_tot] = consumption_savings_evolution

                if eoles:
                    consumption_resirf = output['ResIRF consumption (TWh)'].T[2050]
                    if isinstance(consumption_resirf, pd.Series):  # in the case of greenfield
                        consumption_resirf = consumption_resirf.to_frame()
                    consumption_resirf = consumption_resirf.rename(columns={2050: name_config_tot})
                    consumption_resirf_df = pd.concat([consumption_resirf_df, consumption_resirf], axis=1)

                output_resirf = output["Output global ResIRF ()"]
                stock_heat_pump = pd.Series(output_resirf.loc["Stock Heat pump (Million)"][2049],
                                            index=["Stock Heat pump (Million)"]).to_frame().rename(columns={0: name_config_tot})
                stock_heat_pump_df = pd.concat([stock_heat_pump_df, stock_heat_pump], axis=1)

                try:
                    peak_electricity_load_info_df = output["Peak electricity load"]
                    peak_electricity_load_info_df = peak_electricity_load_info_df[
                        ["peak_electricity_load", "year"]].groupby(["year"]).mean().squeeze()
                    peak_electricity_load_dict[name_config_tot] = peak_electricity_load_info_df

                    emissions = output["Emissions (MtCO2)"]
                    if greenfield:
                        emissions = pd.Series(emissions.squeeze(), index=emissions.index)
                    else:
                        emissions = emissions.squeeze()
                    emissions_dict[name_config_tot] = emissions

                    subsidies = output["Subsidies (%)"] * 100
                    dataframe_subsidy_list = [subsidies]
                    for i in range(4):
                        tmp = subsidies.copy()
                        tmp.index += i + 1
                        dataframe_subsidy_list.append(tmp)
                    dataframe_subsidy = pd.concat(dataframe_subsidy_list, axis=0).sort_index(ascending=True)

                    subsidies_insulation_dict[name_config_tot] = dataframe_subsidy[["Insulation"]].squeeze()
                    subsidies_heater_dict[name_config_tot] = dataframe_subsidy[["Heater"]].squeeze()
                    if secondary_y is not None and name_config_tot == secondary_y:
                        with open(os.path.join(path, 'config', 'config_coupling.json')) as file:
                            config_coupling = json.load(file)
                        price_cap = config_coupling['subsidy']['insulation']['cap']
                        subsidies_insulation_dict[name_config_tot] = subsidies_insulation_dict[name_config_tot] * price_cap / 100
                except:
                    pass

                if eoles:
                    capacities_df = output["Capacities (GW)"]
                    capacities_df = capacities_df[[2030, 2050]]
                    # selected_capacities = ["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "battery1", "battery4"]
                    # capacities_df = capacities_df[selected_capacities]
                    capacities_df.loc["offshore"] = capacities_df.loc["offshore_f"] + capacities_df.loc["offshore_g"]
                    capacities_df.loc["pv"] = capacities_df.loc["pv_g"] + capacities_df.loc["pv_c"]
                    capacities_df.loc["battery"] = capacities_df.loc["battery1"] + capacities_df.loc["battery4"]
                    capacities_df.loc['hydro'] = capacities_df.loc['river'] + capacities_df.loc['lake']
                    capacities_df.loc['peaking plants'] = capacities_df.loc['ocgt'] + capacities_df.loc['ccgt'] + capacities_df.loc['h2_ccgt']

                    subset_cap = ['onshore', 'offshore', 'pv', 'nuclear', 'hydro', 'phs', 'battery', 'ocgt', 'ccgt', 'h2_ccgt', 'peaking plants']
                    # capacities_evolution = capacities_df.loc[subset_cap]
                    capacities_evolution = capacities_df.copy()
                    capacities_evolution.index.name = 'Technology'
                    second_level_index = pd.MultiIndex.from_product([capacities_evolution.index, [name_config]],
                                                                    names=["Technology", "Policy scenario"])
                    capacities_evolution.index = second_level_index
                    capacities_evolution_df = pd.concat([capacities_evolution_df, capacities_evolution], axis=0)

                    capacities_df = output["Capacities (GW)"].T
                    tec_flexibility = ['nuclear', "phs", 'ocgt', 'ccgt', 'h2_ccgt', 'battery1', 'battery4']
                    capacities_flex = capacities_df[tec_flexibility].copy()
                    capacities_flex['Configuration'] = name_config_tot
                    capacities_flex_df = pd.concat([capacities_flex_df,
                                                    capacities_flex])  # TODO: prendre l'année 2050, ou bien montrer l'évolution des capacités au cours du temps

                    capacities_df = output["Capacities (GW)"].T
                    capacities_df = capacities_df[['ocgt', 'ccgt', 'h2_ccgt']]
                    capacities_df['methane_plants'] = capacities_df['ocgt'] + capacities_df['ccgt']
                    capacities_peaking_dict[name_config_tot] = capacities_df[['methane_plants', 'h2_ccgt']]

                    generation = output["Generation (TWh)"]
                    generation_subset = generation[[2030, 2050]]
                    generation_subset.loc['offshore'] = generation_subset.loc['offshore_f'] + generation_subset.loc['offshore_g']
                    generation_subset.loc['pv'] = generation_subset.loc['pv_g'] + generation_subset.loc['pv_c']
                    generation_subset.loc['peaking plants'] = generation_subset.loc['ocgt'] + generation_subset.loc['ccgt'] + generation_subset.loc[
                        'h2_ccgt']
                    generation_subset.loc['hydro'] = generation_subset.loc['river'] + generation_subset.loc['lake']
                    generation_subset.loc['biogas'] = generation_subset.loc['methanization'] + generation_subset.loc['pyrogazification']
                    generation_2050 = generation_subset[[2050]].rename(columns={2050: name_config_tot}).T
                    generation_2050 = pd.concat(
                        [generation_2050.T, consumption_resirf.rename(index={'Electricity': 'Electricity for heating',
                                                                             'Natural gas': 'Gas for heating'})], axis=0)
                    generation_df = pd.concat([generation_df, generation_2050], axis=1)

                    elec_gene = ['onshore', 'offshore', 'pv', 'nuclear', 'hydro', 'peaking plants']
                    gas_gene = ['biogas', 'methanation', 'electrolysis']
                    subset_tot = elec_gene + gas_gene
                    generation_evolution = generation_subset.loc[subset_tot]
                    generation_evolution.index.name = 'Technology'
                    second_level_index = pd.MultiIndex.from_product([generation_evolution.index, [name_config]],
                                                                    names=["Technology", "Policy scenario"])
                    generation_evolution.index = second_level_index
                    generation_evolution_df = pd.concat([generation_evolution_df, generation_evolution], axis=0)

                    conversion_generation = output['Conversion generation (TWh)']
                    conversion_generation = conversion_generation[[2050]]
                    conversion_generation = conversion_generation.rename(columns={2050: name_config_tot}).T
                    # conversion_generation['peaking plants'] = conversion_generation['ocgt'] + conversion_generation['ccgt'] + conversion_generation['h2_ccgt']
                    conversion_generation['peaking plants'] = conversion_generation['ocgt'] + conversion_generation[
                        'ccgt']  # TODO: à modifier pour les prochains runs
                    conversion_generation_df = pd.concat([conversion_generation_df, conversion_generation.T], axis=1)

    if plot:
        if eoles:
            generation_df, conversion_generation_df = generation_df.T, conversion_generation_df.T

            if save_path is None:
                save_path_plot = None
            else:
                save_path_plot = os.path.join(save_path, f"gas_demand_balance.{extension}")
            supply_gas = ['biogas', 'methanation', 'electrolysis']
            demand_gas = ['peaking plants', 'Gas for heating']

            gas_generation_demand = pd.concat([generation_df[['biogas', 'methanation', 'electrolysis', 'Gas for heating']],
                 conversion_generation_df[['peaking plants']]], axis=1)
            gas_generation_demand[demand_gas] = - gas_generation_demand[demand_gas]
            make_stacked_bar_plot(gas_generation_demand, subset=supply_gas + demand_gas,
                                  y_label="Gas demand balance (TWh)",
                                  colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                                  index_int=False, rotation=rotation, dict_legend=DICT_TRANSFORM_LEGEND,
                                  save=save_path_plot, hline=True)

            if save_path is None:
                save_path_plot = None
            else:
                save_path_plot = os.path.join(save_path, f"elec_demand_balance.{extension}")
            supply_elec = ['onshore', 'offshore', 'pv', 'hydro', 'peaking plants']
            demand_elec = ['electrolysis', 'methanation', 'Electricity for heating']
            elec_generation_demand = pd.concat([generation_df[supply_elec + ['Electricity for heating']],
                                                conversion_generation_df[['electrolysis', 'methanation']]], axis=1)
            elec_generation_demand[demand_elec] = - elec_generation_demand[demand_elec]
            # TODO: ajouter la demande en elec
            make_stacked_bar_plot(elec_generation_demand, subset=supply_elec + demand_elec,
                                  y_label="Electricity demand balance (TWh)",
                                  colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                                  index_int=False, rotation=rotation, dict_legend=DICT_TRANSFORM_LEGEND,
                                  save=save_path_plot, hline=True)


        ranking_exogenous_scenario = list(dict_output.keys())
        ranking_policy_scenario = list(dict_output[scenario].keys())
        # Total system costs
        if carbon_constraint:
            subset_annualized_costs = ["Investment electricity costs", "Investment heater costs",
                                       "Investment insulation costs", "Functionment costs", "Health costs"]
        else:
            subset_annualized_costs = ["Investment electricity costs", "Investment heater costs",
                                       "Investment insulation costs", "Functionment costs", "Carbon cost",
                                       "Health costs"]
        if not eoles:  # only includes ResIRF for CBA
            subset_annualized_costs.remove("Investment electricity costs")
        if save_path is None:
            save_path_plot = None
        else:
            save_path_plot = os.path.join(save_path, f"total_system_costs.{extension}")

        total_system_costs_2050_df = total_system_costs_2050_df.stack(level="Policy scenario")
        # total_system_costs_df = total_system_costs_df.reorder_levels(['Policy scenario', 'Costs']).loc[ranking_policy_scenario].reorder_levels(['Costs', 'Policy scenario'])

        make_clusterstackedbar_plot(total_system_costs_2050_df, groupby='Costs', subset=subset_annualized_costs,
                                    y_label="Total system costs (Md€)",
                                    colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                                    dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot, rotation=90,
                                    ranking_exogenous_scenario=ranking_exogenous_scenario, ranking_policy_scenario=ranking_policy_scenario
                                    )

        if save_path is None:
            save_path_plot = None
        else:
            save_path_plot = os.path.join(save_path, f"total_system_costs_2030.{extension}")

        total_system_costs_2030_df = total_system_costs_2030_df.stack(level="Policy scenario")
        # total_system_costs_df = total_system_costs_df.reorder_levels(['Policy scenario', 'Costs']).loc[ranking_policy_scenario].reorder_levels(['Costs', 'Policy scenario'])

        make_clusterstackedbar_plot(total_system_costs_2030_df, groupby='Costs', subset=subset_annualized_costs,
                                    y_label="Total system costs (Md€)",
                                    colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                                    dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot, rotation=90,
                                    ranking_exogenous_scenario=ranking_exogenous_scenario, ranking_policy_scenario=ranking_policy_scenario
                                    )

        # make_stacked_bar_plot(total_system_costs_df.T, subset=subset_annualized_costs,
        #                       y_label="Total system costs (Md€)",
        #                       colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
        #                       index_int=False,
        #                       rotation=90, dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot)

        # Total system costs
        hc_excluded = total_system_costs_2050_df.loc['Total costs'].sub(total_system_costs_2050_df.loc['Health costs'])
        hc_excluded.index = pd.MultiIndex.from_product([['Total costs HC excluded'], hc_excluded.index], names=['Costs', 'Policy scenario'])
        total_system_costs_2050_df = pd.concat([total_system_costs_2050_df, hc_excluded], axis=0)
        # total_system_costs_df = total_system_costs_df.append(hc_excluded)

        reference_rows = total_system_costs_2050_df.loc[total_system_costs_2050_df.index.get_level_values('Policy scenario') == ref]
        total_system_costs_diff_df = total_system_costs_2050_df.subtract(reference_rows.reset_index(level=1, drop=True), level=0)

        if waterfall:
            if save_path is None:
                save_path_plot = None
            else:
                save_path_plot = os.path.join(save_path, f"waterfall_total_system_costs.{extension}")

            tmp = total_system_costs_diff_df.loc[total_system_costs_diff_df.index.get_level_values('Policy scenario') == ref_waterfall].droplevel('Policy scenario')
            tmp = tmp.drop(index=['Total costs HC excluded'])
            tmp = tmp.reindex(['Investment heater costs', 'Investment insulation costs', 'Investment electricity costs', 'Functionment costs', 'Health costs', 'Total costs'])
            waterfall_chart(tmp, colors=resources_data["colors_eoles"], rotation=0, save=save_path_plot, format_y=lambda y, _: '{:.0f} B€'.format(y),
                            title="Difference in total system costs", y_label=None, hline=True, dict_legend=DICT_LEGEND_WATERFALL)
        if health:
            scatter = 'Total costs'
            if carbon_constraint:
                subset_costs = ["Investment electricity costs", "Investment heater costs",
                                "Investment insulation costs", "Functionment costs", "Health costs"]
            else:
                subset_costs = ["Investment electricity costs", "Investment heater costs",
                                "Investment insulation costs", "Functionment costs", "Carbon cost", "Health costs"]
        else:
            scatter = 'Total costs HC excluded'
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

        total_system_costs_diff_df = total_system_costs_diff_df.reindex(['Health costs', 'Functionment costs',
                                                                         'Investment heater costs', 'Investment insulation costs',
                                                                         'Investment electricity costs',
                                                                         'Total costs', 'Total costs HC excluded'], level=0)  # we reindex for sorting the legend
        make_clusterstackedbar_plot(total_system_costs_diff_df, groupby='Costs', subset=subset_costs,
                                    y_label="Total system costs",
                                    colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.0f} B€'.format(y),
                                    dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot, scatter=scatter, ref=ref,
                                    drop=True, hline=True, ranking_exogenous_scenario=ranking_exogenous_scenario,
                                    ranking_policy_scenario=ranking_policy_scenario, legend_loc='right', reorder_labels=True)

        total_operational_costs_2050_df = total_operational_costs_2050_df.stack(level="Policy scenario")
        reference_rows = total_operational_costs_2050_df.loc[total_operational_costs_2050_df.index.get_level_values('Policy scenario') == ref]
        total_operational_costs_diff_df = total_operational_costs_2050_df.subtract(reference_rows.reset_index(level=1, drop=True), level=0)
        if save_path is None:
            save_path_plot = None
        else:
            save_path_plot = os.path.join(save_path, f"difference_total_operational_costs.{extension}")

        make_clusterstackedbar_plot(total_operational_costs_diff_df, groupby='Costs',
                                    y_label="", format_y=lambda y, _: '{:.0f} B€'.format(y),
                                    dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot, ref=ref,
                                    drop=True, hline=True, ranking_exogenous_scenario=ranking_exogenous_scenario,
                                    ranking_policy_scenario=ranking_policy_scenario, legend_loc='right')

        hc_excluded_2030 = total_system_costs_2030_df.loc['Total costs'].sub(total_system_costs_2030_df.loc['Health costs'])
        hc_excluded_2030.index = pd.MultiIndex.from_product([['Total costs HC excluded'], hc_excluded_2030.index], names=['Costs', 'Policy scenario'])
        total_system_costs_2030_df = pd.concat([total_system_costs_2030_df, hc_excluded_2030], axis=0)

        reference_rows = total_system_costs_2030_df.loc[total_system_costs_2030_df.index.get_level_values('Policy scenario') == ref]
        total_system_costs_diff_2030_df = total_system_costs_2030_df.subtract(reference_rows.reset_index(level=1, drop=True), level=0)

        if save_path is None:
            save_path_plot = None
        else:
            save_path_plot = os.path.join(save_path, f"difference_total_system_costs_2030.{extension}")

        make_clusterstackedbar_plot(total_system_costs_diff_2030_df, groupby='Costs', subset=subset_costs,
                                    y_label="Total system costs (Md€)",
                                    colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                                    dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot, scatter=scatter, ref=ref,
                                    drop=True, hline=True, ranking_exogenous_scenario=ranking_exogenous_scenario,
                                    ranking_policy_scenario=ranking_policy_scenario)

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


        complete_system_costs_2050_df = complete_system_costs_2050_df.stack(level="Policy scenario")
        # total_system_costs_df = total_system_costs_df.reorder_levels(['Policy scenario', 'Costs']).loc[ranking_policy_scenario].reorder_levels(['Costs', 'Policy scenario'])

        make_clusterstackedbar_plot(complete_system_costs_2050_df, groupby='Costs', subset=subset_complete_costs,
                                    y_label="Complete system costs in 2050 (Md€/year)",
                                    colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                                    dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot, rotation=90,
                                    ranking_exogenous_scenario=ranking_exogenous_scenario, ranking_policy_scenario=ranking_policy_scenario
                                    )

        hc_excluded = complete_system_costs_2050_df.loc['Total costs'].sub(complete_system_costs_2050_df.loc['Health costs'])
        hc_excluded.index = pd.MultiIndex.from_product([['Total costs HC excluded'], hc_excluded.index], names=['Costs', 'Policy scenario'])
        complete_system_costs_2050_df = pd.concat([complete_system_costs_2050_df, hc_excluded], axis=0)
        # complete_system_costs_2050_df = complete_system_costs_2050_df.append(hc_excluded)

        reference_rows = complete_system_costs_2050_df.loc[complete_system_costs_2050_df.index.get_level_values('Policy scenario') == ref]
        complete_system_costs_2050_diff_df = complete_system_costs_2050_df.subtract(reference_rows.reset_index(level=1, drop=True), level=0)

        if health:
            if carbon_constraint:
                subset_complete_costs = ["Investment electricity costs", "Investment heater costs",
                                         "Investment insulation costs", "Functionment costs", "Health costs"]
            else:
                subset_complete_costs = ["Investment electricity costs", "Investment heater costs",
                                         "Investment insulation costs", "Functionment costs", "Carbon cost",
                                         "Health costs"]
        else:
            if carbon_constraint:
                subset_complete_costs = ["Investment electricity costs", "Investment heater costs",
                                         "Investment insulation costs", "Functionment costs"]
            else:
                subset_complete_costs = ["Investment electricity costs", "Investment heater costs",
                                         "Investment insulation costs", "Functionment costs", "Carbon cost"]

        if not eoles:
            subset_complete_costs.remove('Investment electricity costs')

        # for col in complete_system_costs_2050_diff_df.columns:
        #     if col != ref:
        #         complete_system_costs_2050_diff_df[col] = complete_system_costs_2050_diff_df[col] - \
        #                                                   complete_system_costs_2050_diff_df[ref]

        if save_path is None:
            save_path_plot = None
        else:
            save_path_plot = os.path.join(save_path, f"difference_complete_system_costs_2050.{extension}")

        make_clusterstackedbar_plot(complete_system_costs_2050_diff_df, groupby='Costs', subset=subset_complete_costs,
                                    y_label="Difference of complete system costs in 2050 (Billion € / year)",
                                    colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                                    dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot, scatter='Total costs', ref=ref,
                                    drop=True, hline=True, ranking_exogenous_scenario=ranking_exogenous_scenario,
                                    ranking_policy_scenario=ranking_policy_scenario)

        # Energy generation
        try:  # those graphs cannot be made if we have multiple exogenous scenarios in addition
            if save_path is None:
                save_path_plot = None
            else:
                save_path_plot = os.path.join(save_path, f"electricity_generation.{extension}")

            make_clusterstackedbar_plot(generation_evolution_df.loc[:,2050].to_frame(), groupby='Technology', subset=elec_gene,
                                        y_label="Electricity generation (TWh)",
                                        colors=resources_data["new_colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                                        dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot,
                                        ranking_policy_scenario=ranking_policy_scenario
                                        )

            if save_path is None:
                save_path_plot = None
            else:
                save_path_plot = os.path.join(save_path, f"gas_generation.{extension}")

            make_clusterstackedbar_plot(generation_evolution_df.loc[:,2050].to_frame(), groupby='Technology', subset=gas_gene,
                                        y_label="Gas generation (TWh)",
                                        colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                                        dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot,
                                        ranking_policy_scenario=ranking_policy_scenario
                                        )

            reference_rows = generation_evolution_df.loc[generation_evolution_df.index.get_level_values('Policy scenario') == ref]
            generation_elec_diff_df = generation_evolution_df.subtract(reference_rows.reset_index(level=1, drop=True), level=0)

            if save_path is None:
                save_path_plot = None
            else:
                save_path_plot = os.path.join(save_path, f"electricity_generation_difference.{extension}")

            make_clusterstackedbar_plot(generation_elec_diff_df.loc[:,2050].to_frame(), groupby='Technology', subset=elec_gene,
                                        y_label="Electricity generation (TWh)",
                                        colors=resources_data["new_colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                                        dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot, ref=ref,
                                        drop=True, hline=True, ranking_policy_scenario=ranking_policy_scenario)

            if save_path is None:
                save_path_plot = None
            else:
                save_path_plot = os.path.join(save_path, f"gas_generation_difference.{extension}")

            make_clusterstackedbar_plot(generation_elec_diff_df.loc[:,2050].to_frame(), groupby='Technology', subset=gas_gene,
                                        y_label="Gas generation (TWh)",
                                        colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                                        dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot, ref=ref,
                                        drop=True, hline=True, ranking_policy_scenario=ranking_policy_scenario)

            # Capacity evolution
            if save_path is None:
                save_path_plot = None
            else:
                save_path_plot = os.path.join(save_path, f"electricity_capacity.{extension}")

            subset_elec = ['onshore', 'offshore', 'pv', 'nuclear', 'hydro', 'peaking plants', 'battery']
            # we only consider year 2050 for the plot
            make_clusterstackedbar_plot(capacities_evolution_df.loc[:,2050].to_frame(), groupby='Technology', subset=subset_elec,
                                        y_label="Electricity capacity",
                                        colors=resources_data["new_colors_eoles"], format_y=lambda y, _: '{:.0f} GW'.format(y),
                                        dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot, display_title=False,
                                        ranking_policy_scenario=ranking_policy_scenario, legend_loc='right', reorder_labels=True
                                        )

            reference_rows = capacities_evolution_df.loc[capacities_evolution_df.index.get_level_values('Policy scenario') == ref]
            capacities_evolution_diff_df = capacities_evolution_df.subtract(reference_rows.reset_index(level=1, drop=True), level=0)

            if save_path is None:
                save_path_plot = None
            else:
                save_path_plot = os.path.join(save_path, f"electricity_capacity_difference.{extension}")

            make_clusterstackedbar_plot(capacities_evolution_diff_df.loc[:,2050].to_frame(), groupby='Technology', subset=subset_elec,
                                        y_label="Electricity capacity (GW)",
                                        colors=resources_data["new_colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                                        dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot, ref=ref,
                                        drop=True, hline=True, ranking_policy_scenario=ranking_policy_scenario)

            # All capacity evolution
            if save_path is None:
                save_path_plot = None
            else:
                save_path_plot = os.path.join(save_path, f"all_capacity.{extension}")

            subset_all = ['onshore', 'offshore', 'pv', 'nuclear', 'hydro', 'peaking plants', 'battery', 'methanization', 'pyrogazification', 'electrolysis', 'methanation']
            # we only consider year 2050 for the plot
            make_clusterstackedbar_plot(capacities_evolution_df.loc[:,2050].to_frame(), groupby='Technology', subset=subset_all,
                                        y_label="Energy capacity",
                                        colors=resources_data["new_colors_eoles"], format_y=lambda y, _: '{:.0f} GW'.format(y),
                                        dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot, display_title=False,
                                        ranking_policy_scenario=ranking_policy_scenario, legend_loc='right', reorder_labels=True
                                        )

            if save_path is None:
                save_path_plot = None
            else:
                save_path_plot = os.path.join(save_path, f"all_capacity_difference.{extension}")

            make_clusterstackedbar_plot(capacities_evolution_diff_df.loc[:,2050].to_frame(), groupby='Technology', subset=subset_all,
                                        y_label="Energy capacity (GW)",
                                        colors=resources_data["new_colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                                        dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot, ref=ref,
                                        drop=True, hline=True, ranking_policy_scenario=ranking_policy_scenario, legend_loc='right')

            if waterfall:
                if save_path is None:
                    save_path_plot = None
                else:
                    save_path_plot = os.path.join(save_path, f"waterfall_all_capacity.{extension}")

                tmp = capacities_evolution_diff_df.loc[subset_all,2050].to_frame()
                tmp = tmp.loc[tmp.index.get_level_values('Policy scenario') == ref_waterfall].droplevel('Policy scenario')
                tmp = tmp.loc[~(abs(tmp)<=0.1).all(axis=1)]
                # tmp = tmp.reindex(['Investment heater costs', 'Investment insulation costs', 'Investment electricity costs', 'Functionment costs', 'Health costs'])
                waterfall_chart(tmp, colors=resources_data["new_colors_eoles"], rotation=0, save=save_path_plot, format_y=lambda y, _: '{:.0f} GW'.format(y),
                                title="", y_label=None, hline=True, total=False, unit='GW', float_precision=1)

            # Flexible capacity evolution
            if save_path is None:
                save_path_plot = None
            else:
                save_path_plot = os.path.join(save_path, f"flexible_capacity.{extension}")

            subset_flex = ['nuclear', "phs", 'battery', 'peaking plants']
            make_clusterstackedbar_plot(capacities_evolution_df.loc[:,2050].to_frame(), groupby='Technology', subset=subset_flex,
                                        y_label="Flexible capacity (GW)",
                                        colors=resources_data["new_colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                                        dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot,
                                        ranking_policy_scenario=ranking_policy_scenario
                                        )

            if save_path is None:
                save_path_plot = None
            else:
                save_path_plot = os.path.join(save_path, f"flexible_capacity_difference.{extension}")

            make_clusterstackedbar_plot(capacities_evolution_diff_df.loc[:,2050].to_frame(), groupby='Technology', subset=subset_flex,
                                        y_label="Flexible capacity (GW)",
                                        colors=resources_data["new_colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y),
                                        dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot, ref=ref,
                                        drop=True, hline=True, ranking_policy_scenario=ranking_policy_scenario)

        except:
            pass
        # Total consumption savings
        if save_path is None:
            save_path_plot = None
        else:
            save_path_plot = os.path.join(save_path, f"consumption_savings.{extension}")
        if percent:
            unit = "%"
        else:
            unit = "TWh"
        make_stacked_bar_plot(consumption_savings_tot_df.T, y_label=f"Total consumption savings {unit}",
                              colors=resources_data["colors_resirf"], format_y=lambda y, _: '{:.0f}'.format(y),
                              index_int=False,
                              rotation=90, dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot)

        total_system_costs_2050_new = total_system_costs_2050_df.unstack().copy()
        total_system_costs_2050_new.columns = total_system_costs_2050_new.columns.map(' '.join)
        savings_and_costs_df = pd.concat([consumption_savings_tot_df, total_system_costs_2050_new], axis=0)
        savings_and_costs_df = savings_and_costs_df.T
        plot_comparison_savings(savings_and_costs_df, x="Consumption saving insulation (TWh/year)",
                                y="Consumption saving heater (TWh/year)",
                                save=os.path.join(save_path, f"savings_and_costs.{extension}"),
                                col_for_size="Total costs", smallest_size=smallest_size, biggest_size=biggest_size,
                                fontsize=fontsize,
                                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, unit=unit, coordinates=coordinates)

        savings_and_costs_hp = pd.concat([consumption_savings_tot_df, stock_heat_pump_df, total_system_costs_2050_new],
                                         axis=0)
        savings_and_costs_hp = savings_and_costs_hp.T
        plot_comparison_savings(savings_and_costs_hp, x="Consumption saving insulation (TWh/year)",
                                y="Stock Heat pump (Million)",
                                save=os.path.join(save_path, f"savings_and_costs_hp.{extension}"),
                                col_for_size="Total costs", format_y=lambda y, _: '{:.0f} M'.format(y), format_x=lambda x, _: '{:.0f} {}'.format(x, unit),
                                smallest_size=smallest_size, biggest_size=biggest_size,
                                fontsize=fontsize,
                                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, unit=unit, coordinates=coordinates,
                                remove_legend=remove_legend, s_min=s_min, s_max=s_max)

        # consumption_savings_tot_df_new = consumption_savings_tot_df.copy()
        # consumption_savings_tot_df_new.columns = consumption_savings_tot_df_new.columns.str.strip()
        # consumption_savings_tot_df_new = consumption_savings_tot_df_new.stack().to_frame()
        # consumption_savings_tot_df_new.columns = [total_system_costs_2050_df.columns.name]
        # consumption_savings_tot_df_new.index.names = total_system_costs_2050_df.index.names
        #
        # stock_heat_pump_df_new = stock_heat_pump_df.copy()
        # stock_heat_pump_df_new.columns = stock_heat_pump_df_new.columns.str.strip()
        # stock_heat_pump_df_new = stock_heat_pump_df_new.stack().to_frame()
        # stock_heat_pump_df_new.columns=[total_system_costs_2050_df.columns.name]
        # stock_heat_pump_df_new.index.names = total_system_costs_2050_df.index.names
        #
        # tmp = pd.concat([consumption_savings_tot_df_new, stock_heat_pump_df_new])
        #
        # make_cluster_scatterplot(tmp, x="Consumption saving insulation (TWh/year)", y="Stock Heat pump (Million)", y_label='')

        if eoles:
            emissions_tot = \
            pd.concat([emissions_dict[key].rename(key).to_frame() for key in emissions_dict.keys()], axis=1).loc[
                2050].rename("Emissions (MtCO2)").to_frame().T
            savings_and_emissions_df = pd.concat([consumption_savings_tot_df, emissions_tot], axis=0)
            savings_and_emissions_df = savings_and_emissions_df.T
            plot_comparison_savings(savings_and_emissions_df, x="Consumption saving insulation (TWh/year)",
                                    y="Consumption saving heater (TWh/year)",
                                    save=os.path.join(save_path, f"savings_and_emissions.{extension}"),
                                    col_for_size="Emissions (MtCO2)", smallest_size=100,
                                    biggest_size=400, fontsize=18, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                                    unit=unit, coordinates=coordinates)

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
            make_line_plots(subsidies_insulation_dict, y_label="Subsidies (%)",
                            format_y=lambda y, _: '{:.0f}'.format(y),
                            index_int=True, save=save_path_plot, rotation=45, x_ticks=dataframe_subsidy.index[::2],
                            y_min=0,
                            y_max=100, secondary_y=secondary_y, secondary_axis_spec=secondary_axis_spec)

            # Evolution of heater subsidies
            if save_path is None:
                save_path_plot = None
            else:
                save_path_plot = os.path.join(save_path, f"subsidies_heater.{extension}")
            make_line_plots(subsidies_heater_dict, y_label="Subsidies (%)",
                            format_y=lambda y, _: '{:.2f}'.format(y),
                            index_int=True, save=save_path_plot, rotation=45, x_ticks=dataframe_subsidy.index[::2],
                            y_min=0,
                            y_max=100)

        except:
            pass

        if eoles:

            if save_path is None:
                save_path_plot = None
            else:
                save_path_plot = os.path.join(save_path, f"flex_capacities_2050.{extension}")
            make_stacked_bar_plot(capacities_flex_df.loc[2050].set_index("Configuration"),
                                  y_label="Flexible capacity (GW)",
                                  format_y=lambda y, _: '{:.0f}'.format(y), index_int=False, rotation=90,
                                  save=save_path_plot)  # TODO: ajouter les bonnes couleurs ici
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
    else:
        pass

    total_system_costs_2050_df.to_csv(os.path.join(save_path, "total_system_costs_2050.csv"))
    capacities_evolution_df.loc[:,2050].unstack().to_csv(os.path.join(save_path, "capacities_evolution.csv"))
    generation_evolution_df.loc[:,2050].unstack().to_csv(os.path.join(save_path, "generation_evolution.csv"))
    savings_and_costs_hp.to_csv(os.path.join(save_path, "savings_and_costs_hp.csv"))

    return total_system_costs_2050_df, consumption_savings_tot_df, complete_system_costs_2050_df


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
                   x_label=f"Energy savings through home insulation {unit}",
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


def plot_comparison_savings(df, x, y, save, col_for_size, format_y=lambda y, _: '{:.0f}'.format(y), format_x=lambda x, _: '{:.0f}'.format(x),
                            smallest_size=100,
                            biggest_size=300, fontsize=10, y_min=0, y_max=None, x_min=0, x_max=None,
                            unit="TWh", coordinates=None, remove_legend=False, s_min=None, s_max=None):
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
    if s_min is None:
        s_min, s_max = min(relative_size), max(relative_size)
    size = [smallest_size + (biggest_size - smallest_size)/(s_max - s_min) * (s - s_min) for s in relative_size]
    if y_max is None:
        x_max, y_max = df[x].max() * 1.1, df[y].max() * 1.1

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
        title = "Stock Heat pump \n"
    else:
        title = f"Energy savings through switch to heat pumps {unit} \n"
    if remove_legend:
        title = ""
    if not remove_legend:
        ax = format_ax(ax,
                       title=title,
                       x_label=f"Energy savings through home insulation",
                       format_y=format_y, format_x=format_x,
                       y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max,
                       loc_title="left", c_title="black", loc_xlabel="right")
    else:
        ax = format_ax(ax,
                       # title="Comparison savings (TWh)",
                       title=title,
                       x_label="",
                       format_y=format_y, format_x=format_x,
                       y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max,
                       loc_title="left", c_title="black", loc_xlabel="right", fontsize=19)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    kw = dict(prop="sizes", num=3, func=lambda s: s_min + (s - smallest_size) * (s_max - s_min) / (biggest_size - smallest_size))
    # handles, labels = scatter.legend_elements(prop="sizes")
    # legend2 = ax.legend(handles, labels, loc="upper right", title="Sizes")
    if col_for_size == "Total costs":
        title = "Total system costs (Billion €)"
    else:
        title = col_for_size
    legend2 = ax.legend(*scatter.legend_elements(**kw), title=title, loc='upper left', bbox_to_anchor=(1, 0.5), frameon=False)
    if remove_legend:
        ax.get_legend().remove()

    save_fig(fig, save=save)


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

            annualized_new_investment_df = output["Annualized new investments (1e9€/yr)"]
            annualized_new_energy_capacity_df = output["Annualized costs new energy capacity (1e9€/yr)"]
            functionment_costs_df = output["System functionment (1e9€/yr)"]
            total_system_costs, total_operational_costs = process_total_costs(annualized_new_investment_df, annualized_new_energy_capacity_df,
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
                                         y_label="Total costs (Billion €)",
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
                        carbon_constraint=True, eoles=True, year=2050, health=True):
    """
    Calculates total system (new) costs over the considered time period. This adds investment annuity for each year when the investment was present.
    Remark: we only include new costs, not costs of historical capacities.
    :param annualized_new_investment_df: pd.DataFrame
        Annualized costs of new investment done during the considered year.
    :param annualized_new_energy_capacity_df: pd.DataFrame
        Annualized costs of new energy capacity investment done during the considered year.
    :param functionment_costs_df: pd.DataFrame
        Functionment cost of the system for one year.
    :param eoles: bool
        Indicate whether we have optimized the energy system as well through the EOLES module
    :return:
    """
    annualized_new_investment_df_copy = annualized_new_investment_df.copy().loc[:,:year]
    annualized_new_energy_capacity_df_copy = annualized_new_energy_capacity_df.copy().loc[:,:year]
    functionment_costs_df_copy = functionment_costs_df.copy().loc[:,:year]
    # TODO: j'ai changé avec le fait d'ajouter l'année 2025
    if 2025 in annualized_new_investment_df_copy.columns:
        dict_count = {2025: 6 * 5, 2030: 5 * 5, 2035: 4 * 5, 2040: 3 * 5, 2045: 2 * 5, 2050: 5}
    else:
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
            if health:
                total_costs = elec_inv + heater_inv + insulation_inv + functionment_cost + health_costs
            else:
                total_costs = elec_inv + heater_inv + insulation_inv + functionment_cost
            total_system_costs = pd.Series(
                index=["Investment electricity costs", "Investment heater costs", "Investment insulation costs",
                       "Functionment costs", "Health costs", "Total costs"],
                data=[elec_inv, heater_inv, insulation_inv, functionment_cost, health_costs, total_costs])
        else:  # we only include ResIRF costs
            if health:
                total_costs = heater_inv + insulation_inv + functionment_cost + health_costs
            else:
                total_costs = heater_inv + insulation_inv + functionment_cost
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
            if health:
                total_costs = elec_inv + heater_inv + insulation_inv + functionment_cost + health_costs + carbon_cost
            else:
                total_costs = elec_inv + heater_inv + insulation_inv + functionment_cost + carbon_cost
            total_system_costs = pd.Series(
                index=["Investment electricity costs", "Investment heater costs", "Investment insulation costs",
                       "Functionment costs", "Health costs", "Carbon cost", "Total costs"],
                data=[elec_inv, heater_inv, insulation_inv, functionment_cost, health_costs, carbon_cost, total_costs])
        else:
            if health:
                total_costs = heater_inv + insulation_inv + functionment_cost + health_costs + carbon_cost
            else:
                total_costs = heater_inv + insulation_inv + functionment_cost + carbon_cost
            total_system_costs = pd.Series(
                index=["Investment heater costs", "Investment insulation costs",
                       "Functionment costs", "Health costs", "Carbon cost", "Total costs"],
                data=[heater_inv, insulation_inv, functionment_cost, health_costs, carbon_cost, total_costs])
    return total_system_costs, functionment_costs_df_copy.drop(index=["health_costs"]).sum(axis=1)


def aggregate_capacities(df):
    tmp = df.copy()
    offshore = tmp.loc[['offshore_f','offshore_g'],:].sum()
    offshore = offshore.to_frame().T.rename(index={0: 'offshore'})

    peaking_plants = tmp.loc[['ocgt','ccgt','h2_ccgt'],:].sum()
    peaking_plants = peaking_plants.to_frame().T.rename(index={0: 'peaking plants'})

    pv = tmp.loc[['pv_g','pv_c'],:].sum()
    pv = pv.to_frame().T.rename(index={0: 'pv'})

    battery = tmp.loc[['battery1','battery4'],:].sum()
    battery = battery.to_frame().T.rename(index={0: 'battery'})

    hydro = tmp.loc[['river','lake'],:].sum()
    hydro = hydro.to_frame().T.rename(index={0: 'hydro'})

    tmp = pd.concat([tmp, offshore, peaking_plants, pv, battery, hydro], axis=0).drop(index=['offshore_f','offshore_g','ocgt','ccgt','h2_ccgt','pv_g','pv_c','battery1','battery4','river','lake'])
    tmp = tmp.drop(index=['coal', 'uiom', 'CTES'])
    return tmp

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


def plot_load_profile(hourly_generation1, hourly_generation2, date_start, date_end, profile, label1, label2, climate=2006, save_path=None,
                      y_min=None, y_max=None, x_min=None, x_max=None):
    hourly_generation_subset1 = hourly_generation1.copy()
    hourly_generation_subset1["date"] = hourly_generation_subset1.apply(
        lambda row: datetime.datetime(climate, 1, 1, 0) + datetime.timedelta(hours=row["hour"]),
        axis=1)
    hourly_generation_subset1 = hourly_generation_subset1.set_index("date")

    hourly_generation_subset1 = hourly_generation_subset1.loc[date_start: date_end, :]  # select week of interest

    hourly_generation_subset2 = hourly_generation2.copy()
    hourly_generation_subset2["date"] = hourly_generation_subset2.apply(
        lambda row: datetime.datetime(climate, 1, 1, 0) + datetime.timedelta(hours=row["hour"]),
        axis=1)
    hourly_generation_subset2 = hourly_generation_subset2.set_index("date")

    hourly_generation_subset2 = hourly_generation_subset2.loc[date_start: date_end, :]  # select week of interest

    if save_path is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))

    hourly_generation_subset1[[profile]].squeeze().plot(ax=ax, style='-', c='red', label=label1)
    hourly_generation_subset2[[profile]].squeeze().plot(ax=ax, style='-', c='blue', label=label2)
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

    # format_legend(ax)
    # ax.get_legend().remove()

    save_fig(fig, save=save_path)


def plot_typical_week(hourly_generation, date_start, date_end, climate=2006, methane=True, hydrogen=False, save_path=None,
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

    hourly_generation_subset["hydrogen charging"] = - hourly_generation_subset["hydrogen_in"]
    hourly_generation_subset["hydrogen discharging"] = hourly_generation_subset["hydrogen"]

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
        sub = ["nuclear", "wind", "pv", "hydro", "battery charging", "battery discharging", "phs discharging", "phs charging", "peaking plants",
             "electrolysis", "methanation", "methane"]
    else:
        sub = ["nuclear", "wind", "pv", "hydro", "battery charging", "battery discharging", "phs discharging", "phs charging", "peaking plants",
             "electrolysis", "methanation"]

    if hydrogen:  # we add the display of hydrogen
        sub  = sub + ["hydrogen charging", "hydrogen discharging"]

    prod = hourly_generation_subset[sub]
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
              rotation=None, y_min=None, y_max=None, x_min=None, x_max=None, loc_title=None, loc_xlabel=None, c_title=None,
              fontsize=None):
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

    if fontsize is not None:  # additional condition for the scatter plot graph for paper
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

    if x_label is not None:
        if loc_xlabel is not None:
            if fontsize is not None:
                ax.set_xlabel(x_label, loc=loc_xlabel, fontsize=fontsize)
            else:
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
            if fontsize is not None:
                ax.set_title(title, loc=loc_title, color=c_title, fontsize=fontsize)
            else:
                ax.set_title(title, loc=loc_title, color=c_title)
        else:
            ax.set_title(title)

    return ax


def format_ax_new(ax, y_label=None, title=None, format_x=None,
                  format_y=lambda y, _: y, ymin=None, ymax=None, xinteger=True, xmin=None, x_max=None, loc_title=None,
                  c_title=None):
    """

    Parameters
    ----------
    y_label: str
    format_y: function
    ymin: float or None
    xinteger: bool
    title: str, optional

    Returns
    -------

    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    # ax.spines['bottom'].set_linewidth(2)

    ax.spines['left'].set_visible(True)
    # ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(which=u'both', length=0)
    ax.yaxis.set_tick_params(which=u'both', length=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
    if format_x is not None:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_x))

    if y_label is not None:
        ax.set_ylabel(y_label)

    if title is not None:
        if loc_title is not None:
            ax.set_title(title, loc=loc_title, color=c_title)
        else:
            ax.set_title(title)

    if xmin is not None:
        ax.set_xlim(xmin=xmin)
        _, x_max = ax.get_xlim()
        ax.set_xlim(xmax=x_max * 1.1)

    if ymin is not None:
        ax.set_ylim(ymin=0)
        _, y_max = ax.get_ylim()
        ax.set_ylim(ymax=y_max * 1.1)

    if ymax is not None:
        ax.set_ylim(ymax=ymax, ymin=ymin)

    if xinteger:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

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


def make_cluster_scatterplot(df, x, y, y_label, colors=None, format_y=lambda y, _: '{:.0f}'.format(y), save=None,
                                rotation=0, dict_legend=None, coordinates=None):
    list_keys = df.index.get_level_values(1).unique()
    n_columns = int(len(list_keys)) // 2
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(12.8, 9.6), sharex='all', sharey='all')
    handles, labels = None, None

    for k in range(n_rows * n_columns):
        row = k // n_columns
        column = k % n_columns
        if n_rows * n_columns == 1:  # in this case, we have a single plot
            ax = axes
        else:
            ax = axes[row, column]

        try:
            key = list_keys[k]
            df_temp = df.loc[df.index.get_level_values(1) == key].unstack()

            scatter = ax.scatter(x=df_temp.T[x], y=df_temp.T[y])

            if coordinates is None:
                for scenario, v in df.iterrows():
                    ax.annotate(scenario, xy=(v[x], v[y]), xytext=(20, -5), textcoords="offset points")
            else:
                for scenario, v in df.iterrows():
                    if scenario in coordinates.keys():
                        ax.annotate(scenario, xy=(v[x], v[y]), xytext=coordinates[scenario], textcoords="offset points")
                    else:
                        ax.annotate(scenario, xy=(v[x], v[y]), xytext=(20, -5), textcoords="offset points")

            ax = format_ax_new(ax, format_y=format_y, xinteger=True)
            if k ==0:
                ax.set_ylabel(y_label, color='dimgrey', fontsize=20)
            # ax = format_ax(ax, format_y=format_y, ymin=0, xinteger=True)
            ax.spines['left'].set_visible(False)
            # ax.set_ylim(ymax=y_max)
            # ax.set_ylim(ymin=y_min)
            ax.set_xlabel('')

            # if hline:
            #     ax.axhline(y=0)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
            ax.tick_params(axis='both', which='major', labelsize=19)

            title = key
            if isinstance(key, tuple):
                title = '{}-{}'.format(key[0], key[1])
            ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=16)

            if k == 0:
                handles, labels = ax.get_legend_handles_labels()
                labels = [l.replace('_', ' ') for l in labels]
            ax.get_legend().remove()

        except IndexError:
            ax.axis('off')

    save_fig(fig, save=save)


def colormap(df, custom_cmap=None, format_y=lambda y, _: '{:.0f}'.format(y), save=None, y_label=None, rotation=0,
             title=None, percent=False):
    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(18, 9.6))

    if custom_cmap is None:
        custom_cmap = sns.color_palette("viridis", as_cmap=True, n_colors=len(df)).reversed()
        if percent:
            sns.heatmap(df, ax=ax, cmap=custom_cmap, annot=True, fmt='.1f', annot_kws={"size": 16})
            for t in ax.texts:
                t.set_text(t.get_text() + " %")
        else:
            sns.heatmap(df, ax=ax, cmap=custom_cmap, annot=True, fmt='.0f', annot_kws={"size": 16})
        # ax.xaxis.tick_top()
        # ax.xaxis.set_label_position('top')
    else:
        if percent:
            sns.heatmap(df, ax=ax, cmap=custom_cmap, annot=True, fmt='.1f', annot_kws={"size": 16})
            for t in ax.texts:
                t.set_text(t.get_text() + " %")
        else:
            sns.heatmap(df, ax=ax, cmap=custom_cmap, annot=True, fmt='.0f', annot_kws={"size": 16})
    ax.tick_params(labelbottom=False, labeltop=True)

    # ax = format_ax_new(ax, format_y=format_y, xinteger=True)
    # ax.set_ylabel(y_label, color='dimgrey', fontsize=10, labelpad=100)
    # ax = format_ax(ax, format_y=format_y, ymin=0, xinteger=True)
    # ax.spines['left'].set_visible(False)
    ax.set_xlabel('')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('')

    plt.yticks(rotation=0)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
    # ax.tick_params(axis='both', which='major', labelsize=19)

    ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=16)

    save_fig(fig, save=save)


def make_clusterstackedbar_plot(df, groupby, y_label, subset=None, colors=None, format_y=lambda y, _: '{:.0f}'.format(y), save=None,
                                rotation=0, display_title=True, dict_legend=None, scatter=None, ref=None, drop=False, hline=False, ranking_exogenous_scenario=None,
                                ranking_policy_scenario=None, legend_loc='lower', reorder_labels=False):

    list_keys = list(df.columns)
    if ranking_exogenous_scenario is not None:  # we modify the order of display of the graphs
        assert len(list_keys) == len(ranking_exogenous_scenario)
        list_keys = ranking_exogenous_scenario
    if subset is not None:
        tmp = df[df > 0]
        y_max = tmp[tmp.index.get_level_values(0).isin(subset)].groupby([i for i in tmp.index.names if i != groupby]).sum().max().max() * 1.1

        tmp = df[df < 0]
        y_min = tmp[tmp.index.get_level_values(0).isin(subset)].groupby([i for i in tmp.index.names if i != groupby]).sum().min().min() * 1.1
    else:
        y_max = df[df > 0].groupby([i for i in df.index.names if i != groupby]).sum().max().max() * 1.1
        y_min = df[df < 0].groupby([i for i in df.index.names if i != groupby]).sum().min().min() * 1.1

    n_columns = int(len(list_keys))
    n_rows = 1
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(12.8, 9.6), sharex='all', sharey='all')
    handles, labels = None, None
    for k in range(n_rows * n_columns):

        column = k % n_columns
        if n_rows * n_columns == 1:  # in this case, we have a single plot
            ax = axes
        else:
            ax = axes[column]

        try:
            key = list_keys[k]
            df_temp = df[key].unstack(groupby)
            if ranking_policy_scenario is not None:  # we reorder to modify the display
                df_temp = df_temp.loc[ranking_policy_scenario]
            if drop:
                df_temp = df_temp.drop(ref)
            if subset is not None:
                if colors is not None:
                    df_temp[subset].plot(ax=ax, kind='bar', stacked=True, linewidth=0, color=colors)
                else:
                    df_temp[subset].plot(ax=ax, kind='bar', stacked=True, linewidth=0, color=colors)
            else:
                if colors is not None:
                    df_temp.plot(ax=ax, kind='bar', stacked=True, linewidth=0, color=colors)
                else:
                    df_temp.plot(ax=ax, kind='bar', stacked=True, linewidth=0, color=colors)

            if scatter is not None:
                df_temp[scatter].plot(ax=ax, style='.', c='black', ms=20)
            ax = format_ax_new(ax, format_y=format_y, xinteger=True)
            if k ==0:
                ax.set_ylabel(y_label, color='dimgrey', fontsize=20)
            # ax = format_ax(ax, format_y=format_y, ymin=0, xinteger=True)
            ax.spines['left'].set_visible(False)
            ax.set_ylim(ymax=y_max)
            ax.set_ylim(ymin=y_min)
            ax.set_xlabel('')

            if hline:
                ax.axhline(y=0)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
            ax.tick_params(axis='both', which='major', labelsize=19)

            title = key
            if isinstance(key, tuple):
                title = '{}-{}'.format(key[0], key[1])
            if display_title:
                ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=16)
            else:
                ax.set_title('')

            if k == 0:
                handles, labels = ax.get_legend_handles_labels()
                labels = [l.replace('_', ' ') for l in labels]
            ax.get_legend().remove()

        except IndexError:
            ax.axis('off')

    if dict_legend is not None:
        labels = [dict_legend[e] if e in dict_legend.keys() else e for e in labels]

    if reorder_labels:
        labels = labels[::-1]
        handles = handles[::-1]

    if legend_loc == 'lower':
        fig.legend(handles, labels, loc='lower center', frameon=False, ncol=3,
                   bbox_to_anchor=(0.5, -0.1))
    else:
        fig.legend(handles, labels, loc='center left', frameon=False, ncol=1,
                   bbox_to_anchor=(1, 0.5))

    # fig.suptitle(title, fontsize=18, y=1.02)

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


def waterfall_chart(df, colors=None, rotation=0, save=None, format_y=lambda y, _: '{:.0f}'.format(y), title=None,
                    y_label=None, hline=False, dict_legend=None, total=True, unit='B€', float_precision=0, neg_offset=None,
                    pos_offset=None, df_max=None, df_min=None):
    if isinstance(df, pd.DataFrame):
        df = df.squeeze()
    if dict_legend is not None:
        new_index = {e: dict_legend[e] if e in dict_legend.keys() else e for e in df.index}
        df = df.rename(new_index)
        if df_max is not None:
            df_max = df_max.rename(new_index)
        if df_min is not None:
            df_min = df_min.rename(new_index)

    blank = df.cumsum().shift(1).fillna(0)  # will be used as start point for the bar plot
    if total:
        blank[-1] = 0  # we display the total at the end
    # blank[-1] = 0
    fig, ax = plt.subplots(1, 1, figsize=(14, 9.6))
    if colors is not None:
        df.plot(kind='bar', stacked=True, bottom=blank, title=None, ax=ax, color=[colors[i] for i in df.index])
    else:
        df.plot(kind='bar', stacked=True, bottom=blank, title=None, ax=ax)

    # Calculate and plot error bars if df_min and df_max are provided
    if df_min is not None and df_max is not None:
        # Calculate error margins
        errors_positive = df_max.squeeze() - df
        errors_negative = df - df_min.squeeze()
        errors = [errors_negative, errors_positive]

        # The x positions for the error bars
        x_positions = range(len(df))
        y_positions = df.cumsum()
        y_positions['Total costs'] = df['Total costs']

        # Plot error bars dotted lines
        eb = ax.errorbar(x=x_positions, y=y_positions, yerr=errors, fmt='none', ecolor='darkgrey', elinewidth=2, capsize=5,
                    capthick=2, ls='--')

        eb[-1][0].set_linestyle('--')

    y_height = df.cumsum().shift(1).fillna(0)
    max = df.max()
    if neg_offset is None:
        neg_offset = max / 20
    if pos_offset is None:
        pos_offset =  max / 50

    # Start label loop
    loop = 0
    for (index, val) in df.iteritems():
        # For the last item in the list, we don't want to double count
        if val == df.iloc[-1]:
            if total:
                y = y_height[loop]
                y += pos_offset  # in the case of the final item, we do not want a negative offset even if value is negative
            else:
                y = y_height[loop] + val
                # Determine if we want a neg or pos offset
                if val > 0:
                    y += pos_offset
                else:
                    y -= neg_offset
        else:
            y = y_height[loop] + val
            # Determine if we want a neg or pos offset
            if val > 0:
                y += pos_offset
            else:
                y -= neg_offset
        # if loop > 0:
        if float_precision == 0:
            ax.annotate("{:+,.0f} {}".format(val, unit), (loop, y), ha="center")
        else:
            ax.annotate("{:+,.1f} {}".format(val, unit), (loop, y), ha="center")
        loop += 1

    if blank.max() > 0:  # total est True quand on fait les graphes pour les coûts, et False quand on fait les graphes pour les capacités
        if df_max is not None:
            y_max = (y_positions + errors_positive).max() * 1.1
        else:
            if total:
                y_max = blank.max() * 1.1
            else:
                y_max = (blank + df).max() * 3
    else:
        y_max = 5 * 1.1
    if total:
        y_min = blank.min() * 2
    else:
        y_min = blank.min() * 1.2
    ax.spines['left'].set_visible(False)
    ax.set_ylim(ymax=y_max)
    ax.set_ylim(ymin=y_min)
    ax.set_xlabel('')
    ax = format_ax_new(ax, format_y=format_y, xinteger=True)

    if title is not None:
        if total:
            ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=16)
        else:
            ax.set_title(title, fontweight='bold', color='dimgrey', pad=-3, fontsize=16)

    if y_label is not None:
        ax.set_ylabel(y_label, color='dimgrey', fontsize=20)

    if hline:
        ax.axhline(y=0, c='black')

    # if legend_loc == 'lower':
    #     fig.legend(handles, labels, loc='lower center', frameon=False, ncol=3,
    #                bbox_to_anchor=(0.5, -0.1))
    # else:
    #     fig.legend(handles, labels, loc='center left', frameon=False, ncol=1,
    #                bbox_to_anchor=(1, 0.5))

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
    ax.tick_params(axis='both', which='major', labelsize=18)

    save_fig(fig, save=save)


def plot_ldmi_method(channel, CO2, start, end, colors=None, rotation=0, save=None, format_y=lambda y, _: '{:.0f}'.format(y),
                     title=None, y_label="Emissions "):
    """Plots LDMI decomposition method."""
    new_index = []
    for c in channel.index:
        if len(c.split(' ')) > 1:  # we have two words
            new_index.append(c.split(' ')[0] + ' \n ' + c.split(' ')[1])
        else:
            new_index.append(c)
    channel.index = new_index
    # channel = channel.reindex(['Surface', 'Insulation', 'Share', 'Heating \n intensity', 'Emission content'])
    tmp = pd.concat([channel, CO2[[start, end]]])
    tmp = tmp.reindex([start] + channel.index.to_list() + [end])
    tmp.index = tmp.index.astype(str)
    percent = tmp.copy()
    percent = percent / percent.iloc[0] * 100
    percent.iloc[-1] = (-tmp.iloc[0] + tmp.iloc[-1]) / tmp.iloc[0] * 100

    blank = tmp.cumsum().shift(1).fillna(0)  # will be used as start point for the bar plot
    blank[-1] = 0
    fig, ax = plt.subplots(1, 1, figsize=(14, 9.6))
    if colors is not None:
        tmp.plot(kind='bar', stacked=True, bottom=blank, title=None, ax=ax, color=[colors[i] for i in tmp.index])
    else:
        tmp.plot(kind='bar', stacked=True, bottom=blank, title=None, ax=ax)

    y_height = tmp.cumsum().shift(1).fillna(0)
    max = tmp.max()
    neg_offset, pos_offset = max / 20, max / 50

    # Start label loop
    loop = 0
    for ((index, val), (index2, val2)) in zip(percent.iteritems(), tmp.iteritems()):
        # For the last item in the list, we don't want to double count
        if val2 == tmp.iloc[-1]:
            y = y_height[loop]
            y += pos_offset  # in the case of the final item, we do not want a negative offset even if value is negative
        else:
            y = y_height[loop] + val2
            # Determine if we want a neg or pos offset
            if val > 0:
                y += pos_offset
            else:
                y -= neg_offset
        if loop > 0:
            ax.annotate("{:+,.0f} %".format(val), (loop, y), ha="center")
        loop += 1

    y_max = blank.max() * 1.1
    y_min = blank.min() * 1.1
    ax.spines['left'].set_visible(False)
    ax.set_ylim(ymax=y_max)
    ax.set_ylim(ymin=y_min)
    ax.set_xlabel('')
    ax = format_ax_new(ax, format_y=format_y, xinteger=True)

    if title is not None:
        ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6, fontsize=16)

    if y_label is not None:
        ax.set_ylabel(y_label, color='dimgrey', fontsize=20)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
    ax.tick_params(axis='both', which='major', labelsize=18)

    save_fig(fig, save=save)


if __name__ == '__main__':
    total_system_costs_2050_df = colormap_simulations(overall_folder=Path('outputs') / Path('20231211'),
                                                      config_ref=None,
                                                      # config_ref= {'biogas': 'S3',
                                                      #              'capacity': 'N1',
                                                      #              'demand': 'Reference',
                                                      #              'profile': 'Reference',
                                                      #              'weather': 'Reference'},
                                                      save_path=Path('outputs') / Path('20231211'),
                                                      # subset_configs=['Ban', 'BanRef', 'BanNoPolicy'],
                                                      subset_configs=['Ambitious', 'Ban'],
                                                      percent=True,
                                                      # reorder=['Ambitious', 'Ban'],
                                                      dict_scenario={
                                                          'Ambitious': 'Package 2024',
                                                          'BanRef': 'Package 2021 + Ban',
                                                          'BanNoPolicy': 'No Policy + Ban'
                                                      },
                                                      dict_config_demandsupply={
                                                          'Reference': 'Reference',
                                                          'Elasticity-': 'Lower Elasticity HP',
                                                          'LearningHP+': 'Technical Progress HP',
                                                          'biogasBiogas-': 'Lower Biogas Potential',
                                                          'capaNuc-': 'Lower Nuclear Potential',
                                                          'capaRen-': 'Lower Renewable Potential',
                                                          'capaRen+': 'Higher Renewable Potential',
                                                          'costscostsREN+': 'Higher Renewable Costs',
                                                          'demandReindustrialisation': 'Higher Electricity Demand',
                                                          'demandSobriete': 'Lower Electricity Demand',
                                                          'weather2012': 'Colder Weather'
                                                      }
                                                        )