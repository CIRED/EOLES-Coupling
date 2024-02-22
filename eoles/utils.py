import dataclasses
import math
from importlib import resources
from pathlib import Path
import pandas as pd
from numpy import log

from eoles.write_output import plot_ldmi_method
from eoles.inputs.resources import resources_data

pd.options.mode.chained_assignment = None  # default='warn'
import os
import json
from matplotlib import pyplot as plt
from pyomo.environ import value
import datetime
from copy import deepcopy
from typing import Union
from pickle import load
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
# sns.set_theme(context="talk", style="white", font_scale=1.3)


def get_pandas(path, func=lambda x: pd.read_csv(x)):
    """Function used to read input data"""
    path = Path(path)
    with resources.path(str(path.parent).replace('/', '.'), path.name) as df:
        return func(df)


def get_config(spec=None) -> dict:
    if spec is None:
        with resources.path('eoles.inputs.config', 'config.json') as f:
            with open(f) as file:
                return json.load(file)
    else:
        with resources.path('eoles.inputs.config', f'config_{spec}.json') as f:
            with open(f) as file:
                return json.load(file)


# Heating need

def process_heating_need(dict_heat, climate):
    """Transforms index of heating need into number of hours.
    :param heating_need: pd.DataFrame
        Includes hourly heating need
    :param climate: int
        Year to start counting hours"""
    for key in dict_heat.keys():
        heating_need = dict_heat[key]
        new_index_hour = [int((e - datetime.datetime(climate, 1, 1, 0)).total_seconds() / 3600) for e in
                          heating_need.index]  # transform into number of hours
        heating_need.index = new_index_hour
        heating_need = heating_need.sort_index(ascending=True)
        heating_need = heating_need * 1e-6  # convert kWh to GWh
        dict_heat[key] = heating_need
    return dict_heat


TEMP_SINK = 55


def calculate_hp_cop(climate):
    """Calculates heat pump coefficient based on renewable ninja data."""
    path_weather = Path("eoles") / "inputs" / "hourly_profiles" / "ninja_weather_country_FR_merra-2_population_weighted.csv"
    weather = get_pandas(path_weather,
                         lambda x: pd.read_csv(x, header=2))
    weather["date"] = weather.apply(lambda row: datetime.datetime.strptime(row["time"], '%Y-%m-%d %H:%M:%S'), axis=1)
    weather = weather.loc[(weather.date >= datetime.datetime(climate, 1, 1, 0)) & (
                weather.date <= datetime.datetime(climate, 12, 31, 23))]
    weather["delta_temperature"] = TEMP_SINK - weather["temperature"]
    weather["hp_cop"] = 6.81 - 0.121 * weather["delta_temperature"] + 0.00063 * weather[
        "delta_temperature"] ** 2  # formula for HP performance coefficient
    weather = weather[["date", "hp_cop"]].set_index("date")
    new_index_hour = [int((e - datetime.datetime(climate, 1, 1, 0)).total_seconds() / 3600) for e in
                      weather.index]  # transform into number of hours
    weather.index = new_index_hour
    weather = weather.sort_index(ascending=True)
    return weather


def heating_hourly_profile(method, percentage=None):
    """Creates hourly profile"""
    assert method in ["very_extreme", "extreme", "medium", "valentin", "valentin_modif", "BDEW"]
    heat_load = get_pandas("eoles/inputs/hourly_profiles/heat_load_profile.csv", lambda x: pd.read_csv(x))
    if method == "very_extreme":
        hourly_profile_test = pd.Series(
            [0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0],
            index=pd.TimedeltaIndex(range(0, 24), unit='h'))  # extreme
    elif method == "extreme":
        hourly_profile_test = pd.Series(
            [0.1, 0, 0, 0, 0, 0, 0.05, 0.05, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            index=pd.TimedeltaIndex(range(0, 24), unit='h'))
    elif method == "medium":
        L = [2, 2, 1, 1, 1, 1, 2, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 4, 3, 3]  # profil plus smooth
        hourly_profile_test = pd.Series([e / sum(L) for e in L], index=pd.TimedeltaIndex(range(0, 24), unit='h'))
    elif method == "valentin":
        L = [1850, 1750, 1800, 1850, 1900, 1950, 2050, 2120, 2250, 2100, 2000, 1850, 1700, 1550, 1600, 1650, 1800, 2000,
             2100, 2150, 2200, 2150, 2100, 2000]  # profil issu de Valentin
        # hourly_profile_test = pd.Series([e / sum(L) for e in L], index=pd.TimedeltaIndex(range(0, 24), unit='h'))
        hourly_profile_test = pd.Series(heat_load["residential_space_percentage_valentin"].tolist(), index=pd.TimedeltaIndex(range(0, 24), unit='h'))
    elif method == "BDEW":  # method from Zeyen
        hourly_profile_test = pd.Series(heat_load["residential_space_weekday_percentage_BDEW"].tolist(),
                                        index=pd.TimedeltaIndex(range(0, 24), unit='h'))
    elif method == "valentin_modif":  # utile pour tester la sensibilité au choix du profil horaire
        L = [1850, 1750, 1800, 1850, 1900, 1950, 2050, 2120, 2250, 2100, 2000, 1850, 1700, 1550, 1600, 1650, 1800, 2000,
             2100, 2150, 2200, 2150, 2100, 2000]  # profil issu de Valentin
        hourly_profile_test = pd.Series([e / sum(L) for e in L], index=pd.TimedeltaIndex(range(0, 24), unit='h'))
        threshold = 0.042
        modif = 0.04*percentage
        assert hourly_profile_test[hourly_profile_test > 0.042].shape[0] == 12
        hourly_profile_test[hourly_profile_test > 0.042] = hourly_profile_test[hourly_profile_test > threshold] + modif
        hourly_profile_test[hourly_profile_test <= 0.042] = hourly_profile_test[hourly_profile_test <= threshold] - modif
    return hourly_profile_test


def load_evolution_data(config):
    """Load necessary data for the social planner trajectory"""
    # Load historical data
    existing_capacity_historical = get_pandas("eoles/inputs/historical_data/existing_capacity_historical.csv",
                                              lambda x: pd.read_csv(x, index_col=0))  # GW
    existing_charging_capacity_historical = get_pandas("eoles/inputs/historical_data/existing_charging_capacity_historical.csv",
                                                       lambda x: pd.read_csv(x, index_col=0))  # GW
    existing_energy_capacity_historical = get_pandas("eoles/inputs/historical_data/existing_energy_capacity_historical.csv",
                                                     lambda x: pd.read_csv(x, index_col=0))  # GW
    maximum_capacity_evolution = get_pandas(config["maximum_capacity_evolution"], lambda x: pd.read_csv(x, index_col=[0,1]))  # GW
    maximum_capacity_evolution = maximum_capacity_evolution.loc[config["maximum_capacity_evolution_scenario"]]  # we select the scenario we are interested in

    capex_annuity_fOM_historical = get_pandas("eoles/inputs/historical_data/capex_annuity_fOM_historical.csv",
                                              lambda x: pd.read_csv(x, index_col=0).squeeze())
    capex_annuity_historical = get_pandas("eoles/inputs/historical_data/capex_annuity_historical.csv",
                                              lambda x: pd.read_csv(x, index_col=0).squeeze())
    storage_annuity_historical = get_pandas("eoles/inputs/historical_data/storage_annuity_historical.csv",
                                              lambda x: pd.read_csv(x, index_col=0).squeeze())

    # Import evolution of tertiary and ECS gas demand
    heating_gas_demand_RTE_timesteps = get_pandas("eoles/inputs/demand/heating_gas_demand_tertiary_timesteps.csv",
                                                  lambda x: pd.read_csv(x, index_col=0).squeeze())
    ECS_gas_demand_RTE_timesteps = get_pandas("eoles/inputs/demand/ECS_gas_demand_timesteps.csv",
                                              lambda x: pd.read_csv(x, index_col=0).squeeze())

    return existing_capacity_historical, existing_charging_capacity_historical, existing_energy_capacity_historical,\
           maximum_capacity_evolution, heating_gas_demand_RTE_timesteps, ECS_gas_demand_RTE_timesteps, \
           capex_annuity_fOM_historical, capex_annuity_historical, storage_annuity_historical

### Defining the model

def process_RTE_demand(config, year, demand, scenario, method, calibration=False, hourly_residential_heating_RTE=None):
    """Create electricity demand profile, where we have excluded the residential heating demand, based on RTE projections.
    """
    if not calibration:  # classical setting, we get projected values from RTE scenarios.
        demand_noP2G_RTE_timesteps = get_pandas(config["demand_noP2G_RTE_timesteps"],
                                                lambda x: pd.read_csv(x, index_col=[0,1]).squeeze())
        # demand_noP2G_RTE = demand_noP2G_RTE_timesteps[year]  # in TWh
        demand_noP2G_RTE = demand_noP2G_RTE_timesteps.loc[config["demand_scenario"]]
        demand_noP2G_RTE = demand_noP2G_RTE[year]  # get specific potential for year of interest

        # # Demand for EV
        # demand_ev_timesteps = get_pandas(config["demand_ev_timesteps"],
        #                                                       lambda x: pd.read_csv(x, index_col=[0, 1]).squeeze())
        #
        # # TODO: ajouter ce profile de demande EV dans le preprocessing de la demande
        # demand_ev = demand_ev_timesteps.loc[config["demand_scenario"]]
        # demand_ev = demand_ev[year]  # get specific potential for year of interest
        # demand_ev_ref = demand_ev[2035]  # get reference value for 2035 which is the year of reference for the demand profile

        assert math.isclose(demand.sum(), 580 * 1e3), "Total yearly demand is not correctly calculated."
        adjust_demand = (demand_noP2G_RTE * 1e3 - 580 * 1e3) / 8760  # 580TWh is the total of the profile we use as basis for electricity hourly demand (from RTE), c'est bien vérifié
        demand_elec_RTE_noP2G = demand * (demand_noP2G_RTE / 580)  # new adjustment for demand profile

        demand_residential_heating_RTE_timesteps = get_pandas(config["demand_residential_heating_RTE_timesteps"],
                                                              lambda x: pd.read_csv(x, index_col=[0, 1]).squeeze())

        # demand_residential_heating = demand_residential_heating_RTE_timesteps[year]  # in TWh
        demand_residential_heating = demand_residential_heating_RTE_timesteps.loc[config["demand_scenario"]]
        demand_residential_heating = demand_residential_heating[year]  # get specific potential for year of interest

        hourly_residential_heating_RTE = create_hourly_residential_demand_profile(demand_residential_heating * 1e3,
                                                                                  method=method)  # TODO: a changer a priori, ce n'est plus le bon profil

        # TODO: a changer !! test pour l'impact sur le carbon content
        # demand_elec_RTE_no_residential_heating = demand_elec_RTE_noP2G - hourly_residential_heating_RTE * 38.5/43  # we remove residential electric demand
        demand_elec_RTE_no_residential_heating = demand_elec_RTE_noP2G - hourly_residential_heating_RTE  # we remove residential electric demand

    else:  # in this case, we take a historical demand chronic instead of projected RTE profile, so we do not need to readjust. Moreover, there is no power to gas for now.
        demand_elec_RTE_no_residential_heating = demand
        if hourly_residential_heating_RTE is not None:  # in this case, we also give a profile for electric residential heating through the coupling (which is necessary to calculate carbon content)
            demand_elec_RTE_no_residential_heating = demand_elec_RTE_no_residential_heating - hourly_residential_heating_RTE

    return demand_elec_RTE_no_residential_heating


def profile_ev(total_consumption):
    """Rescale profile for electric vehicule demand to match total consumption. Parameter total_consumption is in TWh."""
    demand_ev = pd.read_csv('eoles/inputs/demand_data_other/demand_transport2050.csv', index_col=0,
                            header=None).reset_index().rename(columns={0: 'vehicule', 1: 'hour', 2: 'demand'})

    # plot the demand profile for vehicule = 'light'

    demand_ev_light = demand_ev.loc[demand_ev.vehicule == 'light']
    demand_ev_light = demand_ev_light.drop(columns=['vehicule'])
    demand_ev_light = demand_ev_light.set_index('hour')

    tot = demand_ev_light.sum()
    demand_ev_light = demand_ev_light + (total_consumption * 1e3 - tot) / 8760 * demand_ev_light
    return demand_ev_light


def calculate_annuities_capex(discount_rate, capex, construction_time, lifetime):
    """Calculate annuities for energy technologies and renovation technologies based on capex data."""
    annuities = construction_time.copy()
    for i in annuities.index:
        annuities.at[i] = discount_rate * capex[i] * (
                discount_rate * construction_time[i] + 1) / (
                                  1 - (1 + discount_rate) ** (-lifetime[i]))
    return annuities


def calculate_annuities_storage_capex(discount_rate, storage_capex, construction_time, lifetime):
    """Calculate annuities for storage technologies based on capex data."""
    storage_annuities = storage_capex.copy()
    for i in storage_annuities.index:
        storage_annuities.at[i] = discount_rate * storage_capex[i] * (
                discount_rate * construction_time[i] + 1) / (
                                          1 - (1 + discount_rate) ** (-lifetime[i]))
    return storage_annuities


def calculate_annuities_renovation(linearized_renovation_costs, miscellaneous):
    """Be careful to units. Renovation costs are initially expressed in 1e9 € contrary to the rest of the costs !!"""
    renovation_annuities = linearized_renovation_costs.copy()
    for archetype in linearized_renovation_costs.index:
        renovation_annuities.at[archetype] = miscellaneous["discount_rate"] * linearized_renovation_costs[
            archetype] * 1e3 * (miscellaneous["discount_rate"] * miscellaneous["construction_time_renov"] + 1) / (
                                                     1 - (1 + miscellaneous["discount_rate"]) ** (
                                                 -miscellaneous["lifetime_renov"]))
    return renovation_annuities


def calculate_annuities_resirf(capex, lifetime, discount_rate):
    """

    :param capex: float
        Overnight cost of renovation and change of heat vector
    :param lifetime: int
        Lifetime of considered investment
    :param discount_rate: float
        Discount rate used in the annuity calculus
    :return:
    """
    return capex * discount_rate / (1 - (1 + discount_rate) ** (-lifetime))


def update_vom_costs_scc(vOM_init, scc, emission_rate):
    """Add emission cost related to social cost of carbon to the natural gas vOM cost.
    :param vOM_init: float
        Initial vOM in M€/GWh
    :param scc: int
        €/tCO2
    :param emission_rate: float
        tCO2/MWh. The default value is the one corresponding to natural gas.

    Returns
    vOM in M€/GWh  = €/kWh
    """
    return vOM_init + scc * emission_rate / 1000


def create_hourly_residential_demand_profile(total_consumption, method="RTE"):
    """Calculates hourly profile from total consumption, using either the methodology from Doudard (2018) or
    methodology from RTE."""
    assert method in ["RTE", "valentin", "BDEW"]
    if method == "RTE":
        percentage_hourly_residential_heating = get_pandas(
            "eoles/inputs/hourly_profiles/percentage_hourly_residential_heating_profile_RTE.csv",
            lambda x: pd.read_csv(x, index_col=0, header=None).squeeze(
                "columns"))
    elif method == "valentin":
        percentage_hourly_residential_heating = get_pandas(
            "eoles/inputs/hourly_profiles/percentage_hourly_residential_heating_profile_valentin.csv",
            lambda x: pd.read_csv(x, index_col=0, header=None).squeeze(
                "columns"))
    elif method == "BDEW":
        percentage_hourly_residential_heating = get_pandas(
            "eoles/inputs/hourly_profiles/percentage_hourly_residential_heating_profile_BDEW.csv",
            lambda x: pd.read_csv(x, index_col=0, header=None).squeeze(
                "columns"))
    else:
        percentage_hourly_residential_heating = get_pandas(
            "eoles/inputs/hourly_profiles/percentage_hourly_residential_heating_profile_doudard.csv",
            lambda x: pd.read_csv(x, index_col=0, header=None).squeeze(
                "columns"))
    hourly_residential_heating = percentage_hourly_residential_heating * total_consumption
    return hourly_residential_heating


def define_month_hours(first_month, nb_years, months_hours, hours_by_months):
    """
    Calculates range of hours for each month
    :param first_month: int
    :param nb_years: int
    :param months_hours: dict
    :param hours_by_months: dict
    :return:
    Dict containing the range of hours for each month considered in the model
    """
    j = first_month + 1
    for i in range(2, 12 * nb_years + 1):
        hour = months_hours[i - 1][-1] + 1  # get the first hour for a given month
        months_hours[i] = range(hour, hour + hours_by_months[j])
        j += 1
        if j == 13:
            j = 1
    return months_hours


### Processing output

def get_technical_cost(model, objective, scc, heat_oil, heat_wood, nb_years, carbon_content, wood_neutral):
    """Returns technical cost (social cost without CO2 emissions-related cost"""
    gene_ngas = sum(value(model.gene["natural_gas", hour]) for hour in model.h)   # GWh
    gene_coal = sum(value(model.gene['coal', hour]) for hour in model.h)  # GWh
    net_emissions = gene_ngas * carbon_content['natural_gas'] / 1000 + heat_oil * carbon_content['oil'] / 1000  + gene_coal * carbon_content['coal'] / 1000  # MtCO2
    if not wood_neutral:
        net_emissions += heat_wood * carbon_content['wood'] / 1000
        emissions = pd.Series({"natural_gas": gene_ngas * carbon_content['natural_gas'] / 1000 / nb_years, "Oil fuel": heat_oil * carbon_content['oil'] / 1000 / nb_years,
                               "Wood fuel": heat_wood * carbon_content['wood'] / 1000 / nb_years, 'Coal': gene_coal * carbon_content['coal'] / 1000 / nb_years})
    else:
        emissions = pd.Series({"natural_gas": gene_ngas * carbon_content['natural_gas'] / 1000 / nb_years, "Oil fuel": heat_oil * carbon_content['oil'] / 1000 / nb_years,
                               'Coal': gene_coal * carbon_content['coal'] / 1000 / nb_years})
    technical_cost = objective - net_emissions * scc / 1000
    return technical_cost, emissions


def extract_capacities(model):
    """Extracts capacities for all technology in GW"""
    list_tec = list(model.tec)
    capacities = pd.Series(index=list_tec, dtype=float)

    for tec in list_tec:
        capacities.loc[tec] = value(model.capacity[tec])

    return capacities


def extract_energy_capacity(model):
    """Extracts energy capacity for all storage technology, in GWh"""
    list_str = list(model.str)
    energy_capacity = pd.Series(index=list_str, dtype=float)

    for tec in list_str:
        energy_capacity.loc[tec] = value(model.energy_capacity[tec])
    return energy_capacity


def extract_charging_capacity(model):
    """Extracts energy capacity for all storage technology, in GWh"""
    list_str = list(model.str)
    charging_capacity = pd.Series(index=list_str, dtype=float)

    for tec in list_str:
        charging_capacity.loc[tec] = value(model.charging_capacity[tec])
    return charging_capacity


def extract_renovation_investment(model, existing_renovation_rate, linearized_renovation_costs, renovation_annuities, nb_years):
    """Investment in renovation (Billion euro), both annualized and non annualized"""
    investment = sum((value(model.renovation_rate[renov]) - existing_renovation_rate[renov]) * linearized_renovation_costs[renov] * nb_years for renov in
                          model.renovation)  # 1e9€  (a verifier, mais je crois que c'est bien l'unité de linearized_renovation_costs)
    annuity_investment = sum((value(model.renovation_rate[renov]) - existing_renovation_rate[renov]) * renovation_annuities[renov] * nb_years for renov in
                          model.renovation) / 1000
    return investment, annuity_investment


def extract_heater_investment(model, existing_capacity, annuities, nb_years):
    """Investment in heaters in Billion euro (annualized)"""
    list_tec = list(model.heat)
    investment_heater = sum(
                (value(model.capacity[tec]) - existing_capacity[tec]) * annuities[tec] * nb_years for tec in
                list_tec) / 1000
    return investment_heater


def extract_electricity_cost(model, existing_capacity, existing_energy_capacity, storage_annuities, annuities,  fOM,
                             vOM, nb_years):
    """Annualized costs in the electricity system, excluding investment costs in heat technologies and renovation."""
    all_but_heater = list(set(list(model.tec)) - set(list(model.heat)))
    elec_costs = (sum(
        (model.capacity[tec] - existing_capacity[tec]) * annuities[tec] * nb_years for tec in
        all_but_heater)
     + sum((model.energy_capacity[storage_tecs] - existing_energy_capacity[storage_tecs]) *
                storage_annuities[
                    storage_tecs] * nb_years for storage_tecs in model.str)
     + sum(model.capacity[tec] * fOM[tec] * nb_years for tec in model.tec)
     + sum(sum(model.gene[tec, h] * vOM[tec] for h in model.h) for tec in model.tec)) / 1000
    return elec_costs


def extract_renovation_rates(model, nb_linearize):
    """Extractions renovation decisions per archetype (in % of initial heating demand)"""
    list_renovation_options = model.renovation  # includes linearized segments
    list_archetype = model.archetype  # includes building archetypes
    renovation_rates = pd.Series(index=list_archetype, dtype=float)
    for a in list_archetype:
        renov_rate = 0
        for l in range(nb_linearize):
            tec = f"{a}_{l}"
            renov_rate += value(model.renovation_rate[tec])
        renovation_rates.loc[a] = renov_rate
    return renovation_rates


def extract_renovation_rates_detailed(model):
    """Extractions renovation decisions per segment of archetype (in % of initial heating demand)"""
    list_renovation_options = model.renovation  # includes linearized segments
    renovation_rates_detailed = pd.Series(index=list_renovation_options, dtype=float)
    for r in list_renovation_options:
        renovation_rates_detailed.loc[r] = value(model.renovation_rate[r])
    return renovation_rates_detailed


def extract_hourly_generation(model, elec_demand, CH4_demand, H2_demand, conversion_efficiency, heat_demand=None, hourly_heat_elec=None, hourly_heat_gas=None):
    """Extracts hourly defined data, including demand, generation and storage
    Returns a dataframe with hourly generation for each hour."""
    list_tec = list(model.tec)
    list_storage_in = [e + "_in" for e in model.str]
    list_storage_charge = [e + "_charge" for e in model.str]
    list_columns = ["hour", "demand"] + list_tec + list_storage_in + list_storage_charge
    hourly_generation = pd.DataFrame(columns=list_columns)
    hourly_generation.loc[:, "hour"] = list(model.h)
    hourly_generation.loc[:, "elec_demand"] = elec_demand
    hourly_generation.loc[:, "CH4_demand"] = CH4_demand
    hourly_generation.loc[:, "H2_demand"] = H2_demand
    if heat_demand is not None:
        hourly_generation.loc[:, "heat_demand"] = heat_demand
    if hourly_heat_elec is not None:
        hourly_generation.loc[:, "heat_elec"] = hourly_heat_elec
    if hourly_heat_gas is not None:
        hourly_generation.loc[:, "heat_gas"] = hourly_heat_gas
    for tec in list_tec:
        hourly_generation[tec] = value(model.gene[tec, :])  # GWh
    for str, str_in in zip(list(model.str), list_storage_in):
        hourly_generation[str_in] = value(model.storage[str, :])  # GWh
    for str, str_charge in zip(list(model.str), list_storage_charge):
        hourly_generation[str_charge] = value(model.stored[str, :])  # GWh
    # We add technologies which include a conversion parameter, to express their hourly generation in GWh-e
    hourly_generation["electrolysis_elec"] = value(model.gene["electrolysis", :]) / conversion_efficiency["electrolysis"]
    hourly_generation["methanation_elec"] = value(model.gene["methanation", :]) / conversion_efficiency["methanation"]
    return hourly_generation  # GWh


def get_carbon_content(hourly_generation, conversion, carbon_content, climate=2006, nb_years=1):
    """Estimates the carbon content of gas and of electric heating, based on methodology by ADEME and RTE (méthode moyenne horaire).
    Returns the result in gCO2/kWh"""
    assert 'heat_elec' in hourly_generation.columns, "Column heat_elec should be included to estimate carbon content of electric heating"

    # Estimate carbon content of gas
    gas_carbon_content = (hourly_generation['natural_gas'].sum() * carbon_content['natural_gas']) / (hourly_generation['natural_gas'] +
                                                                                     hourly_generation['methanization'] + hourly_generation['pyrogazification']).sum()

    # Estimate carbon content of district heating (based on previously estimated carbon content of gas)
    district_heating_content = (hourly_generation['central_gas_boiler'].sum() * gas_carbon_content) / (hourly_generation['central_gas_boiler'] +
                                                                                                       hourly_generation['geothermal'] + hourly_generation['central_wood_boiler'] + hourly_generation['uiom']).sum()

    elec_gene = ["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "nuclear", "phs", "battery1",
                 "battery4", "ocgt", "ccgt", 'coal', "h2_ccgt"]
    tot_cols = elec_gene + ['heat_elec']
    # hourly_generation = hourly_generation[tot_cols]

    hourly_generation['carbon_content'] = hourly_generation.apply(lambda row: (row['ocgt'] / conversion['ocgt'] * gas_carbon_content +
                                                                               row['ccgt'] /  conversion['ccgt'] * gas_carbon_content +
                                                                               row['coal'] * carbon_content['coal']) / sum(
            row[tec] for tec in elec_gene), axis=1)
    tot_heat_elec = hourly_generation.heat_elec.sum()
    hourly_generation['carbon_content_heat'] = hourly_generation.apply(lambda row: row['carbon_content'] * row['heat_elec'] / tot_heat_elec, axis=1)
    heat_elec_carbon_content = hourly_generation.carbon_content_heat.sum()

    if nb_years <= 1:
        hourly_generation["date"] = hourly_generation.apply(lambda row: datetime.datetime(climate, 1, 1, 0) + datetime.timedelta(hours=row["hour"]),
            axis=1)  # we just use this feature to group by day, climate does not really matter here
        hourly_generation["date_only"] = hourly_generation["date"].apply(lambda x: x.date())

        hourly_generation_day = hourly_generation.copy().groupby("date_only")[
            elec_gene + ['heat_elec']].sum().reset_index()
        hourly_generation_day['carbon_content'] = hourly_generation_day.apply(
            lambda row: (row['ocgt'] / conversion['ocgt'] * gas_carbon_content + row['ccgt'] / conversion['ccgt'] * gas_carbon_content + row['coal'] * carbon_content['coal']) / sum(row[tec] for tec in elec_gene), axis=1)

        hourly_generation_day['carbon_content_heat'] = hourly_generation_day.apply(lambda row: row['carbon_content'] * row['heat_elec'] / tot_heat_elec, axis=1)

        heat_elec_carbon_content_day = hourly_generation_day.carbon_content_heat.sum()
    else:
        heat_elec_carbon_content_day = 0

    return gas_carbon_content * 1e3, district_heating_content * 1e3, heat_elec_carbon_content * 1e3, heat_elec_carbon_content_day * 1e3


def extract_peak_load(hourly_generation:pd.DataFrame, conversion_efficiency, input_years):
    """Returns the value of peak load for electricity in GW. Includes electricity demand, as well as demand for electrolysis.
    ATTENTION: cette fonction marche uniquement pour le couplage avec ResIRF, pas pour le social planner. Dans ce cas,
     il faudrait ajouter également à la valeur de la pointe la demande pour les PAC et radiateurs."""
    if "heat_elec" in hourly_generation.columns and "heat_gas" in hourly_generation.columns:
        peak_load = hourly_generation.copy()[["elec_demand", "electrolysis", "methanation", "heat_elec", "heat_gas"]]
    else:
        peak_load = hourly_generation.copy()[["elec_demand", "electrolysis", "methanation"]]

    peak_load["peak_electricity_load"] = peak_load["elec_demand"] + peak_load["electrolysis"] / conversion_efficiency[
        "electrolysis"] + peak_load["methanation"] / conversion_efficiency["methanation"]
    ind = peak_load.index[peak_load["peak_electricity_load"] == peak_load["peak_electricity_load"].max()]
    peak_load_info = peak_load.loc[ind].reset_index().rename(columns={"index": "hour"})
    peak_load_info["nb_year"] = peak_load_info.apply(lambda row: int(row["hour"] // 8760), axis=1)
    peak_load_info["date"] = peak_load_info.apply(lambda row: datetime.datetime(input_years[int(row["nb_year"])], 1, 1, 0) + datetime.timedelta(hours=row["hour"] - 8760*row["nb_year"]),
                            axis=1)  # TODO: a changer si on modifie le climat

    return peak_load_info  # GW


def extract_peak_heat_load(hourly_generation:pd.DataFrame, input_years):
    """Returns the value of peak load for electricity in GW. Includes electricity demand, as well as demand for electrolysis.
    ATTENTION: cette fonction marche uniquement pour le couplage avec ResIRF, pas pour le social planner. Dans ce cas,
     il faudrait ajouter également à la valeur de la pointe la demande pour les PAC et radiateurs."""
    peak_heat_load = hourly_generation.copy()[["elec_demand", "heat_elec", "heat_gas"]]
    ind = peak_heat_load.index[peak_heat_load["heat_elec"] == peak_heat_load["heat_elec"].max()]
    peak_heat_load_info = peak_heat_load.loc[ind].reset_index().rename(columns={"index": "hour"})
    peak_heat_load_info["nb_year"] = peak_heat_load_info.apply(lambda row: int(row["hour"] // 8760), axis=1)
    peak_heat_load_info["date"] = peak_heat_load_info.apply(lambda row: datetime.datetime(input_years[int(row["nb_year"])], 1, 1, 0) + datetime.timedelta(hours=row["hour"] - 8760*row["nb_year"]),
                            axis=1)  # TODO: a changer si on modifie le climat

    return peak_heat_load_info  # GW


def extract_spot_price(model, nb_hours):
    """Extracts spot price"""
    spot_price = pd.DataFrame({"hour": range(nb_hours),
                               "elec_spot_price": [- 1e6 * model.dual[model.electricity_adequacy_constraint[h]] for h in
                                              model.h],
                               "CH4_spot_price": [1e6 * model.dual[model.methane_balance_constraint[h]] for h in
                                                   model.h]
                               })
    return spot_price


def extract_carbon_value(model, carbon_constraint, scc):
    """Extracts the social value of carbon in the considered model."""
    if carbon_constraint:
        # TODO: here we only consider the carbon value for one of the given years !! to modify in future
        carbon_value = -1e3 * model.dual[model.carbon_budget_constraint[0]]  # €/tCO2
    else:
        carbon_value = scc
    return carbon_value


def extract_supply_elec(model, nb_years):
    """Extracts yearly electricity supply per technology in TWh"""
    list_tec = list(model.elec_gene)
    electricity_supply = pd.Series(index=list_tec, dtype=float)

    for tec in list_tec:
        electricity_supply[tec] = sum(value(model.gene[tec, hour]) for hour in model.h) / 1000 / nb_years  # TWh
    return electricity_supply


def extract_primary_gene(model, nb_years):
    """Extracts yearly primary energy generation per source of energy in TWh"""
    list_tec = list(model.primary_gene)
    primary_generation = pd.Series(index=list_tec, dtype=float)

    for tec in list_tec:
        primary_generation[tec] = sum(value(model.gene[tec, hour]) for hour in model.h) / 1000 / nb_years  # TWh
    return primary_generation


def extract_CH4_to_power(model, conversion_efficiency, nb_years):
    """Extracts CH4 generation necessary to produce electricity"""
    list_tec = list(model.from_CH4_to_elec)
    gas_to_power_generation = pd.Series(index=list_tec, dtype=float)

    for tec in list_tec:
        gas_to_power_generation[tec] = sum(value(model.gene[tec, hour]) / conversion_efficiency[tec] for hour in model.h) / 1000 / nb_years  # TWh
    return gas_to_power_generation


def extract_H2_to_power(model, conversion_efficiency, nb_years):
    """Extracts CH4 generation necessary to produce electricity"""
    list_tec = list(model.from_H2_to_elec)
    H2_to_power_generation = pd.Series(index=list_tec, dtype=float)

    for tec in list_tec:
        H2_to_power_generation[tec] = sum(value(model.gene[tec, hour]) / conversion_efficiency[tec] for hour in model.h) / 1000 / nb_years  # TWh
    return H2_to_power_generation


def extract_power_to_CH4(model, conversion_efficiency, nb_years):
    """Extracts electricity generation necessary to produce CH4"""
    list_tec = list(model.from_elec_to_CH4)
    power_to_CH4_generation = pd.Series(index=list_tec, dtype=float)

    for tec in list_tec:
        power_to_CH4_generation[tec] = sum(value(model.gene[tec, hour]) / conversion_efficiency[tec] for hour in model.h) / 1000 / nb_years  # TWh
    return power_to_CH4_generation


def extract_power_to_H2(model, conversion_efficiency, nb_years):
    """Extracts electricity generation necessary to produce H2"""
    list_tec = list(model.from_elec_to_H2)
    power_to_H2_generation = pd.Series(index=list_tec, dtype=float)

    for tec in list_tec:
        power_to_H2_generation[tec] = sum(value(model.gene[tec, hour]) / conversion_efficiency[tec] for hour in model.h) / 1000 / nb_years  # TWh
    return power_to_H2_generation


def extract_heat_gene(model, conversion_efficiency, hp_cop, nb_years):
    """Extracts yearly heat generation per technology in TWh"""
    list_tec = list(model.heat)
    heat_generation = pd.Series(index=list_tec, dtype=float)  # besoin de chaleur (donc en TWh_th)
    heat_consumption = pd.Series(index=list_tec, dtype=float)  # consommation pour satisfaire le besoin de chaleur (donc en TW-th ou TW-e)

    for tec in list_tec:
        if tec == 'wood_boiler' or tec == 'fuel_boiler':
            heat_generation[tec] = sum(value(model.gene[tec, hour])*conversion_efficiency[tec] for hour in model.h) / 1000 / nb_years  # TWh-th
            heat_consumption[tec] = sum(
                value(model.gene[tec, hour]) for hour in model.h) / 1000 / nb_years  # TWh
        elif tec == "heat_pump":
            heat_generation[tec] = sum(value(model.gene[tec, hour]) for hour in model.h) / 1000 / nb_years  # TWh-th
            heat_consumption[tec] = sum(
                value(model.gene[tec, hour]) / hp_cop[hour] for hour in model.h) / 1000 / nb_years  # TWh-e
        else:  # resistive or gas boiler
            heat_generation[tec] = sum(value(model.gene[tec, hour]) for hour in model.h) / 1000 / nb_years  # TWh-th
            heat_consumption[tec] = sum(value(model.gene[tec, hour])/conversion_efficiency[tec] for hour in model.h) / 1000 / nb_years  # TWh-e ou TWh-g
    return heat_generation, heat_consumption


def extract_use_elec(model, nb_years, miscellaneous):
    """Extracts yearly electricity use per technology in TWh"""
    list_tec = list(model.use_elec)
    electricity_use = pd.Series(index=list_tec, dtype=float)

    for tec in list_tec:
        if tec == 'electrolysis':  # for electrolysis, we need to use the efficiency factor to obtain TWhe
            electricity_use[tec] = sum(value(model.gene[tec, hour]) for hour in model.h) / 1000 / nb_years / \
                                   miscellaneous['eta_electrolysis']
        else:
            electricity_use[tec] = sum(value(model.storage[tec, hour]) for hour in model.h) / 1000 / nb_years
    return electricity_use


def extract_annualized_costs_investment_new_capa(capacities, energy_capacities, existing_capacities, existing_energy_capacities,
                                                 annuities, storage_annuities, fOM):
    """
    Returns the annualized costs coming from newly invested capacities and energy capacities. This includes annualized CAPEX + fOM.
    Unit: 1e6€/yr
    :param model: pyomo model
    :param existing_capacities: pd.Series
    :return:
    """
    new_capacity = capacities - existing_capacities  # pd.Series
    costs_new_capacity = pd.concat([new_capacity, annuities, fOM], axis=1, ignore_index=True).rename(columns={0: "new_capacity", 1: "annuities", 2: "fOM"})
    costs_new_capacity["annualized_costs"] = costs_new_capacity["new_capacity"] * (costs_new_capacity["annuities"] + costs_new_capacity["fOM"])  # includes both annuity and fOM ! not to be counted twice in the LCOE

    new_storage_capacity = energy_capacities - existing_energy_capacities
    costs_new_energy_capacity = pd.concat([new_storage_capacity, storage_annuities], axis=1, ignore_index=True).rename(columns={0: "new_capacity", 1: "storage_annuities"})
    costs_new_energy_capacity["annualized_costs"] = costs_new_energy_capacity["new_capacity"] * costs_new_energy_capacity["storage_annuities"]
    return costs_new_capacity[["annualized_costs"]], costs_new_energy_capacity[["annualized_costs"]]


def extract_annualized_costs_investment_new_capa_nofOM(capacities, energy_capacities, existing_capacities, existing_energy_capacities,
                                                 annuities, storage_annuities):
    """
    Returns the annualized investment coming from newly invested capacities and energy capacities, without fOM. Unit: 1e6€/yr
    :param model: pyomo model
    :param existing_capacities: pd.Series
    :return:
    """
    new_capacity = capacities - existing_capacities  # pd.Series
    costs_new_capacity = pd.concat([new_capacity, annuities], axis=1, ignore_index=True).rename(columns={0: "new_capacity", 1: "annuities"})
    costs_new_capacity = costs_new_capacity.dropna()
    costs_new_capacity["annualized_costs"] = costs_new_capacity["new_capacity"] * costs_new_capacity["annuities"]  # includes both annuity and fOM ! not to be counted twice in the LCOE

    new_storage_capacity = energy_capacities - existing_energy_capacities
    costs_new_energy_capacity = pd.concat([new_storage_capacity, storage_annuities], axis=1, ignore_index=True).rename(columns={0: "new_capacity", 1: "storage_annuities"})
    costs_new_energy_capacity = costs_new_energy_capacity.dropna()
    costs_new_energy_capacity["annualized_costs"] = costs_new_energy_capacity["new_capacity"] * costs_new_energy_capacity["storage_annuities"]
    return costs_new_capacity[["annualized_costs"]], costs_new_energy_capacity[["annualized_costs"]]


def extract_functionment_cost(model, capacities, fOM, vOM, generation, oil_consumption, wood_consumption, anticipated_scc, actual_scc, carbon_content,
                              wood_neutral=False, carbon_constraint=True, nb_years=1):
    """Returns functionment cost, including fOM and vOM. vOM for gas and oil include the SCC. Unit: 1e6€/yr
    This function has to update vOM for natural gas and fossil fuel based on the actual scc, and no longer based on the
    anticipated_scc which was used to find optimal investment and dispatch.
    IMPORTANT REMARK: we divide generation by number of total years to get the average yearly generation
    :param anticipated_scc: int
        Anticipated social cost of carbon used to estimate optimal power mix.
    :param actual_scc: int
        Actual social cost of carbon, used to calculate functionment cost.
    """
    # New version
    if not carbon_constraint:  # optimization under social cost of carbon
        vOM_no_scc = vOM.copy()  # we remove the SCC in this vOM
        vOM_no_scc.loc["natural_gas"] = update_vom_costs_scc(vOM_no_scc.loc["natural_gas"], scc=(-anticipated_scc), emission_rate=carbon_content['natural_gas'])  # €/kWh
        vOM_no_scc["oil"] = update_vom_costs_scc(vOM_no_scc["oil"], scc=(- anticipated_scc), emission_rate=carbon_content['oil'])
        if not wood_neutral:
            vOM_no_scc["wood"] = update_vom_costs_scc(vOM_no_scc["wood"], scc=(- anticipated_scc), emission_rate=carbon_content['wood'])

        vOM_SCC_only = (vOM - vOM_no_scc).copy()  # variable cost only due to actual scc, not anticipated scc
        vOM_SCC_only.loc["natural_gas"] = update_vom_costs_scc(vOM_SCC_only.loc["natural_gas"], scc=(actual_scc - anticipated_scc), emission_rate=carbon_content['natural_gas'])  # €/kWh
        vOM_SCC_only["oil"] = update_vom_costs_scc(vOM_SCC_only["oil"], scc=(actual_scc - anticipated_scc), emission_rate=carbon_content['oil'])
        if not wood_neutral:
            vOM_SCC_only["wood"] = update_vom_costs_scc(vOM_SCC_only["wood"], scc=(actual_scc - anticipated_scc), emission_rate=carbon_content['wood'])

        system_fOM_vOM = pd.concat([capacities, fOM, vOM_no_scc, vOM_SCC_only, generation/nb_years], axis=1, ignore_index=True).rename(
            columns={0: "capacity", 1: "fOM", 2: "vOM_no_scc", 3: "vOM_SCC_only", 4: "generation"})
        system_fOM_vOM = system_fOM_vOM.dropna()
        system_fOM_vOM["functionment_cost_noSCC"] = system_fOM_vOM["capacity"] * system_fOM_vOM["fOM"] + system_fOM_vOM["generation"] * system_fOM_vOM["vOM_no_scc"]
        system_fOM_vOM["functionment_cost_SCC"] = system_fOM_vOM["generation"] * system_fOM_vOM["vOM_SCC_only"]
        system_fOM_vOM_df = system_fOM_vOM[["functionment_cost_noSCC"]]

        total_wood_consumption = wood_consumption + sum(value(model.gene["central_wood_boiler", h]) for h in model.h)
        oil_functionment_cost_no_scc, wood_functionment_cost_no_scc = oil_consumption * vOM_no_scc["oil"], total_wood_consumption * vOM_no_scc["wood"]
        carbon_cost = system_fOM_vOM["functionment_cost_SCC"].sum() + oil_consumption * vOM_SCC_only["oil"] + total_wood_consumption * vOM_SCC_only["wood"]

        system_fOM_vOM_df = pd.concat([system_fOM_vOM_df, pd.DataFrame(index=["oil"], data={'functionment_cost_noSCC': [oil_functionment_cost_no_scc]})], axis=0)
        system_fOM_vOM_df = pd.concat([system_fOM_vOM_df, pd.DataFrame(index=["wood"], data={'functionment_cost_noSCC': [wood_functionment_cost_no_scc]})], axis=0)
        system_fOM_vOM_df = pd.concat([system_fOM_vOM_df, pd.DataFrame(index=["carbon_cost"], data={'functionment_cost_noSCC': [carbon_cost]})], axis=0)
        system_fOM_vOM_df = system_fOM_vOM_df.rename(columns={'functionment_cost_noSCC': 'functionment_cost'})
    else:
        new_vOM = vOM.copy()
        system_fOM_vOM = pd.concat([capacities, fOM, new_vOM, generation/nb_years], axis=1, ignore_index=True).rename(columns={0: "capacity", 1: "fOM", 2: "vOM", 3: "generation"})
        system_fOM_vOM = system_fOM_vOM.dropna()
        system_fOM_vOM["functionment_cost"] = system_fOM_vOM["capacity"] * system_fOM_vOM["fOM"] + system_fOM_vOM["generation"] * system_fOM_vOM["vOM"]
        system_fOM_vOM_df = system_fOM_vOM[["functionment_cost"]]
        total_wood_consumption = wood_consumption + sum(value(model.gene["central_wood_boiler", h]) for h in model.h)
        oil_functionment_cost, wood_functionment_cost = oil_consumption * new_vOM["oil"], total_wood_consumption * new_vOM["wood"]
        system_fOM_vOM_df = pd.concat([system_fOM_vOM_df, pd.DataFrame(index=["oil"], data={'functionment_cost': [oil_functionment_cost]})], axis=0)
        system_fOM_vOM_df = pd.concat([system_fOM_vOM_df, pd.DataFrame(index=["wood"], data={'functionment_cost': [wood_functionment_cost]})], axis=0)

    # # OLD VERSION
    # new_vOM = vOM.copy()
    # new_vOM.loc["natural_gas"] = update_ngas_cost(new_vOM.loc["natural_gas"], scc=(actual_scc - anticipated_scc), emission_rate=0.2295)  # €/kWh
    # new_vOM["fuel_boiler"] = update_ngas_cost(new_vOM["fuel_boiler"], scc=(actual_scc - anticipated_scc), emission_rate=0.324)
    #
    # system_fOM_vOM = pd.concat([capacities, fOM, new_vOM, generation], axis=1, ignore_index=True).rename(columns={0: "capacity", 1: "fOM", 2: "vOM", 3: "generation"})
    # system_fOM_vOM = system_fOM_vOM.dropna()
    # system_fOM_vOM["functionment_cost"] = system_fOM_vOM["capacity"] * system_fOM_vOM["fOM"] + system_fOM_vOM["generation"] * system_fOM_vOM["vOM"]
    # system_fOM_vOM_df = system_fOM_vOM[["functionment_cost"]]
    # oil_functionment_cost, wood_functionment_cost = oil_consumption * new_vOM["fuel_boiler"], wood_consumption * new_vOM["wood_boiler"]
    # system_fOM_vOM_df = pd.concat([system_fOM_vOM_df, pd.DataFrame(index=["oil_boiler"], data={'functionment_cost': [oil_functionment_cost]})], axis=0)
    # system_fOM_vOM_df = pd.concat([system_fOM_vOM_df, pd.DataFrame(index=["wood_boiler"], data={'functionment_cost': [wood_functionment_cost]})], axis=0)
    return system_fOM_vOM_df


def annualized_costs_investment_historical(existing_capa_historical_y, annuity_fOM_historical,
                                           existing_energy_capacity_historical_y, storage_annuity_historical):
    """Returns the annualized costs coming from historical capacities and energy capacities. This includes annualized CAPEX + fOM. 1e6 €"""
    costs_capacity_historical = pd.concat([existing_capa_historical_y, annuity_fOM_historical], axis=1, ignore_index=True)  # we only include nonzero historical capacities
    costs_capacity_historical = costs_capacity_historical.rename(columns={0: 'capacity_historical', 1: 'annuity_fOM'}).fillna(0)
    costs_capacity_historical["annualized_costs"] = costs_capacity_historical["capacity_historical"] * costs_capacity_historical["annuity_fOM"]

    costs_energy_capacity_historical = pd.concat([existing_energy_capacity_historical_y, storage_annuity_historical], axis=1, ignore_index=True)  # we only include nonzero historical capacities
    costs_energy_capacity_historical = costs_energy_capacity_historical.rename(columns={0: 'energy_capacity_historical', 1: 'storage_annuity'}).fillna(0)
    costs_energy_capacity_historical["annualized_costs"] = costs_energy_capacity_historical["energy_capacity_historical"] * costs_energy_capacity_historical["storage_annuity"]
    return costs_capacity_historical[["annualized_costs"]], costs_energy_capacity_historical[["annualized_costs"]]


def annualized_costs_investment_historical_nofOM(existing_capa_historical_y, capex_annuity_historical,
                                           existing_energy_capacity_historical_y, storage_annuity_historical):
    """Returns the annualized costs coming from historical capacities and energy capacities. This includes only annualized CAPEX, no fOM."""
    costs_capacity_historical = pd.concat([existing_capa_historical_y, capex_annuity_historical], axis=1, ignore_index=True)  # we only include nonzero historical capacities
    costs_capacity_historical = costs_capacity_historical.rename(columns={0: 'capacity_historical', 1: 'capex_annuity'}).fillna(0)
    costs_capacity_historical["annualized_costs"] = costs_capacity_historical["capacity_historical"] * costs_capacity_historical["capex_annuity"]

    return costs_capacity_historical[["annualized_costs"]]


def process_annualized_costs_per_vector(annualized_costs_capacity, annualized_costs_energy_capacity):
    """Calculates annualized costs related to investment for the different energy vectors (namely, electricity, methane and hydrogen)"""
    elec_balance = ["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "nuclear", "phs",
     "battery1", "battery4", "ocgt", "ccgt", "h2_ccgt"]
    elec_str = ["phs", "battery1", "battery4"]

    CH4_balance = ["methanization", "pyrogazification", "natural_gas", "methanation", "methane"]
    CH4_str = ["methane"]
    CH4_balance_historic = ["natural_gas", "methane"]
    CH4_balance_biogas = ["methanization", "pyrogazification", "methanation"]

    H2_balance = ["electrolysis", "hydrogen"]
    H2_str = ["hydrogen"]

    annualized_costs_elec = annualized_costs_capacity[elec_balance].sum() + annualized_costs_energy_capacity[elec_str].sum()  # includes annuity, fOM and storage annuity
    annualized_costs_CH4 = annualized_costs_capacity[CH4_balance].sum() + annualized_costs_energy_capacity[CH4_str].sum()
    annualized_costs_CH4_naturalgas = annualized_costs_capacity[CH4_balance_historic].sum() + annualized_costs_energy_capacity[CH4_str].sum()
    annualized_costs_CH4_biogas = annualized_costs_capacity[CH4_balance_biogas].sum()
    annualized_costs_H2 = annualized_costs_capacity[H2_balance].sum() + annualized_costs_energy_capacity[H2_str].sum()
    return annualized_costs_elec, annualized_costs_CH4, annualized_costs_CH4_naturalgas, annualized_costs_CH4_biogas, annualized_costs_H2


def calculate_LCOE_gene_tec(list_tec, model, annuities, fOM, vOM, nb_years, gene_per_tec):
    """Calculates LCOE per generating technology with fixed vOM"""
    lcoe = {}
    for tec in list_tec:
        gene = gene_per_tec[tec]  # TWh
        lcoe_tec = (
                    (value(model.capacity[tec])) * (annuities[tec] + fOM[tec]) * nb_years + gene * 1000 * vOM[
                    tec]) / gene  # € / MWh
        lcoe[tec] = lcoe_tec
    return lcoe


def calculate_LCOE_conv_tec(list_tec, model, annuities, fOM, conversion_efficiency, spot_price, nb_years, gene_per_tec):
    """Calculates LCOE per conversion technology, where vOM is the dual of a constraint."""
    lcoe = {}
    for tec in list_tec:
        gene = gene_per_tec[tec]  # TWh
        vOM = sum(spot_price[hour] * (
                value(model.gene[tec, hour]) / conversion_efficiency[tec]) for hour in
                  model.h) / 1e3  # 1e6 €
        lcoe_tec = (
                           value(model.capacity[tec]) * (annuities[tec] + fOM[tec]) * nb_years + vOM) / gene  # € / MWh
        lcoe[tec] = lcoe_tec
    return lcoe


def write_output(results, folder):
    """
    Saves the outputs of the model. No longer use.
    :param results: dict
        Contains the different dataframes outputed by the model
    :param folder: str
        Folder where to save the output
    :return:
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)
    variables_to_save = ["summary", "hourly_generation", "capacities", "energy_capacity", "supply_elec", "use_elec"]
    for variable in variables_to_save:
        path_to_save = os.path.join(folder, f"{variable}.csv")
        df_to_save = results[variable]
        df_to_save.to_csv(path_to_save)


def read_output(folder):
    """No longer use."""
    variables_to_read = ["summary", "hourly_generation", "capacities", "energy_capacity", "supply_elec", "use_elec"]
    o = dict()
    for variable in variables_to_read:
        path_to_read = os.path.join(folder, f"{variable}.csv")
        df = pd.read_csv(path_to_read, index_col=0)
        df = df.rename(columns={'0': variable})
        o[variable] = df
    return o


def format_ax_old(ax, y_label=None, title=None, y_max=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if title is not None:
        ax.set_title(title)

    if y_max is not None:
        ax.set_ylim(ymax=0)

    return ax


def plot_capacities_old(df, y_max=None):
    fig, ax = plt.subplots(1, 1)
    df.plot.bar(ax=ax)
    ax = format_ax_old(ax, y_label="Capacity (GW)", title="Capacities", y_max=y_max)
    plt.show()


def plot_generation(df):
    fig, ax = plt.subplots(1, 1)
    df.plot.pie(ax=ax)


##### CONFIG FUNCTIONS ######

def create_default_options(config_coupling):
    """Updates default parameters for the optimization function. Those parameters correspond to parameters required for optimization,
    and parameters required for the coupling setting. Other specific parameters stay included in the config_coupling dictionary.
     Note that some of those parameters will always be updated,
    since they are required in the config_coupling file."""
    default_config = {
        'acquisition_jitter' : 0.01,  # optional in config_coupling
        'grid_initialize' : False,  # optional in config_coupling
        'normalize_Y' : True,  # optional in config_coupling
        'anticipated_demand_t10' : False,  # optional in config_coupling
        'anticipated_scc' : False,  # optional in config_coupling
        'price_feedback' : False,
        'aggregated_potential' : False,
        'cofp' : False,
        'electricity_constant' : False,
        'optim_eoles': True
    }

    for key in default_config.keys():
        if key in config_coupling.keys():
            default_config[key] = config_coupling[key]
        if key in config_coupling['eoles'].keys():  # for specific case of aggregated potential
            default_config[key] = config_coupling['eoles'][key]
    return default_config

@dataclasses.dataclass
class OptimizationParam:
    acquisition_jitter: float = 0.01
    grid_initialize: bool = False
    normalize_Y : bool = True


@dataclasses.dataclass
class CouplingParam:
    """
        aggregated_potential: If True, then this means that the maximum capacities for each time step is an aggregated potential (thus allowing
        to catch up for capacities not built initially because of myopic optimization)"""
    anticipated_demand_t10: bool = False
    anticipated_scc: bool = False
    price_feedback: bool = False
    aggregated_potential: bool = False
    cofp: bool = False
    electricity_constant: bool = False
    optim_eoles: bool = True


def create_optimization_param(default_config) -> OptimizationParam:
    return OptimizationParam(acquisition_jitter=default_config['acquisition_jitter'],
                             grid_initialize=default_config['grid_initialize'],
                             normalize_Y=default_config['normalize_Y'])


def create_coupling_param(default_config) -> CouplingParam:
    return CouplingParam(anticipated_demand_t10=default_config['anticipated_demand_t10'],
                         anticipated_scc=default_config['anticipated_scc'],
                         price_feedback=default_config['price_feedback'],
                         aggregated_potential=default_config['aggregated_potential'],
                         cofp=default_config['cofp'],
                         electricity_constant=default_config['electricity_constant'])


def modif_config_coupling(design, config_coupling, max_iter_single_iteration=50, cap_MWh=400, cap_tCO2=1000,
                          subsidies_heater=None, subsidies_insulation=None):
    """Creates a new configuration file based on an initial configuration file, for a given subsidy design."""
    # TODO: attention, il y a eu des changements côté Res-IRF: modifier cette fonction !
    config_coupling_update = deepcopy(config_coupling)
    if subsidies_heater is not None:
        config_coupling_update['subsidies_heater'] = subsidies_heater
    if subsidies_insulation is not None:
        config_coupling_update['subsidies_insulation'] = subsidies_insulation
    if design == "uniform":
        config_coupling_update["subsidy"] = {
            'proportional_uniform': None,
            'heater': {
                'policy': 'subsidy_ad_valorem',
                'proportional': None,
                'cap': None
            },
            'insulation': {
                'policy': 'subsidy_ad_valorem',
                'rational_behavior': False,
                'target': None,
                'proportional': None,
                'cap': None
            }
        }
    elif design == "DR":
        config_coupling_update["subsidy"] = {
            'proportional_uniform': None,
            'heater': {
                'policy': 'subsidy_ad_valorem',
                'proportional': None,
                'cap': None
            },
            'insulation': {
                'policy': 'subsidy_ad_valorem',
                'rational_behavior': False,
                'target': "deep_renovation",
                'proportional': None,
                'cap': None
            }
        }
    elif design == "DR_FGE":
        config_coupling_update["subsidy"] = {
            'proportional_uniform': None,
            'heater': {
                'policy': 'subsidy_ad_valorem',
                'proportional': None,
                'cap': None
            },
            'insulation': {
                'policy': 'subsidy_ad_valorem',
                'rational_behavior': False,
                'target': "deep_renovation_fge",
                'proportional': None,
                'cap': None
            }
        }
    elif design == "GR_low_income":  # TODO: nom a changer
        config_coupling_update["subsidy"] = {
            'proportional_uniform': None,
            'heater': {
                'policy': 'subsidy_ad_valorem',
                'proportional': None,
                'cap': None
            },
            'insulation': {
                'policy': 'subsidy_ad_valorem',
                'rational_behavior': False,
                'target': "global_renovation_low_income",
                'proportional': None,
                'cap': None
            }
        }
    elif design == "GR_fg":
        config_coupling_update["subsidy"] = {
            'proportional_uniform': None,
            'heater': {
                'policy': 'subsidy_ad_valorem',
                'proportional': None,
                'cap': None
            },
            'insulation': {
                'policy': 'subsidy_ad_valorem',
                'rational_behavior': False,
                'target': "global_renovation_fg",
                'proportional': None,
                'cap': None
            }
        }
    elif design == "GR_fge":
        config_coupling_update["subsidy"] = {
            'proportional_uniform': None,
            'heater': {
                'policy': 'subsidy_ad_valorem',
                'proportional': None,
                'cap': None
            },
            'insulation': {
                'policy': 'subsidy_ad_valorem',
                'rational_behavior': False,
                'target': "global_renovation_fge",
                'proportional': None,
                'cap': None
            }
        }
    elif design == "out_worst":
        config_coupling_update["subsidy"] = {
            'proportional_uniform': None,
            'heater': {
                'policy': 'subsidy_ad_valorem',
                'proportional': None,
                'cap': None
            },
            'insulation': {
                'policy': 'subsidy_ad_valorem',
                'rational_behavior': False,
                'target': "out_worst",
                'proportional': None,
                'cap': None
            }
        }
    elif design == "centralized_insulation":
        config_coupling_update["subsidy"] = {
            'proportional_uniform': None,
            'heater': {
                'policy': 'subsidy_ad_valorem',
                'proportional': None,
                'cap': None
            },
            'insulation': {
                'policy': 'subsidy_ad_valorem',
                'rational_behavior': True,
                'target': None,
                'proportional': None,
                'cap': None
            }
        }
    elif design == "centralized_insulation_social":
        config_coupling_update["subsidy"] = {
            'proportional_uniform': None,
            'heater': {
                'policy': 'subsidy_ad_valorem',
                'proportional': None,
                'cap': None
            },
            'insulation': {
                'policy': 'subsidy_ad_valorem',
                'rational_behavior': True,
                'social': True,
                'target': None,
                'proportional': None,
                'cap': None
            }
        }
    elif design == "centralized_insulation_heater":
        config_coupling_update["subsidy"] = {
            'proportional_uniform': None,
            'heater': {
                'policy': 'subsidy_ad_valorem',
                'proportional': None,
                'cap': None
            },
            'insulation': {
                'policy': 'subsidy_ad_valorem',
                'rational_behavior': True,
                'target': None,
                'proportional': None,
                'cap': None
            }
        }
    elif design == "no_subsidy_heater":
        config_coupling_update["fix_sub_heater"] = True
        config_coupling_update["max_iter"] = max_iter_single_iteration
        config_coupling_update["subsidy"] = {
            'proportional_uniform': None,
            'heater': {
                'policy': 'subsidy_ad_valorem',
                'proportional': None,
                'cap': None
            },
            'insulation': {
                'policy': 'subsidy_ad_valorem',
                'rational_behavior': False,
                'target': None,
                'proportional': None,
                'cap': None
            }
        }
    elif design == "no_subsidy_insulation":
        config_coupling_update["fix_sub_insulation"] = True
        config_coupling_update["max_iter"] = max_iter_single_iteration
        config_coupling_update["subsidy"] = {
            'proportional_uniform': None,
            'heater': {
                'policy': 'subsidy_ad_valorem',
                'proportional': None,
                'cap': None
            },
            'insulation': {
                'policy': 'subsidy_ad_valorem',
                'rational_behavior': False,
                'target': None,
                'proportional': None,
                'cap': None
            }
        }
    elif design == "no_subsidy_heater_centralized":
        config_coupling_update["fix_sub_heater"] = True
        config_coupling_update["max_iter"] = max_iter_single_iteration
        config_coupling_update["subsidy"] = {
            'proportional_uniform': None,
            'heater': {
                'policy': 'subsidy_ad_valorem',
                'proportional': None,
                'cap': None
            },
            'insulation': {
                'policy': 'subsidy_ad_valorem',
                'rational_behavior': True,
                'target': None,
                'proportional': None,
                'cap': None
            }
        }
    elif design == "MWh_uni":
        config_coupling_update["fix_sub_heater"] = True  # pour l'instant, c'est juste pour forcer à simuler une seule valeur de subvention, qui sera la même pour insulation et heater.
        config_coupling_update["max_iter"] = max_iter_single_iteration
        config_coupling_update["subsidy"] = {
            'proportional_uniform': True,
            'heater': {
                'policy': 'subsidy_proportional',
                'proportional': "MWh_cumac",
                'cap': cap_MWh
            },
            'insulation': {
                'policy': 'subsidy_proportional',
                'rational_behavior': False,
                'target': None,
                'proportional': "MWh_cumac",
                'cap': cap_MWh
            }
        }
    elif design == "proportional":  # we changed name to be more explicit
        config_coupling_update["subsidy"] = {
            'proportional_uniform': False,
            'heater': {
                'policy': 'subsidy_ad_valorem',
                'proportional': None,
                'cap': None
            },
            'insulation': {
                'policy': 'subsidy_proportional',
                'rational_behavior': False,
                'target': None,
                'proportional': "MWh_cumac",
                'cap': cap_MWh
            }
        }
    elif design == "MWh_sep":
        config_coupling_update["subsidy"] = {
            'proportional_uniform': False,
            'heater': {
                'policy': 'subsidy_proportional',
                'proportional': "MWh_cumac",
                'cap': cap_MWh
            },
            'insulation': {
                'policy': 'subsidy_proportional',
                'rational_behavior': False,
                'target': None,
                'proportional': "MWh_cumac",
                'cap': cap_MWh
            }
        }
    elif design == "tCO2_uni":
        config_coupling_update["fix_sub_heater"] = True  # pour l'instant, c'est juste pour forcer à simuler une seule valeur de subvention, qui sera la même pour insulation et heater.
        config_coupling_update["max_iter"] = max_iter_single_iteration
        config_coupling_update["subsidy"] = {
            'proportional_uniform': True,
            'heater': {
                'policy': 'subsidy_proportional',
                'proportional': "tCO2_cumac",
                'cap': cap_tCO2
            },
            'insulation': {
                'policy': 'subsidy_proportional',
                'rational_behavior': False,
                'target': None,
                'proportional': "tCO2_cumac",
                'cap': cap_tCO2
            }
        }
    elif design == "tCO2_sep":
        config_coupling_update["subsidy"] = {
            'proportional_uniform': False,
            'heater': {
                'policy': 'subsidy_proportional',
                'proportional': "tCO2_cumac",
                'cap': cap_tCO2
            },
            'insulation': {
                'policy': 'subsidy_proportional',
                'rational_behavior': False,
                'target': None,
                'proportional': "tCO2_cumac",
                'cap': cap_tCO2
            }
        }
    else:  # 'MWh_tCO2'
        assert design == 'MWh_tCO2', "Design not correctly specified"
        config_coupling_update["subsidy"] = {
            'proportional_uniform': False,
            'heater': {
                'policy': 'subsidy_proportional',
                'proportional': "tCO2_cumac",
                'cap': cap_tCO2
            },
            'insulation': {
                'policy': 'subsidy_proportional',
                'rational_behavior': False,
                'target': None,
                'proportional': "MWh_cumac",
                'cap': cap_MWh
            }
        }
    return config_coupling_update


def modif_config_eoles(config_eoles, config_coupling):
    """Modify EOLES configuration based on specified options in config_coupling.
    Namely, we modify: maximum capacity evolution"""
    config_eoles_update = deepcopy(config_eoles)

    aggregated_potential = config_coupling["eoles"]['aggregated_potential']

    # Choice of evolution of capacity
    if aggregated_potential:
        config_eoles_update["maximum_capacity_evolution"] = "eoles/inputs/technology_potential/maximum_capacity_evolution_aggregated.csv"
    else:
        config_eoles_update["maximum_capacity_evolution"] = "eoles/inputs/technology_potential/maximum_capacity_evolution.csv"

    if config_coupling['greenfield']:
        config_eoles_update["maximum_capacity_evolution"] = "eoles/inputs/technology_potential/maximum_capacity_greenfield.csv"

    # Choice of scenario for available potential
    assert config_coupling["eoles"]['maximum_capacity_scenario'] in ['N1', 'Opt', 'N1nuc', 'N1ren', 'N1ren2', 'N1offshore', 'N1pv', 'N1onshore'], "Scenario for capacity evolution is not correctly specified"
    config_eoles_update["maximum_capacity_evolution_scenario"] = config_coupling["eoles"]['maximum_capacity_scenario']

    assert config_coupling["eoles"]["biomass_potential_scenario"] in ["S3", "S2", "S2p", "S0"], "Biomass potential scenario is not specified correctly in config_coupling."
    config_eoles_update["biomass_potential_scenario"] = config_coupling["eoles"]["biomass_potential_scenario"]

    assert config_coupling["eoles"]["demand_scenario"] in ["Reference", "Reindustrialisation", "Sobriete", "Electrification+"], "Demand scenario is not specified correctly in config_coupling."
    config_eoles_update["demand_scenario"] = config_coupling["eoles"]["demand_scenario"]

    if 'prices' in config_coupling['eoles'].keys():
        config_eoles_update['energy_prices']['rate'].update(config_coupling['eoles']['prices']['rate'])

    # if 'energy' in config_coupling['resirf'].keys():  # we also need to modify EOLES configuration file
    #     new_rate = config_coupling['resirf']['energy']['eoles']['rate']
    #     config_eoles_update['energy_prices']['rate'].update(new_rate)  # we modify growth rate for eoles as well
    #     print(config_eoles_update['energy_prices'])

    if 'costs_supply' in config_coupling['eoles'].keys():  # we modify the costs of supply
        if 'storage_capex_variant' in config_coupling['eoles']['costs_supply'].keys():  # variant for some technologies
            config_eoles_update["storage_capex_variant"] = config_coupling['eoles']['costs_supply']['storage_capex_variant']
        if 'capex_variant' in config_coupling['eoles']['costs_supply'].keys():  # variant for some technologies
            config_eoles_update["capex_variant"] = config_coupling['eoles']['costs_supply']['capex_variant']
        if 'storage_capex' in config_coupling['eoles']['costs_supply'].keys():  # variant for all technologies
            new_storage_capex = config_coupling['eoles']['costs_supply']['storage_capex']
            new_storage_file = f"eoles/inputs/technology_characteristics/{new_storage_capex}.csv"
            assert Path(new_storage_file).is_file(), "Scenario for storage capex is not correctly specified"
            config_eoles_update["storage_capex"] = new_storage_file
        if 'capex' in config_coupling['eoles']['costs_supply'].keys():  # variant for all technologies
            new_capex = config_coupling['eoles']['costs_supply']['capex']
            new_capex_file = f"eoles/inputs/technology_characteristics/{new_capex}.csv"
            assert Path(new_capex_file).is_file(), "Scenario for capex is not correctly specified"
            config_eoles_update["capex"] = new_capex_file

    if 'carbon_budget' in config_coupling['eoles'].keys():
        carbon_budget_spec = config_coupling["eoles"]['carbon_budget']
        config_eoles_update["carbon_budget"] = f"eoles/inputs/technical/{carbon_budget_spec}.csv"

    if 'carbon_budget_resirf' in config_coupling['eoles'].keys():  # parameter used only if we optimize without considering the EOLES module
        carbon_budget_resirf_spec = config_coupling["eoles"]['carbon_budget_resirf']
        config_eoles_update["carbon_budget_resirf"] = f"eoles/inputs/technical/{carbon_budget_resirf_spec}.csv"

    if 'load_factors' in config_coupling['eoles'].keys():  # we modify the considered weather years
        load_factors = config_coupling['eoles']["load_factors"]
        config_eoles_update["load_factors"] = f"eoles/inputs/hourly_profiles/{load_factors}.csv"
        lake_inflows = config_coupling['eoles']["lake_inflows"]
        config_eoles_update["lake_inflows"] = f"eoles/inputs/hourly_profiles/{lake_inflows}.csv"
        config_eoles_update["nb_years"] = config_coupling['eoles']['nb_years']
        config_eoles_update["input_years"] = config_coupling['eoles']['input_years']

    if 'district_heating_potential' in config_coupling['eoles'].keys():  # we modify the available potential for district heating
        district_heating_potential = config_coupling["eoles"]['district_heating_potential']
        path_dh_potential = Path("eoles") / "inputs" / "technology_potential" / f"district_heating_potential_{district_heating_potential}.csv"
        assert path_dh_potential.is_file(), "Scenario for district heating potential is not correctly specified"
        config_eoles_update['district_heating_potential'] =  path_dh_potential

    if "worst_case" in config_coupling["eoles"].keys():  # definition of worst case scenario for EOLES
        if config_coupling["eoles"]["worst_case"]:
            config_eoles_update["capex"] = "eoles/inputs/technology_characteristics/overnight_capex_evolution_high.csv"
            config_eoles_update["storage_capex"] =  "eoles/inputs/technology_characteristics/storage_capex_evolution_high.csv"
            config_eoles_update["maximum_capacity_evolution"] = "eoles/inputs/technology_potential/maximum_capacity_evolution.csv"
            config_eoles_update["maximum_capacity_evolution_scenario"] = "N1"
            config_eoles_update["capacity_factor_nuclear"] = 0.7
            config_eoles_update["biomass_potential"] = f"eoles/inputs/technology_potential/biomass_evolution_S2.csv"

    if "h2_ccgt" in config_coupling["eoles"].keys():
        if not config_coupling["eoles"]["h2_ccgt"]:  # we do not allow h2 ccgt plants
            if "fix_capacities" in config_coupling["scenario_cost_eoles"].keys():
                config_coupling["scenario_cost_eoles"]["fix_capa"]["h2_ccgt"] = 0
            else:
                config_coupling["scenario_cost_eoles"]["fix_capacities"] = {
                    "h2_ccgt": 0
                }

    return config_eoles_update, config_coupling


def modif_config_resirf(config_resirf, config_coupling):
    """This function modifies the ResIRF configuration file based on specified options in config_coupling.
    Namely, we modify: supply, premature replacement, rational behavior, carbon content hypothesis, prices."""
    config_resirf_update = deepcopy(config_resirf)  # corresponds to the resirf configuration file for coupling

    if 'file' in config_resirf_update.keys():  # we load the reference file with all the default options for ResIRF
        path_reference = config_resirf_update['file']
        with open(path_reference) as file:
            config_reference = json.load(file)

    # # Modification supply value
    # config_resirf_update["supply"]["activated_insulation"] = config_coupling["supply_insulation"]
    # config_resirf_update["supply"]["activated_heater"] = config_coupling["supply_heater"]

    # Modification end year
    if 'end' in config_coupling.keys():
        config_resirf_update["end"] = config_coupling["end"]

    if 'carbon_emissions_resirf' in config_coupling.keys():  # modification of carbon emissions assumptions
        config_resirf_update['energy'] = config_reference['energy']
        assert Path(config_coupling['carbon_emissions_resirf']).is_file(), "Carbon emissions as specified are not a correct file"
        config_resirf_update['energy']['carbon_emission'] = config_coupling['carbon_emissions_resirf']

    if 'method_health_cost' in config_coupling['resirf'].keys():  # in that case, we specify the method to estimate health costs
        config_resirf_update['method_health_cost'] = config_coupling['resirf']['method_health_cost']

    if 'technical' in config_coupling['resirf'].keys():
        if 'technical' in config_resirf_update.keys():  # there is already a dictionary specifying some options for technical
            for k,v in config_resirf_update['technical'].items():
                if k in config_coupling['resirf']['technical'].keys():  # needs updating in this case
                    if isinstance(v, dict):
                        v.update(config_coupling['resirf']['technical'][k])
                    else:
                        config_resirf_update['technical'][k] = config_coupling['resirf']['technical'][k]
        else:  # we create the technical dictionary
            config_resirf_update['technical'] = config_coupling['resirf']['technical']

    if 'prices' in config_coupling['resirf'].keys():
        assert 'energy' in config_coupling['resirf'].keys(), 'Energy should be defined in the keys of the configuration'
        config_coupling['resirf']['energy']['energy_prices']['rate'].update(deepcopy(config_coupling['resirf']['prices']['rate']))  # we update prices
        if 'energy' in config_resirf_update.keys():  # there is already a dictionary specifying some options for energy
            for k,v in config_resirf_update['energy'].items():
                if k in config_coupling['resirf']['energy'].keys():  # needs updating in this case
                    if isinstance(v, dict):
                        v.update(config_coupling['resirf']['energy'][k])
                    else:
                        config_resirf_update['energy'][k] = config_coupling['resirf']['energy'][k]
        else:  # we create the energy dictionary
            config_resirf_update['energy'] = config_coupling['resirf']['energy']

    # if 'energy' in config_coupling['resirf'].keys():
    #     if 'energy' in config_resirf_update.keys():  # there is already a dictionary specifying some options for energy
    #         for k,v in config_resirf_update['energy'].items():
    #             if k in config_coupling['resirf']['energy'].keys():  # needs updating in this case
    #                 if isinstance(v, dict):
    #                     v.update(config_coupling['resirf']['energy'][k])
    #                 else:
    #                     config_resirf_update['energy'][k] = config_coupling['resirf']['energy'][k]
    #     else:  # we create the energy dictionary
    #         config_resirf_update['energy'] = config_coupling['resirf']['energy']

    if 'switch_heater' in config_coupling['resirf'].keys():
        if 'switch_heater' in config_resirf_update.keys():  # there is already a dictionary specifying some options for switch_heater
            for k,v in config_resirf_update['switch_heater'].items():
                if k in config_coupling['resirf']['switch_heater'].keys():  # needs updating in this case
                    if isinstance(v, dict):
                        v.update(config_coupling['resirf']['switch_heater'][k])
                    else:
                        config_resirf_update['switch_heater'][k] = config_coupling['resirf']['switch_heater'][k]
        else:  # we create the switch_heater dictionary
            config_resirf_update['switch_heater'] = config_coupling['resirf']['switch_heater']

    if 'policies' in config_coupling.keys():
        config_resirf_update['policies'] = config_coupling['policies']

    # Modification rational behavior insulation
    config_resirf_update["renovation"] = config_reference["renovation"]
    config_resirf_update["renovation"]["rational_behavior"]["activated"] = config_coupling["subsidy"]['insulation']["rational_behavior"]

    if 'lifetime_insulation' in config_coupling.keys():
        config_resirf_update["renovation"]["lifetime_insulation"] = config_coupling["lifetime_insulation"]

    if 'social' in config_coupling['subsidy']['insulation'].keys():
        config_resirf_update['renovation']["rational_behavior"]["social"] = config_coupling["subsidy"]['insulation']["social"]

    if "prices_constant" in config_coupling.keys():  # this hypothesis is always specified in the config_coupling dictionary
        if config_coupling["prices_constant"]:  # if prices are to stay constant, we add this option in the config_resirf file
            config_resirf_update['simple'] = {}
            config_resirf_update["simple"]["prices_constant"] = True

    if "information_rate" in config_coupling.keys():
        config_resirf_update['switch_heater']["information_rate"] = config_coupling["information_rate"]

    if 'no_MF' in config_coupling.keys():
        if config_coupling['no_MF']:  # we want to remove market failures
            policy_noMF = {"landlord": {
                                          "start": 2020,
                                          "end": 2051,
                                          "policy": "regulation",
                                          "gest": "insulation"
                                        },
                          "multi-family": {
                                            "start": 2020,
                                            "end": 2051,
                                            "policy": "regulation",
                                            "gest": "insulation"
                                          }
            }
            config_resirf_update["policies"].update(policy_noMF)

    return config_resirf_update


def check_required_keys_additional(config_additional):
    """
    Checks that the provided configuration dictionary includes the required keys.
    :param config_additional: dict

    """
    required_keys = ['greenfield', 'prices_constant', 'price_feedback', 'lifetime_insulation', 'optim_eoles',
                    'carbon_budget', 'district_heating_potential', 'biomass_potential_scenario', 'demand_scenario', 'aggregated_potential', 'maximum_capacity_scenario']
    assert set(required_keys).issubset(config_additional.keys()), "Some required keys in config_additional are missing"


def check_required_keys_base(config_coupling):
    required_keys = ['no_subsidies', 'subsidies_specified', 'calibration', 'eoles', 'subsidy', 'discount_rate', 'max_iter', 'fix_sub_heater',
                     'fix_sub_insulation', 'health', 'carbon_constraint', 'list_year', 'list_trajectory_scc']
    assert set(required_keys).issubset(config_coupling.keys()), "Some required keys in config_coupling are missing"

def create_configs_coupling(list_design, config_coupling: dict, config_additional: dict, dict_configs=None):
    """
    Creates a list of configs to test from different specified parameters.
    :param list_design:
        List of designs of subsidies
    :param config_coupling:
        initial configuration
    :param config_additional:
        additional parameters for the configuration
    :return:
    """
    check_required_keys_base(config_coupling)  # check that all required keys are included in the coupling configuration
    if 'weather_year' not in config_coupling.keys():  # we accomodate all configuration files which did not include the specification for the weather year
        config_coupling['weather_year'] = 2006
    check_required_keys_additional(config_additional)  # check that all required keys are included in the additional configuration

    config_coupling_update = deepcopy(config_coupling)
    config_coupling_update['greenfield'] = config_additional['greenfield']
    config_coupling_update['prices_constant'] = config_additional['prices_constant']
    config_coupling_update['price_feedback'] = config_additional['price_feedback']
    config_coupling_update['lifetime_insulation'] = config_additional['lifetime_insulation']
    config_coupling_update['eoles']['biomass_potential_scenario'] = config_additional['biomass_potential_scenario']
    config_coupling_update['eoles']['demand_scenario'] = config_additional['demand_scenario']
    config_coupling_update['eoles']['aggregated_potential'] = config_additional['aggregated_potential']
    config_coupling_update['eoles']['maximum_capacity_scenario'] = config_additional['maximum_capacity_scenario']
    config_coupling_update['optim_eoles'] = config_additional['optim_eoles']
    # config_coupling_update['electricity_constant'] = config_additional['electricity_constant']
    carbon_budget = config_additional['carbon_budget']
    district_heating_potential = config_additional['district_heating_potential']

    config_coupling_update['resirf'] = {}  # we create a dictionary for the ResIRF configuration specs

    if 'prices' in config_additional.keys():
        config_coupling_update['resirf']['prices'] = config_additional['prices']['resirf']
        config_coupling_update['resirf']['energy'] = config_additional['energy']  # necessary to update prices afterwards
        config_coupling_update['eoles']['prices'] = config_additional['prices']['eoles']

    if 'technical' in config_additional.keys():  # we modify the specification for the Res-IRF configuration
        config_coupling_update['resirf']['technical'] = config_additional['technical']

    # if 'energy' in config_additional.keys():  # we modify the specification for the Res-IRF configuration
    #     config_coupling_update['resirf']['energy'] = config_additional['energy']

    if 'switch_heater' in config_additional.keys():  # we modify the specification for the Res-IRF configuration
        config_coupling_update['resirf']['switch_heater'] = config_additional['switch_heater']

    # if 'hourly_profile' in config_additional.keys():
    #     hourly_profile = config_additional['hourly_profile']
    #     config_coupling_update['hourly_profile'] = f'project/input/technical/{hourly_profile}.csv'

    if 'costs_supply' in config_additional.keys():
        config_coupling_update['eoles']['costs_supply'] = config_additional['costs_supply']

    if 'method_health_cost' in config_additional.keys():
        config_coupling_update['resirf']['method_health_cost'] = config_additional['method_health_cost']

    if "policies" in config_additional.keys():  # we update the default policies in ResIRF
        policies = config_additional["policies"]
        assert type(policies) is dict  ## we should provide a dictionary in this case
        config_coupling_update['policies'] = policies

    if 'carbon_emissions_resirf' in config_additional.keys():  # carbon emissions are now specified in the files from Res-IRF
        if config_additional['carbon_emissions_resirf'] is not None:
            carbon_emissions_resirf = config_additional['carbon_emissions_resirf']
            config_coupling_update['carbon_emissions_resirf'] = f'project/input/energy/{carbon_emissions_resirf}.csv'
    if carbon_budget is not None:
        config_coupling_update['eoles']['carbon_budget'] = carbon_budget
    if 'carbon_budget_resirf' in config_additional.keys():
        if config_additional['carbon_budget_resirf'] is not None:
            config_coupling_update['eoles']['carbon_budget_resirf'] = config_additional['carbon_budget_resirf']
    if district_heating_potential is not None:  # we specify another potential for district heating (based on one of ADEME scenario)
        config_coupling_update['eoles']['district_heating_potential'] = district_heating_potential
    if 'weather' in config_additional.keys():  # we add specification for other weather years
        assert 'load_factors' in config_additional['weather'].keys(), 'Modification of weather is specified, but missing specification for load factors'
        config_coupling_update['eoles']['load_factors'] = config_additional['weather']['load_factors']
        assert 'lake_inflows' in config_additional['weather'].keys(), 'Modification of weather is specified, but missing specification for lake inflows'
        config_coupling_update['eoles']['lake_inflows'] = config_additional['weather']['lake_inflows']
        assert 'nb_years' in config_additional['weather'].keys(), 'Modification of weather is specified, but missing specification for number of years'
        config_coupling_update['eoles']['nb_years'] = config_additional['weather']['nb_years']
        assert 'input_years' in config_additional['weather'].keys(), 'Modification of weather is specified, but missing specification for included years'
        config_coupling_update['eoles']['input_years'] = config_additional['weather']['input_years']
        config_coupling_update['weather_year'] = config_additional['weather']['input_years'][0]  # we specify the weather year for thermosensible demand in ResIRF.
        # For now, only accomodates a single year as a stress test, not multiple years.

    if dict_configs is None:
        dict_configs = {}

    if list_design is not None:  # we want to update the subsidy design
        for design in list_design:
            name_config = config_additional['name_config']
            name_config = f"{design}_{name_config}"
            if config_additional['subsidies_heater'] is not None:  # in this case, we have specified the value for the subsidies in the configuration file, for each design, as a dictionary. There should be a function to extract those values.
                sub_heater = config_additional['subsidies_heater'][design]
            else:
                sub_heater = None
            if config_additional['subsidies_insulation'] is not None:
                sub_insulation = config_additional['subsidies_insulation'][design]
            else:
                sub_insulation = None
            dict_configs[name_config] = modif_config_coupling(design, config_coupling_update, cap_MWh=config_additional['cap_MWh'],
                                                              cap_tCO2=config_additional['cap_tCO2'],
                                                              subsidies_heater=sub_heater, subsidies_insulation=sub_insulation)
    else:
        name_config = config_additional['name_config']
        dict_configs[name_config] = config_coupling_update
    return dict_configs


def extract_subsidy_value(list_folder, name_config: str):
    """
    Function to extract the value of the subsidies found in the algorithm.
    :param folder: str
        Name of folder where we can find the required results
    :param name_config: str
        Name of configuration used to save the results. For example, "S2_N1_pricefeedback_hcDPE" (this is basically the string at the end of the folder file)
    :return:
    """
    subsidies_heater_dict, subsidies_insulation_dict = {}, {}
    for folder in list_folder:
        if os.path.isdir(folder):
            with open(os.path.join(folder, 'coupling_results.pkl'), "rb") as file:
                output = load(file)
            subsidies = output["Subsidies (%)"]
            subsidies_insulation = subsidies['Insulation'].tolist()
            subsidies_heater = subsidies['Heater'].tolist()
            len_name_config = len(name_config.split('_'))
            name_design = folder.split('/')[-1].split('_')
            name_design = '_'.join(name_design[2:len(name_design) - len_name_config])
            subsidies_heater_dict[name_design] = subsidies_heater
            subsidies_insulation_dict[name_design] = subsidies_insulation
    return subsidies_heater_dict, subsidies_insulation_dict


def find_folders(base_folder, target_string):
    """Gets all files inside a folder which matches a certain pattern"""
    matching_folders = []

    # Iterate over all items in the base folder
    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)

        # Check if the item is a directory and contains the target string
        if os.path.isdir(item_path) and target_string in item:
            matching_folders.append(item_path)

    return matching_folders


def config_resirf_exogenous(sensitivity, config_resirf):
    """This function creates a dictionary of ResIRF configuration files, for exogenous scenarios."""

    dict_config_resirf = {}
    # dict_config_resirf["Reference"] = config_resirf
    if "policies" in sensitivity.keys():
        for scenario in sensitivity["policies"].keys():
            new_config = deepcopy(config_resirf)
            new_config["policies"] = sensitivity["policies"][scenario]
            dict_config_resirf[scenario] = new_config
    if "no_policy" in sensitivity.keys():
        new_config = deepcopy(config_resirf)
        new_config["simple"]["no_policy"] = True
        dict_config_resirf["No policy"] = new_config
    if "current_policies" in sensitivity.keys():
        new_config = deepcopy(config_resirf)
        new_config["simple"]["current_policies"] = True
        dict_config_resirf["Current policies"] = new_config
    return dict_config_resirf


def create_multiple_coupling_configs2(sensitivity, config_resirf, config_coupling):
    dict_config_coupling = {}
    for scenario in sensitivity.keys():
        new_config_coupling = deepcopy(config_coupling)
        new_config_resirf = deepcopy(config_resirf)
        options = sensitivity[scenario]
        if "policies" in options.keys():
            new_config_resirf["policies"] = options["policies"]
        if "no_policy" in options.keys():
            new_config_resirf["simple"]["no_policy"] = True
        if "current_policies" in options.keys():
            new_config_resirf["simple"]["current_policies"] = True
        if "price_feedback" in options.keys():
            new_config_coupling["price_feedback"] = options["price_feedback"]
        new_config_coupling["config_resirf"] = new_config_resirf
        dict_config_coupling[scenario] = new_config_coupling
    return dict_config_coupling


def create_multiple_coupling_configs(dict_config_resirf, config_coupling):
    """Creates a dictionary of coupling configurations, with different ResIRF configurations."""
    dict_config_coupling = {}
    for scenario in dict_config_resirf.keys():
        new_config = deepcopy(config_coupling)
        new_config["config_resirf"] = dict_config_resirf[scenario]
        dict_config_coupling[scenario] = new_config
    return dict_config_coupling


def ldmi_method_resirf(output, carbon_content):
    start, end = 2020, 2049

    heater_vector, energy_vector = ["Electricity-Heat pump water", "Electricity-Performance boiler", "Heating-District heating",
                     "Natural gas-Performance boiler", "Oil fuel-Performance boiler", "Wood fuel-Performance boiler"], ["Electricity", "Natural gas", "Oil fuel", "Wood fuel", "Heating"]
    # select only heater_vector that have 'Surface {} (Million m2)' in output index
    heater_vector = [i for i in heater_vector if 'Surface {} (Million m2)'.format(i) in output.index]

    # Select rows
    rows = []
    rows += ['Surface (Million m2)']
    rows += ['Surface {} (Million m2)'.format(i) for i in heater_vector]
    rows += ['Consumption standard {} (TWh)'.format(i) for i in heater_vector]
    rows += ['Consumption {} (TWh)'.format(i) for i in heater_vector]
    rows += ['Emission content {} (gCO2/kWh)'.format(i) for i in energy_vector]
    rows = [i for i in rows if i in output.index]
    data = output.loc[rows, [start, end]]
    data = data.rename(columns={2049: 2050})
    end = 2050

    # Prepare indicators
    for i in heater_vector:
        data.loc['Share surface {} (%)'.format(i), :] = data.loc['Surface {} (Million m2)'.format(i), :] / data.loc[
            'Surface (Million m2)', :]
        data.loc['Consumption standard {} (TWh/m2)'.format(i), :] = data.loc['Consumption standard {} (TWh)'.format(i), :] / data.loc[
            'Surface {} (Million m2)'.format(i), :]
        data.loc['Heating intensity {} (%)'.format(i), :] = data.loc['Consumption {} (TWh)'.format(i), :] / data.loc[
            'Consumption standard {} (TWh)'.format(i), :]
        data.loc['Emission content {} (gCO2/kWh)'.format(i), :] = data.loc['Emission content {} (gCO2/kWh)'.format(i.split('-')[0]), :]
        # data.loc['Emission {} (MtCO2)'.format(i), :] = data.loc['Consumption {} (TWh)'.format(i), :] * data.loc[
        #     'Emission content {} (gCO2/kWh)'.format(i), :] / 1000

    carbon_content = carbon_content.rename(index={'Emission content Gas (gCO2/kWh)': 'Emission content Natural gas-Performance boiler (gCO2/kWh)',
                                                  'Emission content District heating (gCO2/kWh)': 'Emission content Heating-District heating (gCO2/kWh)'})  # we keep same notations as in Res-IRF (even though it is not natural gas per se)
    carbon_content.loc['Emission content Electricity-Heat pump water (gCO2/kWh)'] = carbon_content.loc['Emission content Electricity heating (gCO2/kWh)']
    carbon_content.loc['Emission content Electricity-Performance boiler (gCO2/kWh)'] = carbon_content.loc['Emission content Electricity heating (gCO2/kWh)']

    carbon_content = carbon_content.drop(index=['Emission content Electricity heating (gCO2/kWh)', 'Emission content Electricity heating daily (gCO2/kWh)'])
    if 2020 not in carbon_content.columns:
        carbon_content.loc[:, 2020] = [229, 101, 33, 33]
    carbon_content = carbon_content.sort_index(axis=1, ascending=True)

    list_energy = ["Electricity-Heat pump water", "Electricity-Performance boiler", "Heating-District heating", "Natural gas-Performance boiler"]
    data = data.drop(index=[f'Emission content {energy} (gCO2/kWh)' for energy in list_energy])
    data = pd.concat([data, carbon_content.loc[:,[2020, 2050]]], axis=0)

    # in case heating system is not used at the end of the period
    data.fillna(0, inplace=True)

    for i in heater_vector:
        data.loc['Emission {} (MtCO2)'.format(i), :] = data.loc['Consumption {} (TWh)'.format(i), :] * data.loc[
            'Emission content {} (gCO2/kWh)'.format(i), :] / 1000

    # Calculate individual effect
    channels_heater = ['Surface (Million m2)', 'Share surface {} (%)', 'Consumption standard {} (TWh/m2)',
                       'Heating intensity {} (%)', 'Emission content {} (gCO2/kWh)']
    result = {}
    manual_treatment = []
    for i in heater_vector:
        # no more emission if no more surface
        if data.loc['Surface {} (Million m2)'.format(i), end] == 0:
            # result['Surface {} (Million m2)'.format(i)] = - data.loc['Emission {} (MtCO2)'.format(i), start]
            manual_treatment += [i]

    for channel in channels_heater:
        result[channel.split(' {}')[0]] = sum([log(data.loc[channel.format(i), end] / data.loc[channel.format(i), start]) *
                         (data.loc['Emission {} (MtCO2)'.format(i), end] - data.loc['Emission {} (MtCO2)'.format(i), start]) /
                         log(data.loc['Emission {} (MtCO2)'.format(i), end] / data.loc['Emission {} (MtCO2)'.format(i), start]) for i in heater_vector if i not in manual_treatment])

    for i in manual_treatment:  # we attribute effect of emissions to switch
        result['Share surface'] += data.loc['Emission {} (MtCO2)'.format(i), end] - data.loc['Emission {} (MtCO2)'.format(i), start]

    rename = {'Surface (Million m2)': 'Surface',
            'Share surface': 'Fuel switch',
            'Consumption standard': 'Insulation',
            'Heating intensity': 'Heating intensity',
            'Emission content': 'Fuel decarbonization'}
    result = pd.Series({rename[k]: v for k, v in result.items()})
    emission = data.loc[[f'Emission {energy} (MtCO2)' for energy in heater_vector],[start,end]].sum()
    return result, emission

def ldmi_method(output_global, efficiency, carbon_content):
    """Estimates emissions reduction attribution across different channels with the LDMI method."""
    # TODO: attention, dans les anciennes simulations, le contenu carbone du DH est exogène alors qu'il est endogène dans nos simulations. A prendre en compte dans les analyses.
    output = output_global.copy()
    output = output.fillna(0)  # we fill NA values at 0
    # energy_vector = ["Natural gas", "District heating", "Wood fuel", "Oil fuel", "Heat pump", "Direct electric"]
    energy_vector = ["Electricity-Heat pump water", "Electricity-Performance boiler", "Heating-District heating",
                     "Natural gas-Performance boiler", "Oil fuel-Performance boiler", "Wood fuel-Performance boiler"]

    for energy in energy_vector:
        output.loc[f'Share {energy} (%)'] = output.loc[f'Surface {energy} (Million m2)'] / output.loc["Surface (Million m2)"]  # share of energy in total useful energy consumption
        output.loc[f'Consumption standard {energy} (TWh/m2)'] = output.loc[f'Consumption standard {energy} (TWh)'] / output.loc[f'Surface {energy} (Million m2)']
        output.loc[f'Heating intensity {energy} (%)'] = output.loc[f'Consumption {energy} (TWh)'] / output.loc[f'Consumption standard {energy} (TWh)']
        output.loc[f'Emission content {energy} (gCO2/kWh)'] = output.loc['Emission content {} (gCO2/kWh)'.format(energy.split('-')[0])]

    subset_rows = ["Surface (Million m2)", 'Consumption (TWh)']
    subset_rows = subset_rows + [f"Consumption {energy} (TWh)" for energy in energy_vector]+ [f"Share {energy} (%)" for energy in energy_vector] + [f'Consumption standard {energy} (TWh/m2)' for energy in energy_vector]\
                  + [f"Heating intensity {energy} (%)" for energy in energy_vector]
    subset_columns = [2020, 2049]

    # output = output.rename(index={'Emission content Heating (gCO2/kWh)': 'Emission content District heating (gCO2/kWh)'})  # use same names as in my model
    output.loc['Emission content Wood fuel-Performance boiler (gCO2/kWh)'] = 0  # we impose a zero carbon content for wood (as we do in EOLES)

    carbon_content = carbon_content.rename(index={'Emission content Gas (gCO2/kWh)': 'Emission content Natural gas-Performance boiler (gCO2/kWh)',
                                                  'Emission content District heating (gCO2/kWh)': 'Emission content Heating-District heating (gCO2/kWh)'})  # we keep same notations as in Res-IRF (even though it is not natural gas per se)
    carbon_content.loc['Emission content Electricity-Heat pump water (gCO2/kWh)'] = carbon_content.loc['Emission content Electricity heating (gCO2/kWh)']
    carbon_content.loc['Emission content Electricity-Performance boiler (gCO2/kWh)'] = carbon_content.loc['Emission content Electricity heating (gCO2/kWh)']

    carbon_content = carbon_content.drop(index=['Emission content Electricity heating (gCO2/kWh)', 'Emission content Electricity heating daily (gCO2/kWh)'])

    if 'Emission content Heating-District heating (gCO2/kWh)' in carbon_content.index:  # new processing
        additional_rows = ['Emission content Oil fuel-Performance boiler (gCO2/kWh)', 'Emission content Wood fuel-Performance boiler (gCO2/kWh)']
    else:
        additional_rows = ['Emission content Oil fuel-Performance boiler (gCO2/kWh)', 'Emission content Wood fuel-Performance boiler (gCO2/kWh)', 'Emission content District heating (gCO2/kWh)']

    subset_rows = subset_rows + additional_rows
    output = output.loc[subset_rows, subset_columns].rename(columns={2049: 2050})  # we rename the column 2049 to 2050

    # merge with carbon_content, to have the same columns
    # TODO: carbon content initial pas fourni pour les simulations endogènes
    try:
        output = pd.concat([output, carbon_content.loc[:,[2020, 2050]]], axis=0)
    except:
        carbon_content.loc[:,2020] = [229, 101, 33, 33]  # a modifier pour mettre les bonnes valeurs
        output = pd.concat([output, carbon_content.loc[:, [2020, 2050]]], axis=0)

    for energy in energy_vector:
        output.loc[f'CO2 emissions {energy} (MtCO2)'] = output.loc[f'Consumption {energy} (TWh)'] * output.loc[f'Emission content {energy} (gCO2/kWh)'] / 1000

    output.loc['CO2 emissions (MtCO2)'] = sum(output.loc[f'CO2 emissions {energy} (MtCO2)'] for energy in energy_vector)

    output_energy = output.loc[[row for row in output.index if any(energy in row for energy in energy_vector)], :]  # subset specific to each energy vector

    new_index = []
    for col in output_energy.index:
        parts = col.split(' (')[0].split(' ')
        new_index.append((' '.join(parts[:-2]), ' '.join(parts[-2:])))
    output_energy.index = pd.MultiIndex.from_tuples(new_index)
    output_CO2 = output_energy.loc['CO2 emissions']
    output_channel = output_energy.drop(index='CO2 emissions', level=0)
    output_channel.loc[:,'Ratio log'] = np.log(output_channel.loc[:,2050] / output_channel.loc[:,2020])

    output_CO2.loc[:,'Delta CO2 emissions'] = output_CO2.loc[:,2050] - output_CO2.loc[:,2020]
    output_CO2.loc[:, 'log Delta CO2 emissions'] = np.log(output_CO2.loc[:, 2050] / output_CO2.loc[:, 2020])

    multiindex = pd.MultiIndex.from_product([output_channel.index.get_level_values(0).unique(),output_CO2.index])
    tmp = pd.concat([output_CO2.loc[:,['Delta CO2 emissions', 'log Delta CO2 emissions']]]*len(output_channel.index.get_level_values(0).unique()))
    tmp.index = multiindex
    output_channel = pd.concat([output_channel, tmp], axis=1)  # we concatenate CO2 emissions information along columns axis

    output_channel.loc[:, 'Total channel'] = output_channel.loc[:,'Ratio log'] * output_channel.loc[:,'Delta CO2 emissions'] / output_channel.loc[:,'log Delta CO2 emissions']
    output_channel = output_channel.loc[:,"Total channel"].to_frame().T.stack().unstack(0).T.sum(axis=1).droplevel(1)  # we get the channels
    output_channel = output_channel[['Emission content', 'Heating intensity', 'Share', 'Consumption standard']]

    output = output.loc[['Surface (Million m2)', 'Consumption (TWh)']]
    output.loc[:, 'Ratio log'] = np.log(output.loc[:, 2050] / output.loc[:, 2020])

    output_CO2.loc[:,'Ratio'] = output_CO2.loc[:,'Delta CO2 emissions'] / output_CO2.loc[:,'log Delta CO2 emissions']

    channel_m2 = output.loc['Surface (Million m2)', 'Ratio log'] * output_CO2.sum(axis=0)['Ratio']
    # channel_insulation = output.loc['Consumption useful (TWh/m2)', 'Ratio log'] * output_CO2.sum(axis=0)['Ratio']
    output_channel = pd.concat([output_channel, pd.Series(data=[channel_m2], index=['Surface'])])

    output_channel = output_channel.reindex(['Surface', 'Share', 'Consumption standard', 'Heating intensity', 'Emission content'])

    return output_channel, output_CO2


def ldmi_method_2(output_global, efficiency, carbon_content):
    # TODO: attention, il y a un facteur de multiplication entre le stock (en million) et la surface.
    output = output_global.copy()
    energy_vector = ["Natural gas", "District heating", "Wood fuel", "Oil fuel", "Heat pump", "Direct electric"]

    for energy in energy_vector:
        output.loc[f"Heating intensity {energy} (%)"] = output.loc["Heating intensity (%)"]  # heating intensity is assumed to be the same for all energy vectors
        output.loc[f'Share {energy} (%)'] = output.loc[f'Stock {energy} (Million)'] / output.loc["Stock (Million)"]  # share of energy in total useful energy consumption
        output.loc[f"Consumption standard {energy} (TWh)"] = output.loc[f"Consumption {energy} (TWh)"] / output.loc["Heating intensity (%)"]  # standard energy consumption
        output.loc[f'Insulation {energy} (%)'] = output.loc[f"Consumption standard {energy} (TWh)"] / output.loc[f'Stock {energy} (Million)']  # share of energy in total useful energy consumption

    # select subset of output, for subset of rows and columns
    # subset of rows is a list subset_rows, and subset of columns is a list subset_columns
    subset_rows = ["Surface (Million m2)"] + [f"Share {energy} (%)" for energy in energy_vector] + [f"Heating intensity {energy} (%)" for energy in energy_vector] + \
                  [f"Insulation {energy} (%)" for energy in energy_vector]
    subset_columns = [2020, 2049]

    output = output.rename(index={'Emission content Heating (gCO2/kWh)': 'Emission content District heating (gCO2/kWh)'})  # use same names as in my model
    output.loc['Emission content Wood fuel (gCO2/kWh)'] = 0  # we impose a zero carbon content for wood (as we do in EOLES)

    if 'Gas carbon content' in carbon_content.index:  # old rows names
        carbon_content = carbon_content.rename(index={'Gas carbon content': 'Emission content Gas (gCO2/kWh)',
                                                      'Electric heating carbon content': 'Emission content Electric heating (gCO2/kWh)',
                                                      'Electric heating carbon content daily': 'Emission content Electric heating daily (gCO2/kWh)'})
    carbon_content = carbon_content.rename(index={'Emission content Gas (gCO2/kWh)': 'Emission content Natural gas (gCO2/kWh)'})  # we keep same notations as in Res-IRF (even though it is not natural gas per se)
    carbon_content.loc['Emission content Heat pump (gCO2/kWh)'] = carbon_content.loc['Emission content Electric heating (gCO2/kWh)']
    carbon_content.loc['Emission content Direct electric (gCO2/kWh)'] = carbon_content.loc['Emission content Electric heating (gCO2/kWh)']

    carbon_content = carbon_content.drop(index=['Emission content Electric heating (gCO2/kWh)', 'Emission content Electric heating daily (gCO2/kWh)'])

    if 'Emission content Heating district (gCO2/kWh)' in carbon_content.index:  # new processing
        additional_rows = ['Emission content Oil fuel (gCO2/kWh)', 'Emission content Wood fuel (gCO2/kWh)']
    else:
        additional_rows = ['Emission content Oil fuel (gCO2/kWh)', 'Emission content Wood fuel (gCO2/kWh)', 'Emission content District heating (gCO2/kWh)']

    subset_rows = subset_rows + additional_rows
    output = output.loc[subset_rows, subset_columns].rename(columns={2049: 2050})  # we rename the column 2049 to 2050

    # merge with carbon_content, to have the same columns
    # TODO: carbon content pas fourni pour les simulations endogènes
    try:
        output = pd.concat([output, carbon_content.loc[:,[2020, 2050]]], axis=0)
    except:
        carbon_content.loc[:,2020] = 0  # a modifier pour mettre les bonnes valeurs
        output = pd.concat([output, carbon_content.loc[:, [2020, 2050]]], axis=0)

    for energy in energy_vector:
        output.loc[f'CO2 emissions {energy} (MtCO2)'] = output.loc["Surface (Million m2)"] * output.loc[f"Share {energy} (%)"] * output.loc[f"Insulation {energy} (%)"] * output.loc[f'Heating intensity {energy} (%)'] * output.loc[f'Emission content {energy} (gCO2/kWh)'] / 1000

    output.loc['CO2 emissions (MtCO2)'] = sum(output.loc[f'CO2 emissions {energy} (MtCO2)'] for energy in energy_vector)

    output_energy = output.loc[[row for row in output.index if any(energy in row for energy in energy_vector)],:]  # subset specific to each energy vector
    # Create a list of new column names
    new_index = []

    # Split the existing column names to create multiindex
    for col in output_energy.index:
        parts = col.split(' (')[0].split(' ')
        new_index.append((' '.join(parts[:-2]), ' '.join(parts[-2:])))
    output_energy.index = pd.MultiIndex.from_tuples(new_index)
    output_CO2 = output_energy.loc['CO2 emissions']
    output_channel = output_energy.drop(index='CO2 emissions', level=0)
    output_channel.loc[:, 'Ratio log'] = np.log(output_channel.loc[:, 2050] / output_channel.loc[:, 2020])

    output_CO2.loc[:, 'Delta CO2 emissions'] = output_CO2.loc[:, 2050] - output_CO2.loc[:, 2020]
    output_CO2.loc[:, 'log Delta CO2 emissions'] = np.log(output_CO2.loc[:, 2050] / output_CO2.loc[:, 2020])

    multiindex = pd.MultiIndex.from_product([output_channel.index.get_level_values(0).unique(), output_CO2.index])
    tmp = pd.concat([output_CO2.loc[:, ['Delta CO2 emissions', 'log Delta CO2 emissions']]] * len(output_channel.index.get_level_values(0).unique()))
    tmp.index = multiindex
    output_channel = pd.concat([output_channel, tmp], axis=1)  # we concatenate CO2 emissions information along columns axis

    output_channel.loc[:, 'Total channel'] = output_channel.loc[:, 'Ratio log'] * output_channel.loc[:,'Delta CO2 emissions'] / output_channel.loc[:,'log Delta CO2 emissions']
    output_channel = output_channel.loc[:, "Total channel"].to_frame().T.stack().unstack(0).T.sum(axis=1).droplevel(1)  # we get the channels
    output_channel = output_channel[['Emission content', 'Heating intensity', 'Share', 'Insulation']]

    output = output.loc[['Surface (Million m2)']]
    output.loc[:, 'Ratio log'] = np.log(output.loc[:, 2050] / output.loc[:, 2020])

    output_CO2.loc[:, 'Ratio'] = output_CO2.loc[:, 'Delta CO2 emissions'] / output_CO2.loc[:, 'log Delta CO2 emissions']

    channel_m2 = output.loc['Surface (Million m2)', 'Ratio log'] * output_CO2.sum(axis=0)['Ratio']
    output_channel = pd.concat([output_channel, pd.Series(data=[channel_m2], index=['Surface'])])

    output_channel = output_channel.reindex(['Surface', 'Insulation', 'Share', 'Heating intensity', 'Emission content'])

    return output_channel, output_CO2

if __name__ == '__main__':
    # path_cop_behrang = Path("eoles") / "inputs" / "hourly_profiles" / "hp_cop.csv"
    # hp_cop_behrang = get_pandas(path_cop_behrang, lambda x: pd.read_csv(x, index_col=0, header=0))
    # hp_cop_new = calculate_hp_cop(climate=2006)
    # hp_cop_new.to_csv(Path("eoles") / "inputs" / "hp_cop_2006.csv")

    list_path = {'Centralized': "outputs/1015_optim_pricefeedback/1015_045146_centralized_insulation_S3_N1_pricefeedback",
                  'Uniform': "outputs/1015_optim_pricefeedback/1015_144041_uniform_S3_N1_pricefeedback",
                  'DR': "outputs/1015_optim_pricefeedback/1015_140727_DR_S3_N1_pricefeedback",
                  'Proportiona': "outputs/1015_optim_pricefeedback/1105_153045_proportional_S3_N1_pricefeedback"}
    list_path = {'Ambitious': "outputs/20231210/1211004316_biogasS3_capacityN1_demandReference_profileReference_policyambitious"}
    list_path = {'Ban': "outputs/20231210/1210235222_biogasS3_capacityN1_demandReference_profileReference_policyban"}
    for scenario, path in zip(list_path.keys(), list_path.values()):
        with open(os.path.join(path, 'coupling_results.pkl'), "rb") as file:
            output = load(file)
            output_resirf = output["Output global ResIRF ()"]
            carbon_content = output["Carbon content (gC02/kWh)"]
            efficiency = pd.read_csv('inputs/technology_characteristics/efficiency_resirf.csv', index_col=0, header=None).squeeze()

        result, emissions = ldmi_method_resirf(output_resirf, carbon_content)
        plot_ldmi_method(result, emissions, 2020, 2050, colors=resources_data['colors_coupling'], rotation=0,
                         title=None,
                         save=f"outputs/20231210/ldmi_{scenario}.png",
                         # save=None,
                         format_y=lambda y, _: '{:.0f} MtCO2'.format(y)
                         )

    # path = "outputs/1016_policies_exogenous_cc_pricefeedback_hcDPE/1013_201614_S2p_N1_ref_cc_pricefeedback_hcDPE"
    # with open(os.path.join(path, 'coupling_results.pkl'), "rb") as file:
    #     output = load(file)
    #     output_resirf = output["Output global ResIRF ()"]
    #     carbon_content = output["Carbon content (gC02/kWh)"]
    #     efficiency = pd.read_csv('inputs/technology_characteristics/efficiency_resirf.csv', index_col=0, header=None).squeeze()
    #
    # output_channel1, output_CO21 = ldmi_method(output_resirf, efficiency, carbon_content)
    # output_channel2, output_CO22 = ldmi_method_2(output_resirf, efficiency, carbon_content)
    #
    # plot_ldmi_method(output_channel1, output_CO21, 2020, 2050, colors=resources_data['colors_coupling'], rotation=0, save=None,
    #                  title="LDMI method - Reference scenario")
    # plot_ldmi_method(output_channel2, output_CO22, 2020, 2050, colors=resources_data['colors_coupling'], rotation=0,
    #                  save=None, title="LDMI method 2 - Reference scenario")
    #
    # path = "outputs/1016_policies_exogenous_cc_pricefeedback_hcDPE/1014_013733_S2p_N1_ban_cc_pricefeedback_hcDPE"
    # with open(os.path.join(path, 'coupling_results.pkl'), "rb") as file:
    #     output = load(file)
    #     output_resirf = output["Output global ResIRF ()"]
    #     carbon_content = output["Carbon content (gC02/kWh)"]
    #     efficiency = pd.read_csv('inputs/technology_characteristics/efficiency_resirf.csv', index_col=0, header=None).squeeze()
    #
    # output_channel3, output_CO23 = ldmi_method(output_resirf, efficiency, carbon_content)
    # plot_ldmi_method(output_channel3, output_CO23, 2020, 2050, colors=resources_data['colors_coupling'], rotation=0,
    #                  save=None, title="LDMI method - Ban on gas boilers")
