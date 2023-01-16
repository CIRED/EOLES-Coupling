from importlib import resources
from pathlib import Path
import pandas as pd
import os
import json
from matplotlib import pyplot as plt
from pyomo.environ import value
import datetime


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


### Defining the model

def process_RTE_demand(config, year, demand, method):
    demand_noP2G_RTE_timesteps = get_pandas(config["demand_noP2G_RTE_timesteps"],
                                            lambda x: pd.read_csv(x, index_col=0).squeeze())
    demand_residential_heating_RTE_timesteps = get_pandas(config["demand_residential_heating_RTE_timesteps"],
                                                          lambda x: pd.read_csv(x, index_col=0).squeeze())
    percentage_hourly_residential_heating_profile = get_pandas(config["percentage_hourly_residential_heating_profile"],
                                                               lambda x: pd.read_csv(x, index_col=0,
                                                                                     header=None).squeeze())
    demand_noP2G_RTE = demand_noP2G_RTE_timesteps[year]  # in TWh
    demand_residential_heating = demand_residential_heating_RTE_timesteps[year]  # in TWh

    adjust_demand = (demand_noP2G_RTE * 1e3 - 580 * 1e3) / 8760  # 580TWh is the total of the profile we use as basis for electricity hourly demand (from RTE)
    demand_elec_RTE_noP2G = demand + adjust_demand  # we adjust demand profile to obtain the correct total amount of demand based on RTE projections without P2G

    hourly_residential_heating_RTE = create_hourly_residential_demand_profile(demand_residential_heating * 1e3,
                                                                              method=method)  # TODO: a changer a priori, ce n'est plus le bon profil

    demand_elec_RTE_no_residential_heating = demand_elec_RTE_noP2G - hourly_residential_heating_RTE  # we remove residential electric demand
    return demand_elec_RTE_no_residential_heating


def calculate_annuities_capex(miscellaneous, capex, construction_time, lifetime):
    """Calculate annuities for energy technologies and renovation technologies based on capex data."""
    annuities = construction_time.copy()
    for i in annuities.index:
        annuities.at[i] = miscellaneous["discount_rate"] * capex[i] * (
                miscellaneous["discount_rate"] * construction_time[i] + 1) / (
                                  1 - (1 + miscellaneous["discount_rate"]) ** (-lifetime[i]))

    # renovation_annuities = linearized_renovation_costs.copy()
    # for i in linearized_renovation_costs.index:
    #     renovation_annuities.at[i] = miscellaneous["discount_rate"] * linearized_renovation_costs[i] * (
    #             miscellaneous["discount_rate"] * miscellaneous["construction_time_renov"] + 1) / (
    #                                          1 - (1 + miscellaneous["discount_rate"]) ** (
    #                                      -miscellaneous["lifetime_renov"]))
    # annuities = pd.concat([annuities, renovation_annuities])  # we add renovation annuities to other annuities
    return annuities


def calculate_annuities_storage_capex(miscellaneous, storage_capex, construction_time, lifetime):
    """Calculate annuities for storage technologies based on capex data."""
    storage_annuities = storage_capex.copy()
    for i in storage_annuities.index:
        storage_annuities.at[i] = miscellaneous["discount_rate"] * storage_capex[i] * (
                miscellaneous["discount_rate"] * construction_time[i] + 1) / (
                                          1 - (1 + miscellaneous["discount_rate"]) ** (-lifetime[i]))
    return storage_annuities


def calculate_annuities_renovation(linearized_renovation_costs, miscellaneous):
    """Be careful to units. Renovation costs are initially expressed in 1e9 € contrary to the rest of the costs !!"""
    renovation_annuities = linearized_renovation_costs.copy()
    for archetype in linearized_renovation_costs.index:
        renovation_annuities.at[archetype] = miscellaneous["discount_rate"] * linearized_renovation_costs[
            archetype] * 1e3 * (
                                                     miscellaneous["discount_rate"] * miscellaneous[
                                                 "construction_time_renov"] + 1) / (
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


def update_ngas_cost(vOM_init, scc, emission_rate=0.2295):
    """Add emission cost related to social cost of carbon to the natural gas vOM cost.
    :param vOM_init: float
        Initial vOM in M€/GWh
    :param scc: int
        €/tCO2
    :param emission_rate: float
        tCO2/MWh

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

def get_technical_cost(model, objective, scc):
    """Returns technical cost (social cost without CO2 emissions-related cost"""
    gene_ngas = sum(value(model.gene["natural_gas", hour]) for hour in model.h) / 1000  # TWh
    net_emissions = gene_ngas * 0.2295
    technical_cost = objective - net_emissions * scc / 1000
    return technical_cost, net_emissions


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


def extract_hourly_generation(model, elec_demand, CH4_demand, H2_demand, heat_demand=None):
    """Extracts hourly defined data, including demand, generation and storage"""
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

    for tec in list_tec:
        hourly_generation[tec] = value(model.gene[tec, :])  # GWh
    for str, str_in in zip(list(model.str), list_storage_in):
        hourly_generation[str_in] = value(model.storage[str, :])  # GWh
    for str, str_charge in zip(list(model.str), list_storage_charge):
        hourly_generation[str_charge] = value(model.stored[str, :])  # GWh
    return hourly_generation  # GWh


def extract_spot_price(model, nb_hours):
    """Extracts spot price"""
    spot_price = pd.DataFrame({"hour": range(nb_hours),
                               "spot_price": [- 1e6 * model.dual[model.electricity_adequacy_constraint[h]] for h in
                                              model.h]})
    return spot_price


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


def extract_heat_gene(model, nb_years):
    """Extracts yearly heat generation per technology in TWh"""
    list_tec = list(model.heat)
    heat_generation = pd.Series(index=list_tec, dtype=float)

    for tec in list_tec:
        heat_generation[tec] = sum(value(model.gene[tec, hour]) for hour in model.h) / 1000 / nb_years  # TWh
    return heat_generation


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
    Returns the LCOE coming from newly invested capacities and energy capacities
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


def annualized_costs_investment_historical(existing_capa_historical_y, annuity_fOM_historical,
                                           existing_energy_capacity_historical_y, storage_annuity_historical):
    """Same as previous function but for historical LCOE"""
    costs_capacity_historical = pd.concat([existing_capa_historical_y, annuity_fOM_historical], axis=1, ignore_index=True)  # we only include nonzero historical capacities
    costs_capacity_historical = costs_capacity_historical.rename(columns={0: 'capacity_historical', 1: 'annuity_fOM'}).fillna(0)
    costs_capacity_historical["annualized_costs"] = costs_capacity_historical["capacity_historical"] * costs_capacity_historical["annuity_fOM"]

    costs_energy_capacity_historical = pd.concat([existing_energy_capacity_historical_y, storage_annuity_historical], axis=1, ignore_index=True)  # we only include nonzero historical capacities
    costs_energy_capacity_historical = costs_energy_capacity_historical.rename(columns={0: 'energy_capacity_historical', 1: 'storage_annuity'}).fillna(0)
    costs_energy_capacity_historical["annualized_costs"] = costs_energy_capacity_historical["energy_capacity_historical"] * costs_energy_capacity_historical["storage_annuity"]
    return costs_capacity_historical[["annualized_costs"]], costs_energy_capacity_historical[["annualized_costs"]]


def process_annualized_costs_per_vector(annualized_costs_capacity, annualized_costs_energy_capacity):
    """Calculates annualized costs related to investment for the different vectors (namely, electricity, methane and hydrogen)"""
    elec_balance = ["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "nuclear", "phs",
     "battery1", "battery4", "ocgt", "ccgt", "h2_ccgt"]
    elec_str = ["phs", "battery1", "battery4"]

    CH4_balance = ["methanization", "pyrogazification", "natural_gas", "methanation", "methane"]
    CH4_str = ["methane"]

    H2_balance = ["electrolysis", "hydrogen"]
    H2_str = ["hydrogen"]

    annualized_costs_elec = annualized_costs_capacity[elec_balance].sum() + annualized_costs_energy_capacity[elec_str].sum()  # includes annuity, fOM and storage annuity
    annualized_costs_CH4 = annualized_costs_capacity[CH4_balance].sum() + annualized_costs_energy_capacity[CH4_str].sum()
    annualized_costs_H2 = annualized_costs_capacity[H2_balance].sum() + annualized_costs_energy_capacity[H2_str].sum()
    return annualized_costs_elec, annualized_costs_CH4, annualized_costs_H2


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
    Saves the outputs of the model
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
    variables_to_read = ["summary", "hourly_generation", "capacities", "energy_capacity", "supply_elec", "use_elec"]
    o = dict()
    for variable in variables_to_read:
        path_to_read = os.path.join(folder, f"{variable}.csv")
        df = pd.read_csv(path_to_read, index_col=0)
        df = df.rename(columns={'0': variable})
        o[variable] = df
    return o


def format_ax(ax, y_label=None, title=None, y_max=None):
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


def plot_capacities(df, y_max=None):
    fig, ax = plt.subplots(1, 1)
    df.plot.bar(ax=ax)
    ax = format_ax(ax, y_label="Capacity (GW)", title="Capacities", y_max=y_max)
    plt.show()


def plot_generation(df):
    fig, ax = plt.subplots(1, 1)
    df.plot.pie(ax=ax)


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
    path_weather = Path("eoles") / "inputs" / "ninja_weather_country_FR_merra-2_population_weighted.csv"
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


def load_evolution_data():
    """Load necessary data for the social planner trajectory"""
    # Load historical data
    existing_capacity_historical = get_pandas("eoles/inputs/historical_data/existing_capacity_historical.csv",
                                              lambda x: pd.read_csv(x, index_col=0))  # GW
    existing_charging_capacity_historical = get_pandas("eoles/inputs/historical_data/existing_charging_capacity_historical.csv",
                                                       lambda x: pd.read_csv(x, index_col=0))  # GW
    existing_energy_capacity_historical = get_pandas("eoles/inputs/historical_data/existing_energy_capacity_historical.csv",
                                                     lambda x: pd.read_csv(x, index_col=0))  # GW
    maximum_capacity_evolution = get_pandas("eoles/inputs/technology_potential/maximum_capacity_evolution.csv",
                                            lambda x: pd.read_csv(x, index_col=0))  # GW

    annuity_fOM_historical = get_pandas("eoles/inputs/historical_data/annuity_fOM_historical.csv",
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
           annuity_fOM_historical, storage_annuity_historical


if __name__ == '__main__':
    path_cop_behrang = Path("eoles") / "inputs" / "hp_cop.csv"
    hp_cop_behrang = get_pandas(path_cop_behrang, lambda x: pd.read_csv(x, index_col=0, header=0))
    hp_cop_new = calculate_hp_cop(climate=2006)
    # hp_cop_new.to_csv(Path("eoles") / "inputs" / "hp_cop_2006.csv")
