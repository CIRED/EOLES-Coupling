import os.path
from copy import deepcopy
from project.coupling import ini_res_irf, simu_res_irf
from project.building import AgentBuildings
from project.input.param import generic_input
from project.write_output import plot_scenario

from GPyOpt.methods import BayesianOptimization
import numpy as np
import logging
import pandas as pd

import datetime
from pickle import dump, load
import json

from eoles.model_resirf_coupling import ModelEOLES
from eoles.utils import get_config, get_pandas, calculate_annuities_resirf
import eoles.utils

LOG_FORMATTER = '%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# consoler handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
logger.addHandler(console_handler)

HOURLY_PROFILE_METHOD = "valentin"
# LIST_REPLACEMENT_HEATER = ['Replacement heater Electricity-Heat pump water (Thousand households)',
#                            'Replacement heater Electricity-Heat pump air (Thousand households)',
#                            'Replacement heater Electricity-Performance boiler (Thousand households)',
#                            'Replacement heater Natural gas-Performance boiler (Thousand households)',
#                            'Replacement heater Oil fuel-Performance boiler (Thousand households)',
#                            'Replacement heater Wood fuel-Performance boiler (Thousand households)']

# LIST_STOCK_HEATER = ['Stock Electricity-Heat pump air (Thousand households)',
#                      'Stock Electricity-Heat pump water (Thousand households)',
#                      'Stock Electricity-Performance boiler (Thousand households)',
#                      'Stock Natural gas-Performance boiler (Thousand households)',
#                      'Stock Natural gas-Standard boiler (Thousand households)',
#                      'Stock Oil fuel-Performance boiler (Thousand households)',
#                      'Stock Oil fuel-Standard boiler (Thousand households)',
#                      'Stock Wood fuel-Performance boiler (Thousand households)',
#                      'Stock Wood fuel-Standard boiler (Thousand households)']

LIST_STOCK_HEATER = ["Stock Direct electric (Million)",
                     "Stock Heat pump (Million)",
                     "Stock Oil fuel (Million)",
                     "Stock Wood fuel (Million)",
                     "Stock Natural gas (Million)"]

LIST_REPLACEMENT_HEATER = ['Switch Electricity-Heat pump water (Thousand households)',
                           'Switch Electricity-Heat pump air (Thousand households)',
                           'Switch Electricity-Performance boiler (Thousand households)',
                           'Switch Natural gas-Performance boiler (Thousand households)',
                           'Switch Oil fuel-Performance boiler (Thousand households)',
                           'Switch Wood fuel-Performance boiler (Thousand households)']


# Functions in case we need to use pickle to copy the Buildings object
from pathlib import Path
def copy_building(building: AgentBuildings):
    path = Path("/tmp")  # absolute path
    date = datetime.datetime.now().strftime("%m%d%H%M%S%f.pkl")
    filename = path / date
    with open(filename, "wb") as file:
        dump(building, file)
    return filename


def load_building(filename: Path):
    assert filename.exists()
    with open(filename, "rb") as file:
        building = load(file)

    return building


def resirf_eoles_coupling_static(subvention, buildings, energy_prices, taxes, cost_heater, cost_insulation, demolition_rate, flow_built,
                                 post_inputs, policies_heater, policies_insulation,
                                 start_year_resirf, timestep_resirf, config_eoles,
                                 year_eoles, anticipated_year_eoles, scc, hourly_gas_exogeneous,
                                 existing_capacity, existing_charging_capacity, existing_energy_capacity,
                                 maximum_capacity, method_hourly_profile,
                                 scenario_cost, existing_annualized_costs_elec, existing_annualized_costs_CH4,
                                 existing_annualized_costs_H2, lifetime_renov=50, lifetime_heater=20,
                                 discount_rate=0.045, sub_design=None, health=True, carbon_constraint=False,
                                 rebound=True, technical_progress=None, financing_cost=None, premature_replacement=None):
    """

    :param subvention: 2 dimensional numpy array
        Attention to the shape of the array necessary for optimization !!
    :return:
    """

    # example one year simulation
    endyear_resirf = start_year_resirf + timestep_resirf

    sub_heater, sub_insulation = float(subvention[0, 0]), float(subvention[0, 1])
    # print(f'Subvention: {sub_heater}, {sub_insulation}')

    buildings_copy = deepcopy(buildings)
    energy_prices_copy, taxes_copy, cost_heater_copy, cost_insulation_copy, demolition_rate_copy = deepcopy(energy_prices), deepcopy(taxes), deepcopy(cost_heater), deepcopy(cost_insulation), deepcopy(demolition_rate)
    flow_built_copy, post_inputs_copy, policies_heater_copy, policies_insulation_copy = deepcopy(flow_built), deepcopy(post_inputs), deepcopy(policies_heater), deepcopy(policies_insulation)
    technical_progress_copy, financing_cost_copy, premature_replacement_copy = deepcopy(technical_progress), deepcopy(financing_cost), deepcopy(premature_replacement)

    # simulation between start and end - flow output represents annual values for year "end"

    output, stock, heating_consumption = simu_res_irf(buildings=buildings_copy, sub_heater=sub_heater, sub_insulation=sub_insulation,
                                                      start=start_year_resirf, end=endyear_resirf, energy_prices=energy_prices_copy,
                                                      taxes=taxes_copy, cost_heater=cost_heater_copy, cost_insulation=cost_insulation_copy,
                                                      lifetime_heater=lifetime_heater, demolition_rate=demolition_rate_copy, flow_built=flow_built_copy,
                                                      post_inputs=post_inputs_copy, policies_heater=policies_heater_copy,
                                                      policies_insulation=policies_insulation_copy,
                                                      sub_design=sub_design, financing_cost=financing_cost_copy, climate=2006, smooth=False, efficiency_hour=True,
                                                      output_consumption=True, full_output=False, rebound=rebound,
                                                      technical_progress=technical_progress_copy,
                                                      premature_replacement=premature_replacement_copy)

    # heating_consumption = output[endyear_resirf - 1]['Consumption (kWh/h)']
    heating_consumption = heating_consumption.sort_index(ascending=True)

    hourly_heat_elec = heating_consumption.loc["Electricity"] * 1e-6  # GWh
    hourly_heat_gas = heating_consumption.loc["Natural gas"] * 1e-6  # GWh
    hourly_heat_elec = hourly_heat_elec.reset_index().drop(columns=["index"]).squeeze()
    hourly_heat_gas = hourly_heat_gas.reset_index().drop(columns=["index"]).squeeze()
    hourly_heat_gas = hourly_heat_gas + hourly_gas_exogeneous  # we add exogeneous gas demand

    oil_consumption = heating_consumption.sum(axis=1)["Oil fuel"] * 1e-6  # GWh
    wood_consumption = heating_consumption.sum(axis=1)["Wood fuel"] * 1e-6  # GWh
    # print(oil_consumption, wood_consumption, heating_consumption.sum(axis=1)["Electricity"] * 1e-6, heating_consumption.sum(axis=1)["Natural gas"] * 1e-6)

    investment_heater_cost = output.loc["Investment heater WT (Billion euro)"].sum()  # 1e9 €
    investment_insulation_cost = output.loc["Investment insulation WT (Billion euro)"].sum()  # 1e9 €
    health_cost = output.loc["Health cost (Billion euro)"][endyear_resirf-1]  # 1e9 €  # TODO: a verifier. On ne prend que la dernière valeur pour avoir seulement le flux de coût de santé

    # TODO: attention, certains coûts sont peut-être déjà la somme de coûts annuels !!

    annuity_investment_cost = calculate_annuities_resirf(investment_heater_cost, lifetime=lifetime_heater, discount_rate=discount_rate) + \
                              calculate_annuities_resirf(investment_insulation_cost, lifetime=lifetime_renov, discount_rate=discount_rate)
    if health:
        annuity_health_cost = health_cost
    else:
        annuity_health_cost = 0
    # print(annuity_health_cost, annuity_investment_cost)

    m_eoles = ModelEOLES(name="trajectory", config=config_eoles, path="eoles/outputs", logger=logger, nb_years=1,
                         hourly_heat_elec=hourly_heat_elec, hourly_heat_gas=hourly_heat_gas, wood_consumption=wood_consumption,
                         oil_consumption=oil_consumption,
                         existing_capacity=existing_capacity, existing_charging_capacity=existing_charging_capacity,
                         existing_energy_capacity=existing_energy_capacity, maximum_capacity=maximum_capacity,
                         method_hourly_profile=method_hourly_profile,
                         anticipated_social_cost_of_carbon=scc, year=year_eoles, anticipated_year=anticipated_year_eoles,
                         scenario_cost=scenario_cost, existing_annualized_costs_elec=existing_annualized_costs_elec,
                         existing_annualized_costs_CH4=existing_annualized_costs_CH4,
                         existing_annualized_costs_H2=existing_annualized_costs_H2, carbon_constraint=carbon_constraint,
                         discount_rate=discount_rate)
    m_eoles.build_model()
    solver_results, status, termination_condition = m_eoles.solve(solver_name="gurobi")

    if termination_condition == "infeasibleOrUnbounded":
        logger.info("Infeasible problem")

    objective = m_eoles.objective
    # print(objective)

    # TODO: attention, à changer si on considère plusieurs années météo dans EOLES
    objective += annuity_health_cost
    objective += annuity_investment_cost
    return np.array([objective])  # return an array


def optimize_blackbox_resirf_eoles_coupling(buildings, energy_prices, taxes, cost_heater, cost_insulation, demolition_rate, flow_built,
                                            post_inputs, policies_heater, policies_insulation,
                                            start_year_resirf, timestep_resirf, config_eoles,
                                            year_eoles, anticipated_year_eoles, scc, hourly_gas_exogeneous,
                                            existing_capacity, existing_charging_capacity, existing_energy_capacity,
                                            maximum_capacity, method_hourly_profile,
                                            scenario_cost, existing_annualized_costs_elec,
                                            existing_annualized_costs_CH4,
                                            existing_annualized_costs_H2, lifetime_renov=50, lifetime_heater=20,
                                            discount_rate=0.045,
                                            max_iter=20, initial_design_numdata=3, grid_initialize=False, acquisition_jitter=0.01,
                                            normalize_Y=True, plot=False,
                                            fix_sub_heater=False, fix_sub_insulation=False,
                                            sub_design=None, health=True, carbon_constraint=False,
                                            rebound=True, technical_progress=None, financing_cost=None,
                                            premature_replacement=None):
    """
    Finds optimal subsidies by a blackbox optimization process, relying on bayesian optimization and gaussian processes.
    :param lifetime_renov: int
        Lifetime to obtain annuities
    :param lifetime_heater: int
        Lifetime to obtain annuities
    :param max_iter: int
        Number of iterations of the blackbox optimiztaion
    :param initial_design_numdata: int
        Number of initial points to estimate before starting the blackbox optimization process
    :param grid_initialize: bool
        Whether to specify manually the points to calculate the function before starting the blackbox optimization process
    :param acquisition_jitter: float
        This param controls how much exploitation versus exploration the blackbox algorithm will do
    :param plot: bool
        Whether to plot the result
    :param fix_sub_heater: bool
        Whether we want to fix heater subsidy to 0
    :param fix_sub_insulation: bool
        Whether we want to fix insulation subsidy to 0
    :return:
    """
    assert (not fix_sub_insulation) or (not fix_sub_heater), "It is not possible to fix both sub heater and sub insulation."
    if fix_sub_heater:
        bounds2d = [{'name': 'sub_heater', 'type': 'continuous', 'domain': (0, 0)},
                    {'name': 'sub_insulation', 'type': 'continuous', 'domain': (0, 1)}]
    elif fix_sub_insulation:
        bounds2d = [{'name': 'sub_heater', 'type': 'continuous', 'domain': (0, 1)},
                    {'name': 'sub_insulation', 'type': 'continuous', 'domain': (0, 0)}]
    else:
        bounds2d = [{'name': 'sub_heater', 'type': 'continuous', 'domain': (0, 1)},
                    {'name': 'sub_insulation', 'type': 'continuous', 'domain': (0, 1)}]

    if grid_initialize:
        if not fix_sub_heater and not fix_sub_insulation:
            X_init = np.array([[0.0, 0.0],
                               [0.2, 0.0],
                               [0.5, 0.0],
                               [0.8, 0.0],
                               [1.0, 0.0],
                               [0.0, 0.5],
                               [0.2, 0.5],
                               [0.5, 0.5],
                               [0.8, 0.5],
                               [1.0, 0.5],
                               [0.0, 0.9],
                               [0.2, 0.9],
                               [0.5, 0.9],
                               [0.8, 0.9],
                               [1.0, 0.9]
                               ])
        else:
            X_init = None
    else:
        X_init = None
    optimizer = BayesianOptimization(
        f=lambda x: resirf_eoles_coupling_static(x, buildings=buildings, energy_prices=energy_prices,
                                                 taxes=taxes, cost_heater=cost_heater,
                                                 cost_insulation=cost_insulation, demolition_rate=demolition_rate, flow_built=flow_built,
                                                 post_inputs=post_inputs, policies_heater=policies_heater,
                                                 policies_insulation=policies_insulation, start_year_resirf=start_year_resirf,
                                                 timestep_resirf=timestep_resirf,
                                                 config_eoles=config_eoles, year_eoles=year_eoles,
                                                 anticipated_year_eoles=anticipated_year_eoles,
                                                 scc=scc, hourly_gas_exogeneous=hourly_gas_exogeneous,
                                                 existing_capacity=existing_capacity,
                                                 existing_charging_capacity=existing_charging_capacity,
                                                 existing_energy_capacity=existing_energy_capacity,
                                                 maximum_capacity=maximum_capacity,
                                                 method_hourly_profile=method_hourly_profile,
                                                 scenario_cost=scenario_cost,
                                                 existing_annualized_costs_elec=existing_annualized_costs_elec,
                                                 existing_annualized_costs_CH4=existing_annualized_costs_CH4,
                                                 existing_annualized_costs_H2=existing_annualized_costs_H2,
                                                 lifetime_renov=lifetime_renov, lifetime_heater=lifetime_heater,
                                                 discount_rate=discount_rate, sub_design=sub_design,
                                                 health=health, carbon_constraint=carbon_constraint,
                                                 rebound=rebound, technical_progress=technical_progress,
                                                 financing_cost=financing_cost, premature_replacement=premature_replacement),
        domain=bounds2d,
        model_type='GP',  # gaussian process
        # kernel=kernel,
        acquisition_type='EI',  # expected improvement algorithm
        acquisition_jitter=acquisition_jitter,
        noise_var=0,  # no noise in surrogate function evaluation
        exact_feval=True,  # no noise in evaluations
        normalize_Y=normalize_Y,  # not sure what is the best option to use here
        maximize=False,
        verbosity=True,
        # evaluator_type='local_penalization',
        # num_cores=2,
        initial_design_numdata=initial_design_numdata,  # number of initial points before starting optimization
        initial_design_type="latin",
        X=X_init)
    optimizer.run_optimization(max_iter=max_iter)

    if plot:
        optimizer.plot_acquisition()
    return optimizer, optimizer.x_opt


def resirf_eoles_coupling_greenfield(buildings, energy_prices, taxes, cost_heater, cost_insulation, demolition_rate, flow_built,
                                  post_inputs, policies_heater, policies_insulation,
                                  scc, scenario_cost, config_eoles, config_coupling,
                                  add_CH4_demand=False,
                                  lifetime_renov=50, lifetime_heater=20,
                                  technical_progress=None, financing_cost=None, premature_replacement=None,
                                  optimization=True, list_sub_heater=None, list_sub_insulation=None,
                                     acquisition_jitter=0.01, grid_initialize=False, normalize_Y=True):
    # importing evolution of historical capacity and expected evolution of demand
    existing_capacity_historical, existing_charging_capacity_historical, existing_energy_capacity_historical, \
    maximum_capacity_evolution, heating_gas_demand_RTE_timesteps, ECS_gas_demand_RTE_timesteps, capex_annuity_fOM_historical, \
    capex_annuity_historical, storage_annuity_historical = eoles.utils.load_evolution_data(config=config_eoles, greenfield=True)

    existing_capacity_historical = existing_capacity_historical.drop(
        ["heat_pump", "resistive", "gas_boiler", "fuel_boiler", "wood_boiler"], axis=0)

    # Initialize results dataframes

    reindex_primary_prod = ["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "nuclear",
                            "methanization", "pyrogazification", "natural_gas"]

    index_conversion_prod = ["ocgt", "ccgt", "methanation", "electrolysis"]

    capacity_df = pd.DataFrame(index=existing_capacity_historical.index, dtype=float)
    new_capacity_df = pd.DataFrame(index=existing_capacity_historical.index, dtype=float)
    generation_df = pd.DataFrame(index=existing_capacity_historical.index, dtype=float)
    primary_generation_df = pd.DataFrame(index=reindex_primary_prod, dtype=float)
    conversion_generation_df = pd.DataFrame(index=index_conversion_prod, dtype=float)
    charging_capacity_df = pd.DataFrame(index=existing_charging_capacity_historical.index, dtype=float)
    energy_capacity_df = pd.DataFrame(index=existing_energy_capacity_historical.index, dtype=float)
    spot_price_df = pd.DataFrame(dtype=float)
    peak_electricity_load_df, peak_heat_load_df = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)
    hourly_generation_2050 = pd.DataFrame()

    weighted_average_elec_price, weighted_average_CH4_price, weighted_average_H2_price = [], [], []
    list_lcoe_elec, list_lcoe_elec_volume, list_lcoe_elec_value, list_lcoe_CH4_value, list_lcoe_CH4_volume, list_lcoe_CH4_volume_noSCC = [], [], [], [], [], []
    list_emissions = []
    list_electricity_price_ht, list_transport_distribution_lcoe = [], []
    list_elec_annualized, list_transport_distrib_annualized, list_insulation_annualized, list_heater_annualized, list_healthcost_annualized = [], [], [], [], []

    annualized_new_investment_df, annualized_new_energy_capacity_df, functionment_costs_df = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)  # used to estimate costs
    annualized_historical_capacity_df, annualized_historical_energy_capacity_df = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)

    annualized_costs_new_capacity = pd.DataFrame(0, index=existing_capacity_historical.index,
                                                 columns=["annualized_costs"], dtype=float)
    annualized_costs_new_energy_capacity = pd.DataFrame(0, index=existing_energy_capacity_historical.index,
                                                        columns=["annualized_costs"], dtype=float)

    if optimization:
        list_sub_heater, list_sub_insulation = [], []
    list_cost_rebound = []
    list_electricity_consumption, list_gas_consumption, list_wood_consumption, list_oil_consumption = [], [], [], []
    resirf_consumption_yearly_df = pd.DataFrame(dtype=float)
    resirf_costs_df = pd.DataFrame(dtype=float)
    resirf_consumption_saving_df = pd.DataFrame(dtype=float)
    replacement_heater_df = pd.DataFrame(index=LIST_REPLACEMENT_HEATER, dtype=float)
    stock_heater_df = pd.DataFrame(index=LIST_STOCK_HEATER, dtype=float)
    investment_cost_eff_df = pd.DataFrame(dtype=float)

    output_global_ResIRF, stock_global_ResIRF = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)
    list_global_annualized_costs = []
    dict_optimizer = {}

    # Run ResIRF for the first 5 years, with current policies and no additional subsidy
    output_opt, stock_opt, heating_consumption = simu_res_irf(buildings=buildings, sub_heater=None,
                                                              sub_insulation=None,
                                                              start=2020,
                                                              end=2025, energy_prices=energy_prices,
                                                              taxes=taxes,
                                                              cost_heater=cost_heater, cost_insulation=cost_insulation,
                                                              lifetime_heater=lifetime_heater, flow_built=flow_built,
                                                              post_inputs=post_inputs, policies_heater=policies_heater,
                                                              policies_insulation=policies_insulation,
                                                              climate=2006, smooth=False, efficiency_hour=True,
                                                              demolition_rate=demolition_rate,
                                                              output_consumption=True,
                                                              full_output=True,
                                                              sub_design=config_coupling["sub_design"],
                                                              rebound=config_coupling["rebound"],
                                                              technical_progress=technical_progress,
                                                              financing_cost=financing_cost, premature_replacement=premature_replacement)
    # we add initial values to observe what happens
    output_global_ResIRF = pd.concat([output_global_ResIRF, output_opt], axis=1)
    stock_global_ResIRF = pd.concat([stock_global_ResIRF, stock_opt], axis=1)

    # Parameters for greenfield
    year_eoles, anticipated_year_eoles = 2050, 2050
    start_year_resirf, timestep_resirf = 2025, 25
    list_anticipated_year = [2050]

    #### Get existing and maximum capacities
    existing_capacity = existing_capacity_historical[
        [str(anticipated_year_eoles)]].squeeze()  # get historical capacity still installed for year of interest
    existing_charging_capacity = existing_charging_capacity_historical[
        [str(anticipated_year_eoles)]].squeeze()
    existing_energy_capacity = existing_energy_capacity_historical[
        [str(anticipated_year_eoles)]].squeeze()

    new_maximum_capacity = maximum_capacity_evolution[
        [str(anticipated_year_eoles)]].squeeze()  # get maximum new capacity to be built

    maximum_capacity = (
            existing_capacity + new_maximum_capacity).dropna()  # we drop nan values, which correspond to technologies without any upper bound

    #### Historical annualized costs based on historical costs
    annualized_costs_capacity_historical, annualized_costs_energy_capacity_historical = eoles.utils.annualized_costs_investment_historical(
        existing_capacity, capex_annuity_fOM_historical,
        existing_energy_capacity, storage_annuity_historical)

    annualized_costs_capacity_nofOM_historical = eoles.utils.annualized_costs_investment_historical_nofOM(
        existing_capacity, capex_annuity_historical,
        existing_energy_capacity, storage_annuity_historical)

    ### Compile total annualized investment costs from existing capacities (both historical capacities + newly built capacities before t)
    # Necessary for calculus of LCOE accounting for evolution of capacities

    existing_annualized_costs_elec, existing_annualized_costs_CH4, existing_annualized_costs_H2 = eoles.utils.process_annualized_costs_per_vector(
        annualized_costs_capacity_historical[["annualized_costs"]].squeeze(),
        annualized_costs_energy_capacity_historical[["annualized_costs"]].squeeze())
    # print(existing_annualized_costs_elec, existing_annualized_costs_CH4)

    if add_CH4_demand:
        # Create additional gas profile (tertiary heating + ECS)
        heating_gas_demand = heating_gas_demand_RTE_timesteps[anticipated_year_eoles] * 1e3  # in TWh
        ECS_gas_demand = ECS_gas_demand_RTE_timesteps[anticipated_year_eoles] * 1e3  # in TWh
        ECS_demand_hourly = ECS_gas_demand / 8760
        hourly_gas = eoles.utils.create_hourly_residential_demand_profile(total_consumption=heating_gas_demand,
                                                                          method=HOURLY_PROFILE_METHOD)  # value for gas heating demand in tertiary sector
        hourly_ECS = pd.Series(ECS_demand_hourly, index=hourly_gas.index)
        hourly_exogeneous_CH4 = hourly_gas + hourly_ECS
    else:
        # we do not add any demand
        hourly_exogeneous_CH4 = eoles.utils.create_hourly_residential_demand_profile(total_consumption=0,
                                                                                     method=HOURLY_PROFILE_METHOD)

    if optimization:
        # Find optimal subsidy
        optimizer, opt_sub = \
            optimize_blackbox_resirf_eoles_coupling(buildings, energy_prices, taxes, cost_heater, cost_insulation,
                                                    demolition_rate, flow_built, post_inputs, policies_heater,
                                                    policies_insulation,
                                                    start_year_resirf, timestep_resirf,
                                                    config_eoles, year_eoles, anticipated_year_eoles, scc,
                                                    hourly_gas_exogeneous=hourly_exogeneous_CH4,
                                                    existing_capacity=existing_capacity,
                                                    existing_charging_capacity=existing_charging_capacity,
                                                    existing_energy_capacity=existing_energy_capacity,
                                                    maximum_capacity=maximum_capacity, method_hourly_profile="valentin",
                                                    scenario_cost=scenario_cost,
                                                    existing_annualized_costs_elec=existing_annualized_costs_elec,
                                                    existing_annualized_costs_CH4=existing_annualized_costs_CH4,
                                                    existing_annualized_costs_H2=existing_annualized_costs_H2,
                                                    lifetime_renov=lifetime_renov, lifetime_heater=lifetime_heater,
                                                    discount_rate=config_coupling["discount_rate"], plot=False,
                                                    max_iter=config_coupling["max_iter"], initial_design_numdata=3,
                                                    grid_initialize=grid_initialize,
                                                    acquisition_jitter=acquisition_jitter,
                                                    normalize_Y=normalize_Y,
                                                    fix_sub_heater=config_coupling["fix_sub_heater"],
                                                    fix_sub_insulation=config_coupling["fix_sub_insulation"],
                                                    sub_design=config_coupling["sub_design"],
                                                    health=config_coupling["health"],
                                                    carbon_constraint=config_coupling["carbon_constraint"],
                                                    rebound=config_coupling["rebound"],
                                                    technical_progress=technical_progress,
                                                    financing_cost=financing_cost,
                                                    premature_replacement=premature_replacement)
        dict_optimizer.update({year_eoles: optimizer})
        opt_sub_heater, opt_sub_insulation = opt_sub[0], opt_sub[1]
        list_sub_heater.append(opt_sub_heater)
        list_sub_insulation.append(opt_sub_insulation)
    else:
        opt_sub_heater, opt_sub_insulation = list_sub_heater[0], list_sub_insulation[0]

    # Rerun ResIRF with optimal subvention parameters
    endyear_resirf = start_year_resirf + timestep_resirf

    output_opt, stock_opt, heating_consumption = simu_res_irf(buildings=buildings, sub_heater=opt_sub_heater,
                                                              sub_insulation=opt_sub_insulation,
                                                              start=start_year_resirf,
                                                              end=endyear_resirf, energy_prices=energy_prices,
                                                              taxes=taxes,
                                                              cost_heater=cost_heater,
                                                              cost_insulation=cost_insulation,
                                                              lifetime_heater=lifetime_heater,
                                                              flow_built=flow_built,
                                                              post_inputs=post_inputs,
                                                              policies_heater=policies_heater,
                                                              policies_insulation=policies_insulation,
                                                              climate=2006, smooth=False, efficiency_hour=True,
                                                              demolition_rate=demolition_rate,
                                                              output_consumption=True,
                                                              full_output=True,
                                                              sub_design=config_coupling["sub_design"],
                                                              rebound=config_coupling["rebound"],
                                                              technical_progress=technical_progress,
                                                              financing_cost=financing_cost,
                                                              premature_replacement=premature_replacement)
    output_global_ResIRF = pd.concat([output_global_ResIRF, output_opt], axis=1)
    stock_global_ResIRF = pd.concat([stock_global_ResIRF, stock_opt], axis=1)

    # heating_consumption = output_opt[endyear_resirf - 1]['Consumption (kWh/h)']  # old
    heating_consumption = heating_consumption.sort_index(ascending=True)

    hourly_heat_elec = heating_consumption.loc["Electricity"] * 1e-6  # GWh
    hourly_heat_gas = heating_consumption.loc["Natural gas"] * 1e-6  # GWh
    hourly_heat_elec = hourly_heat_elec.reset_index().drop(columns=["index"]).squeeze()
    hourly_heat_gas = hourly_heat_gas.reset_index().drop(columns=["index"]).squeeze()
    hourly_heat_gas = hourly_heat_gas + hourly_exogeneous_CH4  # we add exogeneous gas demand

    electricity_consumption = heating_consumption.sum(axis=1)["Electricity"] * 1e-9  # TWh
    gas_consumption = heating_consumption.sum(axis=1)["Natural gas"] * 1e-9  # TWh
    oil_consumption = heating_consumption.sum(axis=1)["Oil fuel"] * 1e-9  # TWh
    wood_consumption = heating_consumption.sum(axis=1)["Wood fuel"] * 1e-9  # TWh

    list_consumption_yearly = ["Consumption Electricity (TWh)", "Consumption Natural gas (TWh)",
                               "Consumption Oil fuel (TWh)", "Consumption Wood fuel (TWh)"]
    resirf_consumption_yearly = output_opt.loc[
        [ind for ind in output_opt.index if ind in list_consumption_yearly]]  # TWh
    resirf_consumption_yearly_df = pd.concat([resirf_consumption_yearly_df, resirf_consumption_yearly], axis=1)

    list_inv = ["Investment heater WT (Billion euro)", "Investment insulation WT (Billion euro)",
                "Subsidies heater (Billion euro)",
                "Subsidies insulation (Billion euro)", "Health cost (Billion euro)"]
    investment_resirf = output_opt.loc[[ind for ind in output_opt.index if ind in list_inv]]  # 1e9 €
    resirf_costs_df = pd.concat([resirf_costs_df, investment_resirf], axis=1)

    list_saving = ["Consumption saving heater (TWh/year)", "Consumption saving insulation (TWh/year)"]
    consumption_saving = output_opt.loc[[ind for ind in output_opt.index if ind in list_saving]]  # TWh
    resirf_consumption_saving_df = pd.concat([resirf_consumption_saving_df, consumption_saving], axis=1)

    annuity_investment_heater_cost = calculate_annuities_resirf(
        output_opt.loc["Investment heater WT (Billion euro)"].sum(),
        lifetime=lifetime_heater, discount_rate=config_coupling["discount_rate"])
    annuity_investment_insulation_cost = calculate_annuities_resirf(
        output_opt.loc["Investment insulation WT (Billion euro)"].sum(), lifetime=lifetime_renov,
        discount_rate=config_coupling["discount_rate"])

    annuity_health_cost = output_opt.loc["Health cost (Billion euro)"][endyear_resirf - 1]

    replacement_heater = output_opt.loc[[ind for ind in output_opt.index if ind in LIST_REPLACEMENT_HEATER]]
    replacement_heater_df = pd.concat([replacement_heater_df, replacement_heater], axis=1)

    stock_output = output_opt.loc[[ind for ind in output_opt.index if ind in LIST_STOCK_HEATER]]
    stock_output = stock_output[stock_output.columns[-1]]

    if 'Cost rebound (Billion euro)' in output_opt.index:
        rebound_cost = output_opt.loc['Cost rebound (Billion euro)'].sum()  # 1e9 €
        list_cost_rebound.append(rebound_cost)

    list_cost_eff = ['Investment heater / saving (euro/kWh)', 'Investment insulation / saving (euro/kWh)']
    investment_cost_eff = output_opt.loc[[ind for ind in output_opt.index if ind in list_cost_eff]]  # TWh
    investment_cost_eff_df = pd.concat([investment_cost_eff_df, investment_cost_eff], axis=1)

    list_electricity_consumption.append(electricity_consumption)
    list_gas_consumption.append(gas_consumption)
    list_wood_consumption.append(wood_consumption)
    list_oil_consumption.append(oil_consumption)

    stock_heater_df = pd.concat([stock_heater_df, stock_output.to_frame()], axis=1)
    list_insulation_annualized.append(annuity_investment_insulation_cost)
    list_heater_annualized.append(annuity_investment_heater_cost)
    list_healthcost_annualized.append(annuity_health_cost)

    # Rerun EOLES with optimal parameters
    m_eoles = ModelEOLES(name="trajectory", config=config_eoles, path="eoles/outputs", logger=logger, nb_years=1,
                         hourly_heat_elec=hourly_heat_elec, hourly_heat_gas=hourly_heat_gas,
                         wood_consumption=wood_consumption * 1e3,  # GWh
                         oil_consumption=oil_consumption * 1e3,
                         existing_capacity=existing_capacity, existing_charging_capacity=existing_charging_capacity,
                         existing_energy_capacity=existing_energy_capacity, maximum_capacity=maximum_capacity,
                         method_hourly_profile="valentin",
                         anticipated_social_cost_of_carbon=scc, actual_social_cost_of_carbon=scc,
                         year=year_eoles, anticipated_year=anticipated_year_eoles,
                         scenario_cost=scenario_cost, existing_annualized_costs_elec=existing_annualized_costs_elec,
                         existing_annualized_costs_CH4=existing_annualized_costs_CH4,
                         existing_annualized_costs_H2=existing_annualized_costs_H2,
                         carbon_constraint=config_coupling["carbon_constraint"],
                         discount_rate=config_coupling["discount_rate"])

    m_eoles.build_model()
    solver_results, status, termination_condition = m_eoles.solve(solver_name="gurobi")

    if termination_condition == "infeasibleOrUnbounded":
        logger.info("Carbon budget is violated.")
        return None, None

    ### Spot price
    spot_price = m_eoles.spot_price.rename(
        columns={"elec_spot_price": f"elec_spot_price_{anticipated_year_eoles}",
                 "CH4_spot_price": f"CH4_spot_price_{anticipated_year_eoles}"})
    spot_price_df = pd.concat([spot_price_df, spot_price], axis=1)

    peak_electricity_load = m_eoles.peak_electricity_load_info
    peak_electricity_load["year"] = anticipated_year_eoles
    peak_electricity_load_df = pd.concat([peak_electricity_load_df, peak_electricity_load], axis=0)

    peak_heat_load = m_eoles.peak_heat_load_info
    peak_heat_load["year"] = anticipated_year_eoles
    peak_heat_load_df = pd.concat([peak_heat_load_df, peak_heat_load], axis=0)

    list_emissions.append(m_eoles.emissions.sum())  # TODO: à verifier

    if anticipated_year_eoles == 2050:
        hourly_generation_2050 = m_eoles.hourly_generation

    #### Get annuity and functionment cost corresponding to each technology
    new_capacity_annualized_costs_nofOM = m_eoles.new_capacity_annualized_costs_nofOM / 1000  # 1e9 € / yr
    new_capacity_annualized_costs_nofOM = pd.concat([new_capacity_annualized_costs_nofOM,
                                                     pd.DataFrame(
                                                         index=["investment_heater", "investment_insulation"],
                                                         data={'annualized_costs': [
                                                             annuity_investment_heater_cost,
                                                             annuity_investment_insulation_cost]})], axis=0)
    annualized_new_investment_df = pd.concat([annualized_new_investment_df,
                                              new_capacity_annualized_costs_nofOM.rename(
                                                  columns={"annualized_costs": anticipated_year_eoles})],
                                             axis=1)

    new_energy_capacity_annualized_costs_nofOM = m_eoles.new_energy_capacity_annualized_costs_nofOM / 1000
    annualized_new_energy_capacity_df = pd.concat([annualized_new_energy_capacity_df,
                                                   new_energy_capacity_annualized_costs_nofOM.rename(
                                                       columns={"annualized_costs": anticipated_year_eoles})],
                                                  axis=1)

    functionment_cost = m_eoles.functionment_cost / 1000  # 1e9 € / yr
    functionment_cost = pd.concat([functionment_cost,
                                   pd.DataFrame(index=["health_costs"],
                                                data={'functionment_cost': [annuity_health_cost]})], axis=0)
    functionment_costs_df = pd.concat([functionment_costs_df, functionment_cost.rename(
        columns={"functionment_cost": anticipated_year_eoles})], axis=1)

    # Historical annuity cost
    annualized_costs_capacity_nofOM_historical = annualized_costs_capacity_nofOM_historical / 1000  # 1e9 € / yr
    annualized_costs_energy_capacity_historical = annualized_costs_energy_capacity_historical / 1000  # 1e9 € / yr
    annualized_historical_capacity_df = pd.concat([annualized_historical_capacity_df,
                                                   annualized_costs_capacity_nofOM_historical.rename(
                                                       columns={'annualized_costs': anticipated_year_eoles})],
                                                  axis=1)
    annualized_historical_energy_capacity_df = pd.concat([annualized_historical_energy_capacity_df,
                                                          annualized_costs_energy_capacity_historical.rename(
                                                              columns={
                                                                  'annualized_costs': anticipated_year_eoles})],
                                                         axis=1)

    #### Update total capacities and new capacities
    new_capacity = m_eoles.capacities - existing_capacity  # we get the newly installed capacities at year y
    new_charging_capacity = m_eoles.charging_capacity - existing_charging_capacity
    new_energy_capacity = m_eoles.energy_capacity - existing_energy_capacity
    gene_per_tec = pd.Series(m_eoles.generation_per_technology)

    # update annualized costs from new capacity built until time t included
    annualized_costs_new_capacity = pd.concat([annualized_costs_new_capacity[["annualized_costs"]],
                                               m_eoles.new_capacity_annualized_costs[
                                                   ["annualized_costs"]].rename(
                                                   columns={'annualized_costs': 'new_annualized_costs'})],
                                              axis=1)
    annualized_costs_new_capacity["annualized_costs"] = annualized_costs_new_capacity["annualized_costs"] + \
                                                        annualized_costs_new_capacity['new_annualized_costs']

    annualized_costs_new_energy_capacity = pd.concat(
        [annualized_costs_new_energy_capacity[["annualized_costs"]],
         m_eoles.new_energy_capacity_annualized_costs[
             ["annualized_costs"]].rename(
             columns={'annualized_costs': 'new_annualized_costs'})],
        axis=1)
    annualized_costs_new_energy_capacity["annualized_costs"] = annualized_costs_new_energy_capacity[
                                                                   "annualized_costs"] + \
                                                               annualized_costs_new_energy_capacity[
                                                                   'new_annualized_costs']

    capacity_df = pd.concat(
        [capacity_df, m_eoles.capacities.to_frame().rename(columns={0: anticipated_year_eoles})], axis=1)
    charging_capacity_df = pd.concat(
        [charging_capacity_df,
         m_eoles.charging_capacity.to_frame().rename(columns={0: anticipated_year_eoles})], axis=1)
    generation_df = pd.concat(
        [generation_df, gene_per_tec.to_frame().rename(columns={0: anticipated_year_eoles})], axis=1)
    energy_capacity_df = pd.concat(
        [energy_capacity_df, m_eoles.energy_capacity.to_frame().rename(columns={0: anticipated_year_eoles})],
        axis=1)
    new_capacity_df = pd.concat(
        [new_capacity_df, new_capacity.to_frame().rename(columns={0: anticipated_year_eoles})], axis=1)
    primary_generation_df = pd.concat([primary_generation_df,
                                       m_eoles.primary_generation.reindex(
                                           reindex_primary_prod).to_frame().rename(
                                           columns={0: anticipated_year_eoles})], axis=1)
    conversion_generation = pd.concat([m_eoles.CH4_to_power_generation.to_frame(),
                                       m_eoles.power_to_CH4_generation.to_frame(),
                                       m_eoles.power_to_H2_generation.to_frame()], axis=0)
    conversion_generation_df = pd.concat(
        [conversion_generation_df, conversion_generation.rename(columns={0: anticipated_year_eoles})], axis=1)

    list_elec_annualized.append(m_eoles.objective)
    list_transport_distrib_annualized.append(m_eoles.transport_distribution_cost)
    total_annualized_costs = annuity_investment_heater_cost + annuity_investment_insulation_cost + annuity_health_cost + m_eoles.objective
    list_global_annualized_costs.append(total_annualized_costs)

    weighted_average_elec_price.append(m_eoles.summary["weighted_elec_price_demand"])
    weighted_average_CH4_price.append(m_eoles.summary["weighted_CH4_price_demand"])
    weighted_average_H2_price.append(m_eoles.summary["weighted_H2_price_demand"])

    list_lcoe_elec.append(m_eoles.summary["lcoe_elec"])
    list_lcoe_elec_volume.append(m_eoles.summary["lcoe_elec_volume"])
    list_lcoe_elec_value.append(m_eoles.summary["lcoe_elec_value"])
    list_lcoe_CH4_value.append(m_eoles.summary["lcoe_CH4_value"])
    list_lcoe_CH4_volume.append(m_eoles.summary["lcoe_CH4_volume"])
    list_lcoe_CH4_volume_noSCC.append(m_eoles.summary["lcoe_CH4_volume_noSCC"])

    price_df = pd.DataFrame(
        {'Average electricity price': weighted_average_elec_price, 'Average CH4 price': weighted_average_CH4_price,
         'Average H2 price': weighted_average_H2_price,
         'LCOE electricity': list_lcoe_elec, 'LCOE electricity value': list_lcoe_elec_value,
         'LCOE electricity volume': list_lcoe_elec_volume,
         'LCOE CH4 value': list_lcoe_CH4_value, 'LCOE CH4 volume': list_lcoe_CH4_volume,
         'LCOE CH4 volume noSCC': list_lcoe_CH4_volume_noSCC},
        index=list_anticipated_year)

    annualized_system_costs_df = pd.DataFrame(
        {'Annualized electricity system costs': list_elec_annualized,
         'Annualized investment heater costs': list_heater_annualized,
         'Annualized investment insulation costs': list_insulation_annualized,
         'Annualized health costs': list_healthcost_annualized,
         "Annualized total costs": list_global_annualized_costs},
        index=list_anticipated_year
    )

    resirf_subsidies_df = pd.DataFrame({'Heater': list_sub_heater, 'Insulation': list_sub_insulation},
                                       index=list_anticipated_year)  # we keep list_year, as the subsidies are applied from y to y+5

    resirf_consumption_df = pd.DataFrame(
        {'Electricity': list_electricity_consumption, "Natural gas": list_gas_consumption,
         'Oil fuel': list_oil_consumption, 'Wood fuel': list_wood_consumption}, index=list_anticipated_year)

    output = {
        "Capacities (GW)": capacity_df,
        "New capacities (GW)": new_capacity_df,
        "Generation (TWh)": generation_df,
        "Primary generation (TWh)": primary_generation_df,
        "Conversion generation (TWh)": conversion_generation_df,
        "Charging capacity (GW)": charging_capacity_df,
        "Energy capacity (GW)": energy_capacity_df,
        "Annualized new investments (1e9€/yr)": annualized_new_investment_df,
        "Annualized costs new energy capacity (1e9€/yr)": annualized_new_energy_capacity_df,
        "Annualized costs historical capacity (1e9€/yr)": annualized_historical_capacity_df,
        "Annualized costs historical energy capacity (1e9€/yr)": annualized_historical_energy_capacity_df,
        "System functionment (1e9€/yr)": functionment_costs_df,
        "Prices (€/MWh)": price_df,
        "Emissions (MtCO2)": pd.DataFrame({"Emissions": list_emissions}, index=list_anticipated_year),
        "Peak electricity load": peak_electricity_load_df,
        "Peak heat load": peak_heat_load_df,
        "Subsidies (%)": resirf_subsidies_df,
        "ResIRF costs (Billion euro)": resirf_costs_df.T,
        "ResIRF costs eff (euro/kWh)": investment_cost_eff_df.T,
        "ResIRF consumption (TWh)": resirf_consumption_df,
        "ResIRF consumption yearly (TWh)": resirf_consumption_yearly_df.T,
        "ResIRF consumption savings (TWh)": resirf_consumption_saving_df.T,
        "ResIRF replacement heater (Thousand)": replacement_heater_df.T,
        "ResIRF stock heater (Thousand)": stock_heater_df,
        "Annualized system costs (Billion euro / year)": annualized_system_costs_df,
        'Transport and distribution costs (Billion euro / year)': pd.DataFrame(
            {"Transport and distribution": list_transport_distrib_annualized}, index=list_anticipated_year),
        "Output global ResIRF ()": output_global_ResIRF,
        "Stock global ResIRF ()": stock_global_ResIRF,
        "Spot price EOLES (€ / MWh)": spot_price_df,
        "Hourly generation 2050 (GWh)": hourly_generation_2050,
        "Energy prices (€/kWh)": energy_prices
    }

    return output, buildings, dict_optimizer


def resirf_eoles_coupling_dynamic(buildings, energy_prices, taxes, cost_heater, cost_insulation, demolition_rate, flow_built,
                                  post_inputs, policies_heater, policies_insulation,
                                  list_year, list_trajectory_scc, scenario_cost, config_eoles, config_coupling,
                                  add_CH4_demand=False, one_shot_setting=False,
                                  lifetime_renov=50, lifetime_heater=20,
                                  technical_progress=None, financing_cost=None, premature_replacement=None,
                                  anticipated_scc=False, anticipated_demand_t10=False, optimization=True,
                                  list_sub_heater=None, list_sub_insulation=None, price_feedback=False, energy_prices_ht=None,
                                  energy_taxes=None, acquisition_jitter=0.01, grid_initialize=False, normalize_Y=True):
    """Performs multistep optimization of capacities and subsidies.
    :param config_coupling: dict
        Includes a number of parametrization for configuration of the coupling
    :param lifetime_renov: int
    :param lifetime_heater: int
    :param rebound: bool
    :param technical_progress: bool
    :param financing_cost: ??
    :param anticipated_scc: bool
        If True, the social planner considers the social cost at t+5 when computing optimal subsidies and investment decisions.
    :param anticipated_demand_t10: bool
        If True, we anticipate electricity demand at t+10, instead of anticipating it at t+5.
    :param optimization: bool
        If set to True, blackbox optimization is performed. If set to False, subsidies are provided as inputs, and the model is simply simulated.
    :param list_sub_heater: list
        Provided when optimization is set to False. Provides the list of heater subsidies for the different time steps
    :param list_sub_insulation: list
        Provided when optimization is set to False. Provides the list of insulation subsidies for the different time steps
    """
    if not optimization:
        assert list_sub_heater is not None, "Parameter list_sub_heater should be provided when optimization is set to False."
        assert list_sub_insulation is not None, "Parameter list_sub_insulation should be provided when optimization is set to False."
    assert anticipated_scc in [False, "t5", "average"], "Parameter anticipated_scc is not in the allowed range of parameters."
    # INITIALIZATION OF EOLES PARAMETERS

    # importing evolution of historical capacity and expected evolution of demand
    existing_capacity_historical, existing_charging_capacity_historical, existing_energy_capacity_historical, \
    maximum_capacity_evolution, heating_gas_demand_RTE_timesteps, ECS_gas_demand_RTE_timesteps, capex_annuity_fOM_historical, \
    capex_annuity_historical, storage_annuity_historical = eoles.utils.load_evolution_data(config=config_eoles, greenfield=False)

    existing_capacity_historical = existing_capacity_historical.drop(
        ["heat_pump", "resistive", "gas_boiler", "fuel_boiler", "wood_boiler"], axis=0)

    # TODO: il faudra ajouter les capacités pour charging et discharging

    # Initialize results dataframes
    new_capacity_tot = pd.Series(0, index=existing_capacity_historical.index, dtype=float)
    new_charging_capacity_tot = pd.Series(0, index=existing_charging_capacity_historical.index, dtype=float)
    new_energy_capacity_tot = pd.Series(0, index=existing_energy_capacity_historical.index, dtype=float)

    reindex_primary_prod = ["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "nuclear",
                            "methanization", "pyrogazification", "natural_gas"]

    index_conversion_prod = ["ocgt", "ccgt", "methanation", "electrolysis"]

    capacity_df = pd.DataFrame(index=existing_capacity_historical.index, dtype=float)
    new_capacity_df = pd.DataFrame(index=existing_capacity_historical.index, dtype=float)
    generation_df = pd.DataFrame(index=existing_capacity_historical.index, dtype=float)
    primary_generation_df = pd.DataFrame(index=reindex_primary_prod, dtype=float)
    conversion_generation_df = pd.DataFrame(index=index_conversion_prod, dtype=float)
    charging_capacity_df = pd.DataFrame(index=existing_charging_capacity_historical.index, dtype=float)
    energy_capacity_df = pd.DataFrame(index=existing_energy_capacity_historical.index, dtype=float)
    spot_price_df = pd.DataFrame(dtype=float)
    peak_electricity_load_df, peak_heat_load_df = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)
    hourly_generation_2050 = pd.DataFrame()

    weighted_average_elec_price, weighted_average_CH4_price, weighted_average_H2_price = [], [], []
    list_lcoe_elec, list_lcoe_elec_volume, list_lcoe_elec_value, list_lcoe_CH4_value, list_lcoe_CH4_volume, list_lcoe_CH4_volume_noSCC = [], [], [], [], [], []
    list_emissions = []
    list_electricity_price_ht, list_transport_distribution_lcoe = [], []
    list_elec_annualized, list_transport_distrib_annualized, list_insulation_annualized, list_heater_annualized, list_healthcost_annualized = [], [], [], [], []

    annualized_new_investment_df, annualized_new_energy_capacity_df, functionment_costs_df = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)  # used to estimate costs
    annualized_historical_capacity_df, annualized_historical_energy_capacity_df = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)

    annualized_costs_new_capacity = pd.DataFrame(0, index=existing_capacity_historical.index,
                                                 columns=["annualized_costs"], dtype=float)
    annualized_costs_new_energy_capacity = pd.DataFrame(0, index=existing_energy_capacity_historical.index,
                                                        columns=["annualized_costs"], dtype=float)

    if optimization:
        list_sub_heater, list_sub_insulation = [], []
    list_cost_rebound = []
    list_electricity_consumption, list_gas_consumption, list_wood_consumption, list_oil_consumption = [], [], [], []
    resirf_consumption_yearly_df = pd.DataFrame(dtype=float)
    resirf_costs_df = pd.DataFrame(dtype=float)
    resirf_consumption_saving_df = pd.DataFrame(dtype=float)
    replacement_heater_df = pd.DataFrame(index=LIST_REPLACEMENT_HEATER, dtype=float)
    stock_heater_df = pd.DataFrame(index=LIST_STOCK_HEATER, dtype=float)
    investment_cost_eff_df = pd.DataFrame(dtype=float)

    output_global_ResIRF, stock_global_ResIRF = pd.DataFrame(dtype=float), pd.DataFrame(dtype=float)
    list_global_annualized_costs = []
    dict_optimizer = {}

    # Run ResIRF for the first 5 years, with current policies and no additional subsidy
    output_opt, stock_opt, heating_consumption = simu_res_irf(buildings=buildings, sub_heater=None,
                                                              sub_insulation=None,
                                                              start=2020,
                                                              end=2025, energy_prices=energy_prices,
                                                              taxes=taxes,
                                                              cost_heater=cost_heater, cost_insulation=cost_insulation,
                                                              lifetime_heater=lifetime_heater, flow_built=flow_built,
                                                              post_inputs=post_inputs, policies_heater=policies_heater,
                                                              policies_insulation=policies_insulation,
                                                              climate=2006, smooth=False, efficiency_hour=True,
                                                              demolition_rate=demolition_rate,
                                                              output_consumption=True,
                                                              full_output=True,
                                                              sub_design=config_coupling["sub_design"],
                                                              rebound=config_coupling["rebound"],
                                                              technical_progress=technical_progress,
                                                              financing_cost=financing_cost, premature_replacement=premature_replacement)
    # we add initial values to observe what happens
    output_global_ResIRF = pd.concat([output_global_ResIRF, output_opt], axis=1)
    stock_global_ResIRF = pd.concat([stock_global_ResIRF, stock_opt], axis=1)

    list_anticipated_year = [y+5 for y in list_year]  # we create list of years used for supply = demand
    list_scc_year = list_scc_yearly()
    for t, (y, scc) in enumerate(zip(list_year, list_trajectory_scc)):
        # IMPORTANT REMARK: here, list_trajectory_scc is already compiled so that list_trajectory_scc[t] corresponds to the SCC at the END of the period !
        print(f"Year {y}, SCC {scc}")
        if not one_shot_setting:  # classical setting
            if not anticipated_demand_t10:  # classical setting
                if y < 2050:
                    year_eoles, anticipated_year_eoles, start_year_resirf, timestep_resirf = y, y + 5, y, 5
                else:  # specific case for 2050
                    year_eoles, anticipated_year_eoles, start_year_resirf, timestep_resirf = y, y, y, 5
            else:
                if y < 2045:
                    year_eoles, anticipated_year_eoles, start_year_resirf, timestep_resirf = y, y + 10, y, 5
                else:  # specific case for 2050
                    year_eoles, anticipated_year_eoles, start_year_resirf, timestep_resirf = y, 2050, y, 5
        else:  # one shot setting
            year_eoles, anticipated_year_eoles, start_year_resirf, timestep_resirf = 2050, 2050, 2025, 5

        if anticipated_scc == "t5":  # we modify the scc to consider the scc at t+5
            if y < 2045:
                scc = list_trajectory_scc[t+1]
                actual_scc = list_trajectory_scc[t]
            else:  # 2045
                actual_scc = scc
        elif anticipated_scc == "average":  # we modify the scc to consider the average scc over time period of new system
            scc = average_scc_discounted(list_scc_year, lifetime=25, initial_time=t, discount_rate=0.032)
            actual_scc = list_trajectory_scc[t]
        else:
            actual_scc = scc  # no difference between anticipated and actual scc

        #### Get existing and maximum capacities
        existing_capa_historical_y = existing_capacity_historical[
            [str(anticipated_year_eoles)]].squeeze()  # get historical capacity still installed for year of interest
        existing_charging_capacity_historical_y = existing_charging_capacity_historical[
            [str(anticipated_year_eoles)]].squeeze()
        existing_energy_capacity_historical_y = existing_energy_capacity_historical[
            [str(anticipated_year_eoles)]].squeeze()

        new_maximum_capacity_y = maximum_capacity_evolution[
            [str(anticipated_year_eoles)]].squeeze()  # get maximum new capacity to be built

        # Existing capacities at year y
        existing_capacity = existing_capa_historical_y + new_capacity_tot  # existing capacity are equal to newly built
        # capacities over the whole time horizon before t + existing capacity (from before 2020)
        existing_charging_capacity = existing_charging_capacity_historical_y + new_charging_capacity_tot
        existing_energy_capacity = existing_energy_capacity_historical_y + new_energy_capacity_tot

        maximum_capacity = (
                existing_capacity + new_maximum_capacity_y).dropna()  # we drop nan values, which correspond to technologies without any upper bound

        #### Historical annualized costs based on historical costs
        annualized_costs_capacity_historical, annualized_costs_energy_capacity_historical = eoles.utils.annualized_costs_investment_historical(
            existing_capa_historical_y, capex_annuity_fOM_historical,
            existing_energy_capacity_historical_y, storage_annuity_historical)

        annualized_costs_capacity_nofOM_historical = eoles.utils.annualized_costs_investment_historical_nofOM(
            existing_capa_historical_y, capex_annuity_historical,
            existing_energy_capacity_historical_y, storage_annuity_historical)

        ### Compile total annualized investment costs from existing capacities (both historical capacities + newly built capacities before t)
        # Necessary for calculus of LCOE accounting for evolution of capacities
        annualized_costs_capacity = pd.concat([annualized_costs_capacity_historical.rename(columns={'annualized_costs': 'historical_annualized_costs'}),
             annualized_costs_new_capacity], axis=1)
        annualized_costs_capacity['annualized_costs'] = annualized_costs_capacity['historical_annualized_costs'] + annualized_costs_capacity['annualized_costs']
        annualized_costs_energy_capacity = pd.concat([annualized_costs_energy_capacity_historical.rename(
            columns={'annualized_costs': 'historical_annualized_costs'}), annualized_costs_new_energy_capacity], axis=1)
        annualized_costs_energy_capacity['annualized_costs'] = annualized_costs_energy_capacity['historical_annualized_costs'] + \
                                                               annualized_costs_energy_capacity['annualized_costs']

        existing_annualized_costs_elec, existing_annualized_costs_CH4, existing_annualized_costs_H2 = eoles.utils.process_annualized_costs_per_vector(
            annualized_costs_capacity[["annualized_costs"]].squeeze(),
            annualized_costs_energy_capacity[["annualized_costs"]].squeeze())
        # print(existing_annualized_costs_elec, existing_annualized_costs_CH4)

        if add_CH4_demand:
            # Create additional gas profile (tertiary heating + ECS)
            heating_gas_demand = heating_gas_demand_RTE_timesteps[anticipated_year_eoles] * 1e3  # in TWh
            ECS_gas_demand = ECS_gas_demand_RTE_timesteps[anticipated_year_eoles] * 1e3  # in TWh
            ECS_demand_hourly = ECS_gas_demand / 8760
            hourly_gas = eoles.utils.create_hourly_residential_demand_profile(total_consumption=heating_gas_demand,
                                                                              method=HOURLY_PROFILE_METHOD)  # value for gas heating demand in tertiary sector
            hourly_ECS = pd.Series(ECS_demand_hourly, index=hourly_gas.index)
            hourly_exogeneous_CH4 = hourly_gas + hourly_ECS
        else:
            # we do not add any demand
            hourly_exogeneous_CH4 = eoles.utils.create_hourly_residential_demand_profile(total_consumption=0,
                                                                                         method=HOURLY_PROFILE_METHOD)

        if optimization:
            # Find optimal subsidy
            optimizer, opt_sub = \
                optimize_blackbox_resirf_eoles_coupling(buildings, energy_prices, taxes, cost_heater, cost_insulation,
                                                        demolition_rate, flow_built, post_inputs, policies_heater, policies_insulation,
                                                        start_year_resirf, timestep_resirf,
                                                        config_eoles, year_eoles, anticipated_year_eoles, scc,
                                                        hourly_gas_exogeneous=hourly_exogeneous_CH4,
                                                        existing_capacity=existing_capacity,
                                                        existing_charging_capacity=existing_charging_capacity,
                                                        existing_energy_capacity=existing_energy_capacity,
                                                        maximum_capacity=maximum_capacity, method_hourly_profile="valentin",
                                                        scenario_cost=scenario_cost,
                                                        existing_annualized_costs_elec=existing_annualized_costs_elec,
                                                        existing_annualized_costs_CH4=existing_annualized_costs_CH4,
                                                        existing_annualized_costs_H2=existing_annualized_costs_H2,
                                                        lifetime_renov=lifetime_renov, lifetime_heater=lifetime_heater,
                                                        discount_rate=config_coupling["discount_rate"], plot=False,
                                                        max_iter=config_coupling["max_iter"], initial_design_numdata=3,
                                                        grid_initialize=grid_initialize, acquisition_jitter=acquisition_jitter,
                                                        normalize_Y=normalize_Y,
                                                        fix_sub_heater=config_coupling["fix_sub_heater"], fix_sub_insulation=config_coupling["fix_sub_insulation"],
                                                        sub_design=config_coupling["sub_design"],
                                                        health=config_coupling["health"], carbon_constraint=config_coupling["carbon_constraint"],
                                                        rebound=config_coupling["rebound"], technical_progress=technical_progress,
                                                        financing_cost=financing_cost, premature_replacement=premature_replacement)
            dict_optimizer.update({y: optimizer})
            opt_sub_heater, opt_sub_insulation = opt_sub[0], opt_sub[1]
            list_sub_heater.append(opt_sub_heater)
            list_sub_insulation.append(opt_sub_insulation)
        else:
            opt_sub_heater, opt_sub_insulation = list_sub_heater[t], list_sub_insulation[t]

        # Rerun ResIRF with optimal subvention parameters
        endyear_resirf = start_year_resirf + timestep_resirf

        output_opt, stock_opt, heating_consumption = simu_res_irf(buildings=buildings, sub_heater=opt_sub_heater,
                                                       sub_insulation=opt_sub_insulation, start=start_year_resirf,
                                                       end=endyear_resirf, energy_prices=energy_prices, taxes=taxes,
                                                       cost_heater=cost_heater, cost_insulation=cost_insulation,
                                                       lifetime_heater=lifetime_heater, flow_built=flow_built,
                                                       post_inputs=post_inputs, policies_heater=policies_heater,
                                                       policies_insulation=policies_insulation,
                                                       climate=2006, smooth=False, efficiency_hour=True, demolition_rate=demolition_rate,
                                                       output_consumption=True,
                                                       full_output=True, sub_design=config_coupling["sub_design"],
                                                       rebound=config_coupling["rebound"], technical_progress=technical_progress,
                                                       financing_cost=financing_cost, premature_replacement=premature_replacement)
        # output_opt['Total'] = output_opt.sum(axis=1)
        output_global_ResIRF = pd.concat([output_global_ResIRF, output_opt], axis=1)
        stock_global_ResIRF = pd.concat([stock_global_ResIRF, stock_opt], axis=1)

        # heating_consumption = output_opt[endyear_resirf - 1]['Consumption (kWh/h)']  # old
        heating_consumption = heating_consumption.sort_index(ascending=True)

        hourly_heat_elec = heating_consumption.loc["Electricity"] * 1e-6  # GWh
        hourly_heat_gas = heating_consumption.loc["Natural gas"] * 1e-6  # GWh
        hourly_heat_elec = hourly_heat_elec.reset_index().drop(columns=["index"]).squeeze()
        hourly_heat_gas = hourly_heat_gas.reset_index().drop(columns=["index"]).squeeze()
        hourly_heat_gas = hourly_heat_gas + hourly_exogeneous_CH4  # we add exogeneous gas demand

        electricity_consumption = heating_consumption.sum(axis=1)["Electricity"] * 1e-9  # TWh
        gas_consumption = heating_consumption.sum(axis=1)["Natural gas"] * 1e-9  # TWh
        oil_consumption = heating_consumption.sum(axis=1)["Oil fuel"] * 1e-9  # TWh
        wood_consumption = heating_consumption.sum(axis=1)["Wood fuel"] * 1e-9  # TWh

        list_consumption_yearly = ["Consumption Electricity (TWh)", "Consumption Natural gas (TWh)",
                                   "Consumption Oil fuel (TWh)", "Consumption Wood fuel (TWh)"]
        resirf_consumption_yearly = output_opt.loc[[ind for ind in output_opt.index if ind in list_consumption_yearly]]  # TWh
        resirf_consumption_yearly_df = pd.concat([resirf_consumption_yearly_df, resirf_consumption_yearly], axis=1)

        list_inv = ["Investment heater WT (Billion euro)", "Investment insulation WT (Billion euro)", "Subsidies heater (Billion euro)",
                    "Subsidies insulation (Billion euro)", "Health cost (Billion euro)"]
        investment_resirf = output_opt.loc[[ind for ind in output_opt.index if ind in list_inv]]  # 1e9 €
        resirf_costs_df = pd.concat([resirf_costs_df, investment_resirf], axis=1)

        list_saving = ["Consumption saving heater (TWh/year)", "Consumption saving insulation (TWh/year)"]
        consumption_saving = output_opt.loc[[ind for ind in output_opt.index if ind in list_saving]]  # TWh
        resirf_consumption_saving_df = pd.concat([resirf_consumption_saving_df, consumption_saving], axis=1)

        annuity_investment_heater_cost = calculate_annuities_resirf(output_opt.loc["Investment heater WT (Billion euro)"].sum(),
                                                                    lifetime=lifetime_heater, discount_rate=config_coupling["discount_rate"])
        annuity_investment_insulation_cost = calculate_annuities_resirf( output_opt.loc["Investment insulation WT (Billion euro)"].sum(), lifetime=lifetime_renov,
                                                                    discount_rate=config_coupling["discount_rate"])

        annuity_health_cost = output_opt.loc["Health cost (Billion euro)"][endyear_resirf - 1]

        replacement_heater = output_opt.loc[[ind for ind in output_opt.index if ind in LIST_REPLACEMENT_HEATER]]
        replacement_heater_df = pd.concat([replacement_heater_df, replacement_heater], axis=1)

        stock_output = output_opt.loc[[ind for ind in output_opt.index if ind in LIST_STOCK_HEATER]]
        stock_output = stock_output[stock_output.columns[-1]]

        if 'Cost rebound (Billion euro)' in output_opt.index:
            rebound_cost = output_opt.loc['Cost rebound (Billion euro)'].sum()  # 1e9 €
            list_cost_rebound.append(rebound_cost)

        list_cost_eff = ['Investment heater / saving (euro/kWh)', 'Investment insulation / saving (euro/kWh)']
        investment_cost_eff = output_opt.loc[[ind for ind in output_opt.index if ind in list_cost_eff]]  # TWh
        investment_cost_eff_df = pd.concat([investment_cost_eff_df, investment_cost_eff], axis=1)

        list_electricity_consumption.append(electricity_consumption)
        list_gas_consumption.append(gas_consumption)
        list_wood_consumption.append(wood_consumption)
        list_oil_consumption.append(oil_consumption)

        stock_heater_df = pd.concat([stock_heater_df, stock_output.to_frame()], axis=1)
        list_insulation_annualized.append(annuity_investment_insulation_cost)
        list_heater_annualized.append(annuity_investment_heater_cost)
        list_healthcost_annualized.append(annuity_health_cost)

        # Rerun EOLES with optimal parameters
        m_eoles = ModelEOLES(name="trajectory", config=config_eoles, path="eoles/outputs", logger=logger, nb_years=1,
                             hourly_heat_elec=hourly_heat_elec, hourly_heat_gas=hourly_heat_gas,
                             wood_consumption=wood_consumption * 1e3,  # GWh
                             oil_consumption=oil_consumption * 1e3,
                             existing_capacity=existing_capacity, existing_charging_capacity=existing_charging_capacity,
                             existing_energy_capacity=existing_energy_capacity, maximum_capacity=maximum_capacity,
                             method_hourly_profile="valentin",
                             anticipated_social_cost_of_carbon=scc, actual_social_cost_of_carbon=actual_scc, year=year_eoles, anticipated_year=anticipated_year_eoles,
                             scenario_cost=scenario_cost, existing_annualized_costs_elec=existing_annualized_costs_elec,
                             existing_annualized_costs_CH4=existing_annualized_costs_CH4,
                             existing_annualized_costs_H2=existing_annualized_costs_H2, carbon_constraint=config_coupling["carbon_constraint"],
                             discount_rate=config_coupling["discount_rate"])

        m_eoles.build_model()
        solver_results, status, termination_condition = m_eoles.solve(solver_name="gurobi")

        if termination_condition == "infeasibleOrUnbounded":
            logger.info("Carbon budget is violated.")
            if optimization:
                resirf_subsidies_df = pd.DataFrame({'Heater': list_sub_heater, 'Insulation': list_sub_insulation},
                                                   index=list_year[:t+1])
            else:
                resirf_subsidies_df = pd.DataFrame({'Heater': list_sub_heater[:t+1], 'Insulation': list_sub_insulation[:t+1]},
                                                   index=list_year[:t+1])
            output = {
                "Capacities (GW)": capacity_df,
                 "New capacities (GW)": new_capacity_df,
                 "Generation (TWh)": generation_df,
                 "Primary generation (TWh)": primary_generation_df,
                 "Conversion generation (TWh)": conversion_generation_df,
                 "Charging capacity (GW)": charging_capacity_df,
                 "Energy capacity (GW)": energy_capacity_df,
                 "Annualized new investments (1e9€/yr)": annualized_new_investment_df,
                 "Annualized costs new energy capacity (1e9€/yr)": annualized_new_energy_capacity_df,
                "Annualized costs historical capacity (1e9€/yr)": annualized_historical_capacity_df,
                "Annualized costs historical energy capacity (1e9€/yr)": annualized_historical_energy_capacity_df,
                 "System functionment (1e9€/yr)": functionment_costs_df,
                 "Emissions (MtCO2)": pd.DataFrame({"Emissions": list_emissions}, index=list_anticipated_year[:t + 1]),
                 "Peak electricity load": peak_electricity_load_df,
                 "Peak heat load": peak_heat_load_df,
                 "Subsidies (%)": resirf_subsidies_df,
                 "ResIRF costs (Billion euro)": resirf_costs_df.T,
                 "ResIRF costs eff (euro/kWh)": investment_cost_eff_df.T,
                 "ResIRF consumption yearly (TWh)": resirf_consumption_yearly_df.T,
                 "ResIRF consumption savings (TWh/year)": resirf_consumption_saving_df.T,
                 "ResIRF replacement heater (Thousand)": replacement_heater_df.T,
                 "ResIRF stock heater (Thousand)": stock_heater_df,
                 "Output global ResIRF ()": output_global_ResIRF,
                "Stock global ResIRF ()": stock_global_ResIRF,
                 "Spot price EOLES (€ / MWh)": spot_price_df
             }
            if price_feedback:
                output.update({
                    "Electricity price without tax": pd.DataFrame(
                        {"Electricity price without tax": list_electricity_price_ht}, index=list_anticipated_year[:t + 1]),
                    "Transport and distribution price": pd.DataFrame(
                        {"Transport and distribution price": list_transport_distribution_lcoe}, index=list_anticipated_year[:t + 1])
                })
            return output, buildings, dict_optimizer

        if anticipated_demand_t10:  # here, we modify the value of anticipated_year_eoles for saving dataframes
            anticipated_year_eoles = anticipated_year_eoles - 5

        if price_feedback:  # we use EOLES LCOE to estimate future energy prices for period [t+1;t+2[
            lcoe_elec = m_eoles.summary["lcoe_elec"]
            transport_and_distribution_lcoe = m_eoles.summary["transport_and_distrib_lcoe"]
            gas_furniture_cost = m_eoles.summary["lcoe_CH4_volume_noSCC"]
            elec_price_ht, gas_price_ht = electricity_gas_price_ht(lcoe_elec, transport_and_distribution_lcoe, gas_furniture_cost,
                                                                   calib_elec_lcoe=config_coupling["calibration_elec_lcoe"],
                                                                   calib_elec_transport_distrib=config_coupling["calibration_elec_transport_distrib"],
                                                                   calib_gas=config_coupling["calibration_gas_lcoe"],
                                                                   year=anticipated_year_eoles)
            list_electricity_price_ht.append(elec_price_ht)
            list_transport_distribution_lcoe.append(transport_and_distribution_lcoe)
            # energy_prices = new_projection_prices(energy_prices, elec_price_ht, energy_taxes, start=anticipated_year_eoles, end=anticipated_year_eoles+5)
            energy_prices = new_projection_prices_with_gas(energy_prices, elec_price_ht, gas_price_ht, energy_taxes,
                                                           start=anticipated_year_eoles, end=anticipated_year_eoles+5)  # we update energy prices beliefs for the coming years
        ### Spot price
        spot_price = m_eoles.spot_price.rename(columns={"elec_spot_price": f"elec_spot_price_{anticipated_year_eoles}",
                                                        "CH4_spot_price": f"CH4_spot_price_{anticipated_year_eoles}"})
        spot_price_df = pd.concat([spot_price_df, spot_price], axis=1)

        peak_electricity_load = m_eoles.peak_electricity_load_info
        peak_electricity_load["year"] = anticipated_year_eoles
        peak_electricity_load_df = pd.concat([peak_electricity_load_df, peak_electricity_load], axis=0)

        peak_heat_load = m_eoles.peak_heat_load_info
        peak_heat_load["year"] = anticipated_year_eoles
        peak_heat_load_df = pd.concat([peak_heat_load_df, peak_heat_load], axis=0)

        list_emissions.append(m_eoles.emissions.sum())  # TODO: à verifier

        if anticipated_year_eoles == 2050:
            hourly_generation_2050 = m_eoles.hourly_generation

        #### Get annuity and functionment cost corresponding to each technology
        new_capacity_annualized_costs_nofOM = m_eoles.new_capacity_annualized_costs_nofOM / 1000  # 1e9 € / yr
        new_capacity_annualized_costs_nofOM = pd.concat([new_capacity_annualized_costs_nofOM,
                   pd.DataFrame(index=["investment_heater", "investment_insulation"], data={'annualized_costs': [annuity_investment_heater_cost, annuity_investment_insulation_cost]})], axis=0)
        annualized_new_investment_df = pd.concat([annualized_new_investment_df, new_capacity_annualized_costs_nofOM.rename(columns={"annualized_costs": anticipated_year_eoles})], axis=1)

        new_energy_capacity_annualized_costs_nofOM = m_eoles.new_energy_capacity_annualized_costs_nofOM / 1000
        annualized_new_energy_capacity_df = pd.concat([annualized_new_energy_capacity_df, new_energy_capacity_annualized_costs_nofOM.rename(columns={"annualized_costs": anticipated_year_eoles})], axis=1)

        functionment_cost = m_eoles.functionment_cost / 1000  # 1e9 € / yr
        functionment_cost = pd.concat([functionment_cost,
                   pd.DataFrame(index=["health_costs"], data={'functionment_cost': [annuity_health_cost]})], axis=0)
        functionment_costs_df = pd.concat([functionment_costs_df, functionment_cost.rename(columns={"functionment_cost": anticipated_year_eoles})], axis=1)

        # Historical annuity cost
        annualized_costs_capacity_nofOM_historical = annualized_costs_capacity_nofOM_historical / 1000  # 1e9 € / yr
        annualized_costs_energy_capacity_historical = annualized_costs_energy_capacity_historical / 1000  # 1e9 € / yr
        annualized_historical_capacity_df = pd.concat([annualized_historical_capacity_df, annualized_costs_capacity_nofOM_historical.rename(columns={'annualized_costs': anticipated_year_eoles})], axis=1)
        annualized_historical_energy_capacity_df = pd.concat([annualized_historical_energy_capacity_df, annualized_costs_energy_capacity_historical.rename(columns={'annualized_costs': anticipated_year_eoles})], axis=1)

        #### Update total capacities and new capacities
        new_capacity = m_eoles.capacities - existing_capacity  # we get the newly installed capacities at year y
        new_charging_capacity = m_eoles.charging_capacity - existing_charging_capacity
        new_energy_capacity = m_eoles.energy_capacity - existing_energy_capacity
        gene_per_tec = pd.Series(m_eoles.generation_per_technology)

        new_capacity_tot = new_capacity_tot + new_capacity  # total newly built capacity over the time horizon until t included
        new_charging_capacity_tot = new_charging_capacity_tot + new_charging_capacity
        new_energy_capacity_tot = new_energy_capacity_tot + new_energy_capacity

        # update annualized costs from new capacity built until time t included
        annualized_costs_new_capacity = pd.concat([annualized_costs_new_capacity[["annualized_costs"]],
                                                   m_eoles.new_capacity_annualized_costs[["annualized_costs"]].rename(
                                                       columns={'annualized_costs': 'new_annualized_costs'})], axis=1)
        annualized_costs_new_capacity["annualized_costs"] = annualized_costs_new_capacity["annualized_costs"] + \
                                                            annualized_costs_new_capacity['new_annualized_costs']
        annualized_costs_new_capacity = annualized_costs_new_capacity.drop(columns=['new_annualized_costs'])

        annualized_costs_new_energy_capacity = pd.concat([annualized_costs_new_energy_capacity[["annualized_costs"]],
                                                          m_eoles.new_energy_capacity_annualized_costs[
                                                              ["annualized_costs"]].rename(
                                                              columns={'annualized_costs': 'new_annualized_costs'})],
                                                         axis=1)
        annualized_costs_new_energy_capacity["annualized_costs"] = annualized_costs_new_energy_capacity[
                                                                       "annualized_costs"] + \
                                                                   annualized_costs_new_energy_capacity[
                                                                       'new_annualized_costs']
        annualized_costs_new_energy_capacity = annualized_costs_new_energy_capacity.drop(
            columns=['new_annualized_costs'])

        capacity_df = pd.concat([capacity_df, m_eoles.capacities.to_frame().rename(columns={0: anticipated_year_eoles})], axis=1)
        charging_capacity_df = pd.concat(
            [charging_capacity_df, m_eoles.charging_capacity.to_frame().rename(columns={0: anticipated_year_eoles})], axis=1)
        generation_df = pd.concat([generation_df, gene_per_tec.to_frame().rename(columns={0: anticipated_year_eoles})], axis=1)
        energy_capacity_df = pd.concat([energy_capacity_df, m_eoles.energy_capacity.to_frame().rename(columns={0: anticipated_year_eoles})],
                                       axis=1)
        new_capacity_df = pd.concat([new_capacity_df, new_capacity.to_frame().rename(columns={0: anticipated_year_eoles})], axis=1)
        primary_generation_df = pd.concat([primary_generation_df,
                                           m_eoles.primary_generation.reindex(reindex_primary_prod).to_frame().rename(
                                               columns={0: anticipated_year_eoles})], axis=1)
        conversion_generation = pd.concat([m_eoles.CH4_to_power_generation.to_frame(),
                                           m_eoles.power_to_CH4_generation.to_frame(),
                                           m_eoles.power_to_H2_generation.to_frame()], axis=0)
        conversion_generation_df = pd.concat([conversion_generation_df, conversion_generation.rename(columns={0: anticipated_year_eoles})], axis=1)

        list_elec_annualized.append(m_eoles.objective)
        list_transport_distrib_annualized.append(m_eoles.transport_distribution_cost)
        total_annualized_costs = annuity_investment_heater_cost + annuity_investment_insulation_cost + annuity_health_cost + m_eoles.objective
        list_global_annualized_costs.append(total_annualized_costs)

        weighted_average_elec_price.append(m_eoles.summary["weighted_elec_price_demand"])
        weighted_average_CH4_price.append(m_eoles.summary["weighted_CH4_price_demand"])
        weighted_average_H2_price.append(m_eoles.summary["weighted_H2_price_demand"])

        list_lcoe_elec.append(m_eoles.summary["lcoe_elec"])
        list_lcoe_elec_volume.append(m_eoles.summary["lcoe_elec_volume"])
        list_lcoe_elec_value.append(m_eoles.summary["lcoe_elec_value"])
        list_lcoe_CH4_value.append(m_eoles.summary["lcoe_CH4_value"])
        list_lcoe_CH4_volume.append(m_eoles.summary["lcoe_CH4_volume"])
        list_lcoe_CH4_volume_noSCC.append(m_eoles.summary["lcoe_CH4_volume_noSCC"])

    price_df = pd.DataFrame(
        {'Average electricity price': weighted_average_elec_price, 'Average CH4 price': weighted_average_CH4_price,
         'Average H2 price': weighted_average_H2_price,
         'LCOE electricity': list_lcoe_elec, 'LCOE electricity value': list_lcoe_elec_value, 'LCOE electricity volume': list_lcoe_elec_volume,
         'LCOE CH4 value': list_lcoe_CH4_value, 'LCOE CH4 volume': list_lcoe_CH4_volume, 'LCOE CH4 volume noSCC': list_lcoe_CH4_volume_noSCC},
        index=list_anticipated_year)

    annualized_system_costs_df = pd.DataFrame(
        {'Annualized electricity system costs': list_elec_annualized,
         'Annualized investment heater costs': list_heater_annualized,
         'Annualized investment insulation costs': list_insulation_annualized,
         'Annualized health costs': list_healthcost_annualized, "Annualized total costs": list_global_annualized_costs},
        index=list_anticipated_year
    )

    resirf_subsidies_df = pd.DataFrame({'Heater': list_sub_heater, 'Insulation': list_sub_insulation},
                                    index=list_year)  # we keep list_year, as the subsidies are applied from y to y+5

    resirf_consumption_df = pd.DataFrame(
        {'Electricity': list_electricity_consumption, "Natural gas": list_gas_consumption,
         'Oil fuel': list_oil_consumption, 'Wood fuel': list_wood_consumption}, index=list_anticipated_year)

    output = {
        "Capacities (GW)": capacity_df,
        "New capacities (GW)": new_capacity_df,
        "Generation (TWh)": generation_df,
        "Primary generation (TWh)": primary_generation_df,
        "Conversion generation (TWh)": conversion_generation_df,
        "Charging capacity (GW)": charging_capacity_df,
        "Energy capacity (GW)": energy_capacity_df,
        "Annualized new investments (1e9€/yr)": annualized_new_investment_df,
        "Annualized costs new energy capacity (1e9€/yr)": annualized_new_energy_capacity_df,
        "Annualized costs historical capacity (1e9€/yr)": annualized_historical_capacity_df,
        "Annualized costs historical energy capacity (1e9€/yr)": annualized_historical_energy_capacity_df,
        "System functionment (1e9€/yr)": functionment_costs_df,
        "Prices (€/MWh)": price_df,
        "Emissions (MtCO2)": pd.DataFrame({"Emissions": list_emissions}, index=list_anticipated_year),
        "Peak electricity load": peak_electricity_load_df,
        "Peak heat load": peak_heat_load_df,
        "Subsidies (%)": resirf_subsidies_df,
        "ResIRF costs (Billion euro)": resirf_costs_df.T,
        "ResIRF costs eff (euro/kWh)": investment_cost_eff_df.T,
        "ResIRF consumption (TWh)": resirf_consumption_df,
        "ResIRF consumption yearly (TWh)": resirf_consumption_yearly_df.T,
        "ResIRF consumption savings (TWh)": resirf_consumption_saving_df.T,
        "ResIRF replacement heater (Thousand)": replacement_heater_df.T,
        "ResIRF stock heater (Thousand)": stock_heater_df,
        "Annualized system costs (Billion euro / year)": annualized_system_costs_df,
        'Transport and distribution costs (Billion euro / year)': pd.DataFrame({"Transport and distribution": list_transport_distrib_annualized}, index=list_anticipated_year),
        "Output global ResIRF ()": output_global_ResIRF,
        "Stock global ResIRF ()": stock_global_ResIRF,
        "Spot price EOLES (€ / MWh)": spot_price_df,
        "Hourly generation 2050 (GWh)": hourly_generation_2050,
        "Energy prices (€/kWh)": energy_prices
    }

    if price_feedback:
        output.update({
            "Electricity price without tax": pd.DataFrame({"Electricity price without tax": list_electricity_price_ht}, index=list_anticipated_year),
            "Transport and distribution price": pd.DataFrame({"Transport and distribution price": list_transport_distribution_lcoe}, index=list_anticipated_year)
        })

    return output, buildings, dict_optimizer


def list_scc_yearly():
    """Returns a list of social cost of carbon by extrapolating linearly the values for each year. After 2050, we
    assume that the value does not evolve anymore.
    ATTENTION: on prend ici les valeurs de SCC commençant à 2030 ! A changer si on change d'autres choses d'ici là."""
    list_scc = [250, 350, 500, 650, 775]
    new_L = []
    for t in range(len(list_scc)-1):
        for i in range(1,5,1):
            new_L.append(list_scc[t] + (list_scc[t+1] - list_scc[t])/5*i)
    for i in range(30):
        final_scc = list_scc[-1] * 1.032**(i+1)  # we assume an increase following the discount rate trend
        new_L.append(final_scc)
    new_L = list_scc + new_L
    new_L.sort()
    return new_L


def average_scc(list_scc, lifetime, initial_time):
    """Calculates average scc during the global lifetime of the system."""
    initial_time_yearly = initial_time*5
    final_time_yearly = initial_time_yearly + lifetime
    return np.array(list_scc[initial_time_yearly:final_time_yearly]).mean()


def average_scc_discounted(list_scc, lifetime, initial_time, discount_rate):  # TODO: a modifier !
    """Calculates average scc during the global lifetime of the system, with a discount factor."""
    initial_time_yearly = initial_time * 5
    final_time_yearly = initial_time_yearly + lifetime
    discount_rate_list = [1/(1+discount_rate)**(i) for i in range(lifetime)]
    discounted_scc = [scc*discount_factor for (scc, discount_factor) in zip(list_scc[initial_time_yearly:final_time_yearly], discount_rate_list)]
    return np.array(discounted_scc).mean()


def electricity_price_ht(system_lcoe, system_transport_and_distrib, calib_elec, year):
    """Estimates electricity price without tax in €/MWh."""
    electricity_prices_snbc = get_pandas("eoles/inputs/electricity_price_snbc.csv", lambda x: pd.read_csv(x, index_col=0).squeeze("columns"))
    electricity_prices_snbc_year = electricity_prices_snbc[[str(year)]].squeeze()
    commercial = electricity_prices_snbc_year["Commercial costs"]
    price_ht = (system_lcoe + system_transport_and_distrib) * calib_elec + commercial
    return price_ht


def electricity_gas_price_ht(elec_lcoe, elec_transport_distrib, gas_furniture_cost, calib_elec_lcoe, calib_elec_transport_distrib, calib_gas, year):
    """

    :param elec_system_cost: float
        LCOE of electricity coming from EOLES, and including capacities and energy, transport and distribution. Unit: €/MWh
    :param gas_furniture_cost: float
        LCOE of gas coming from EOLES, including only capacities and energy. Unit: €/MWh
    :param calib_elec: float
        Calibration parameter based on 2019 SNBC data
    :param calib_gas: float
        Calibration parameter based on 2019 SNBC data
    :param year: int
        Year of interest
    :return:
    """
    electricity_prices_snbc = get_pandas("eoles/inputs/electricity_price_snbc.csv", lambda x: pd.read_csv(x, index_col=0).squeeze("columns"))
    electricity_prices_snbc_year = electricity_prices_snbc[[str(year)]].squeeze()
    commercial_electricity = electricity_prices_snbc_year["Commercial costs"]
    elec_price_ht = elec_lcoe * calib_elec_lcoe + elec_transport_distrib * calib_elec_transport_distrib + commercial_electricity

    gas_prices_snbc = get_pandas("eoles/inputs/gas_price_snbc.csv", lambda x: pd.read_csv(x, index_col=0).squeeze("columns"))
    gas_prices_snbc_year = gas_prices_snbc[[str(year)]].squeeze()
    distribution_gas = gas_prices_snbc_year["Distribution network"]
    transport_gas = gas_prices_snbc_year["Transport network"]
    gas_price_ht = gas_furniture_cost * calib_gas + distribution_gas + transport_gas

    return elec_price_ht, gas_price_ht


def new_projection_prices(energy_prices, price_elec_ht, energy_taxes, start, end):
    """Modification of energy prices (from SNBC) based on EOLES output
    :param energy_prices: pd.DataFrame
        Contains energy prices with tax, as estimated by SNBC.
    :param energy_prices_ht: pd.DataFrame
        Contains energy prices without tax, as estimated by SNBC.
        """
    anticipated_energy_prices = energy_prices.copy().loc[start:end-1, "Electricity"]
    anticipated_energy_prices[:] = price_elec_ht / 1000  # €/kWh
    # anticipated_energy_prices.loc[start:end-1, "Electricity"] = price_elec_ht / 1000  # only interested in the given time interval. Unit: €/kWh
    # new_energy_prices.loc[:, "Natural gas"] = price_gas_ht  # TODO: ajouter la modification du prix du gaz

    energy_vta = anticipated_energy_prices * generic_input['vta_energy_prices']["Electricity"]

    anticipated_energy_prices = anticipated_energy_prices.add(energy_taxes.loc[start:end-1, "Electricity"], fill_value=0)  # we add exogenous taxes, as we would in ResIRF
    anticipated_energy_prices = anticipated_energy_prices.add(energy_vta, fill_value=0)  # we add VTA
    energy_prices.loc[start:end-1, "Electricity"] = anticipated_energy_prices
    return energy_prices


def new_projection_prices_with_gas(energy_prices, price_elec_ht, price_gas_ht, energy_taxes, start, end):
    """Modification of energy prices (from SNBC) based on EOLES output
    :param energy_prices: pd.DataFrame
        Contains energy prices with tax, as estimated by SNBC.
    :param energy_prices_ht: pd.DataFrame
        Contains energy prices without tax, as estimated by SNBC.
        """
    anticipated_energy_prices = energy_prices.copy().loc[start:end-1, ["Electricity", "Natural gas"]]
    anticipated_energy_prices.loc[:, "Electricity"] = price_elec_ht / 1000  # €/kWh
    anticipated_energy_prices.loc[:, "Natural gas"] = price_gas_ht / 1000  # €/kWh
    # anticipated_energy_prices.loc[start:end-1, "Electricity"] = price_elec_ht / 1000  # only interested in the given time interval. Unit: €/kWh
    # new_energy_prices.loc[:, "Natural gas"] = price_gas_ht  # TODO: ajouter la modification du prix du gaz

    energy_vta = anticipated_energy_prices * generic_input['vta_energy_prices'][["Electricity", "Natural gas"]]

    anticipated_energy_prices = anticipated_energy_prices.add(energy_taxes.loc[start:end-1, ["Electricity", "Natural gas"]], fill_value=0)  # we add exogenous taxes, as we would in ResIRF
    anticipated_energy_prices = anticipated_energy_prices.add(energy_vta, fill_value=0)  # we add VTA
    energy_prices.loc[start:end-1, ["Electricity", "Natural gas"]] = anticipated_energy_prices  # we update final values
    return energy_prices


def calibration_price(config_eoles, scc=100):
    """Returns calibration factor based on the SNBC LCOE."""
    # Initialization needed for calibration
    existing_capacity_historical, existing_charging_capacity_historical, existing_energy_capacity_historical, \
    maximum_capacity_evolution, heating_gas_demand_RTE_timesteps, ECS_gas_demand_RTE_timesteps, capex_annuity_fOM_historical, capex_annuity_historical, storage_annuity_historical = eoles.utils.load_evolution_data(config=config_eoles)

    existing_capacity_historical = existing_capacity_historical.drop(
        ["heat_pump", "resistive", "gas_boiler", "fuel_boiler", "wood_boiler"], axis=0)

    anticipated_year = 2020
    anticipated_historical = 2025  # ce sont les memes valeurs que en 2020, avec nos hypothèses
    new_capacity_tot = pd.Series(0, index=existing_capacity_historical.index, dtype=float)
    new_charging_capacity_tot = pd.Series(0, index=existing_charging_capacity_historical.index, dtype=float)
    new_energy_capacity_tot = pd.Series(0, index=existing_energy_capacity_historical.index, dtype=float)
    annualized_costs_new_capacity = pd.DataFrame(0, index=existing_capacity_historical.index,
                                                 columns=["annualized_costs"], dtype=float)
    annualized_costs_new_energy_capacity = pd.DataFrame(0, index=existing_energy_capacity_historical.index,
                                                        columns=["annualized_costs"], dtype=float)

    #### Get existing and maximum capacities
    existing_capa_historical_y = existing_capacity_historical[
        [str(anticipated_historical)]].squeeze()  # get historical capacity still installed for year of interest
    existing_charging_capacity_historical_y = existing_charging_capacity_historical[
        [str(anticipated_historical)]].squeeze()
    existing_energy_capacity_historical_y = existing_energy_capacity_historical[
        [str(anticipated_historical)]].squeeze()

    new_maximum_capacity_y = maximum_capacity_evolution[
        [str(anticipated_historical)]].squeeze()  # get maximum new capacity to be built

    # Existing capacities at year y
    existing_capacity = existing_capa_historical_y + new_capacity_tot  # existing capacity are equal to newly built
    # capacities over the whole time horizon before t + existing capacity (from before 2020)
    existing_charging_capacity = existing_charging_capacity_historical_y + new_charging_capacity_tot
    existing_energy_capacity = existing_energy_capacity_historical_y + new_energy_capacity_tot

    maximum_capacity = (
            existing_capacity + new_maximum_capacity_y).dropna()  # we drop nan values, which correspond to technologies without any upper bound

    #### Historical LCOE based on historical costs
    annualized_costs_capacity_historical, annualized_costs_energy_capacity_historical = eoles.utils.annualized_costs_investment_historical(
        existing_capa_historical_y, capex_annuity_fOM_historical,
        existing_energy_capacity_historical_y, storage_annuity_historical)

    ### Compile total annualized investment costs from existing capacities (both historical capacities + newly built capacities before t)
    # Necessary for calculus of LCOE accounting for evolution of capacities
    annualized_costs_capacity = pd.concat(
        [annualized_costs_capacity_historical.rename(columns={'annualized_costs': 'historical_annualized_costs'}),
         annualized_costs_new_capacity], axis=1)
    annualized_costs_capacity['annualized_costs'] = annualized_costs_capacity['historical_annualized_costs'] + \
                                                    annualized_costs_capacity['annualized_costs']
    annualized_costs_energy_capacity = pd.concat([annualized_costs_energy_capacity_historical.rename(
        columns={'annualized_costs': 'historical_annualized_costs'}), annualized_costs_new_energy_capacity], axis=1)
    annualized_costs_energy_capacity['annualized_costs'] = annualized_costs_energy_capacity[
                                                               'historical_annualized_costs'] + \
                                                           annualized_costs_energy_capacity['annualized_costs']

    existing_annualized_costs_elec, existing_annualized_costs_CH4, existing_annualized_costs_H2 = eoles.utils.process_annualized_costs_per_vector(
        annualized_costs_capacity[["annualized_costs"]].squeeze(),
        annualized_costs_energy_capacity[["annualized_costs"]].squeeze())

    # Hourly profiles for residential demand
    hourly_heat_elec = eoles.utils.create_hourly_residential_demand_profile(total_consumption=45 * 1e3,
                                                         method=HOURLY_PROFILE_METHOD)
    hourly_heat_gas = eoles.utils.create_hourly_residential_demand_profile(total_consumption=128 * 1e3,
                                                         method=HOURLY_PROFILE_METHOD)  # on ne se préoccupe pas du gaz chauffage ici, considéré comme indépendant (réaliste pour la situation actuelle)
    # We use year = 2025 to get technology parameters, and anticipated_year = 2020 to get demand data for 2020, since calibration is done in 2020.
    m_eoles = ModelEOLES(name="trajectory", config=config_eoles, path="eoles/outputs", logger=logger, nb_years=1,
                         hourly_heat_elec=hourly_heat_elec, hourly_heat_gas=hourly_heat_gas,
                         wood_consumption=0, oil_consumption=0,
                         existing_capacity=existing_capacity, existing_charging_capacity=existing_charging_capacity,
                         existing_energy_capacity=existing_energy_capacity, maximum_capacity=maximum_capacity,
                         method_hourly_profile="valentin",
                         anticipated_social_cost_of_carbon=scc, actual_social_cost_of_carbon=scc, year=anticipated_year + 5, anticipated_year=anticipated_year,
                         scenario_cost=None, existing_annualized_costs_elec=existing_annualized_costs_elec,
                         existing_annualized_costs_CH4=existing_annualized_costs_CH4,
                         existing_annualized_costs_H2=existing_annualized_costs_H2, carbon_constraint=False,
                         discount_rate=0.032)
    m_eoles.build_model()
    solver_results, status, termination_condition = m_eoles.solve(solver_name="gurobi")
    lcoe_elec = m_eoles.summary["lcoe_elec"]
    transport_and_distribution_lcoe = m_eoles.summary["transport_and_distrib_lcoe"]
    snbc_lcoe_2020 = 48.6
    snbc_transport_distribution = 39.5+10.8  # we make the assumption that since there is very little offshore wind power plants, we can assimilate this to transport and distribution without offshore (as in m_eoles)
    # calibration_elec_lcoe = (snbc_lcoe_2020 + snbc_transport_distribution) / (lcoe_elec + transport_and_distribution_lcoe)
    calibration_elec_lcoe = snbc_lcoe_2020 / lcoe_elec
    calibration_elec_transport_distrib = snbc_transport_distribution / transport_and_distribution_lcoe

    lcoe_CH4_noSCC = m_eoles.summary["lcoe_CH4_volume_noSCC"]
    snbc_gas_price = 23.50
    calibration_gas = snbc_gas_price / lcoe_CH4_noSCC

    return calibration_elec_lcoe, calibration_elec_transport_distrib, calibration_gas, m_eoles


def get_energy_prices_and_taxes(config):
    """
    Returns energy prices without tax and energy taxes (from ResIRF config).
    :param config: json file
        This is a ResIRF config
    :return:
    """
    with open(config) as file:
        config = json.load(file).get('Reference')
    energy_prices = get_pandas(config['macro']['energy_prices'], lambda x: pd.read_csv(x, index_col=[0]).rename_axis('Year').rename_axis('Heating energy', axis=1))

    energy_taxes = get_pandas(config['macro']['energy_taxes'], lambda x: pd.read_csv(x, index_col=[0]).rename_axis('Year').rename_axis('Heating energy', axis=1))
    return energy_prices, energy_taxes
