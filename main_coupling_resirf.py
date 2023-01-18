import json
import os
from importlib import resources

import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from pickle import dump, load

from project.sensitivity import ini_res_irf, simu_res_irf
from project.model import get_inputs, social_planner

from project.model import create_logger, get_config, get_inputs
from eoles.model_resirf_coupling import ModelEOLES
from eoles.utils import get_config, get_pandas, calculate_annuities_resirf
import eoles.utils
from eoles.coupling_resirf_eoles import resirf_eoles_coupling_dynamic, optimize_blackbox_resirf_eoles_coupling, resirf_eoles_coupling_dynamic_no_opti
import logging


from matplotlib import pyplot as plt

HOURLY_PROFILE_METHOD = "valentin"
HEATING_MULTIPLYING_PARAMETER = 1.1
# ['Occupancy status', 'Income owner', 'Income tenant', 'Housing type', 'Heating system']
AGGREGATION_ARCHETYPE = ['Wall class', "Housing type"]
CLIMATE_YEAR = 2006

LOG_FORMATTER = '%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# consoler handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
logger.addHandler(console_handler)


def test_convergence(max_iter, initial_design_numdata):
    existing_capacity_historical, existing_charging_capacity_historical, existing_energy_capacity_historical, \
    maximum_capacity_evolution, heating_gas_demand_RTE_timesteps, ECS_gas_demand_RTE_timesteps, annuity_fOM_historical, storage_annuity_historical = eoles.utils.load_evolution_data()

    existing_capacity_historical = existing_capacity_historical.drop(
        ["heat_pump", "resistive", "gas_boiler", "fuel_boiler", "wood_boiler"], axis=0)

    year_eoles, anticipated_year_eoles = 2025, 2030
    start_year_resirf, timestep_resirf = 2025, 5

    existing_capa_historical_y = existing_capacity_historical[
        [str(anticipated_year_eoles)]].squeeze()  # get historical capacity still installed for year of interest
    existing_charging_capacity_historical_y = existing_charging_capacity_historical[
        [str(anticipated_year_eoles)]].squeeze()
    existing_energy_capacity_historical_y = existing_energy_capacity_historical[[str(anticipated_year_eoles)]].squeeze()

    new_maximum_capacity_y = maximum_capacity_evolution[
        [str(anticipated_year_eoles)]].squeeze()  # get maximum new capacity to be built

    # Existing capacities at year y
    existing_capacity = existing_capa_historical_y
    existing_charging_capacity = existing_charging_capacity_historical_y
    existing_energy_capacity = existing_energy_capacity_historical_y

    maximum_capacity = (
            existing_capacity + new_maximum_capacity_y).dropna()  # we drop nan values, which correspond to technologies without any upper bound

    #### Historical LCOE based on historical costs
    annualized_costs_capacity_historical, annualized_costs_energy_capacity_historical = eoles.utils.annualized_costs_investment_historical(
        existing_capa_historical_y, annuity_fOM_historical,
        existing_energy_capacity_historical_y, storage_annuity_historical)
    annualized_costs_new_capacity = pd.DataFrame(0, index=existing_capacity_historical.index,
                                                 columns=["annualized_costs"], dtype=float)
    annualized_costs_new_energy_capacity = pd.DataFrame(0, index=existing_energy_capacity_historical.index,
                                                        columns=["annualized_costs"], dtype=float)

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

    ### Create additional gas profile (tertiary heating + ECS)
    heating_gas_demand = heating_gas_demand_RTE_timesteps[anticipated_year_eoles] * 1e3  # in TWh
    ECS_gas_demand = ECS_gas_demand_RTE_timesteps[anticipated_year_eoles] * 1e3  # in TWh
    ECS_demand_hourly = ECS_gas_demand / 8760
    hourly_gas = eoles.utils.create_hourly_residential_demand_profile(total_consumption=heating_gas_demand,
                                                                      method=HOURLY_PROFILE_METHOD)  # value for gas heating demand in tertiary sector
    hourly_ECS = pd.Series(ECS_demand_hourly, index=hourly_gas.index)
    hourly_exogeneous_CH4 = hourly_gas + hourly_ECS

    # Find optimal subsidy
    optimizer, opt_sub = \
        optimize_blackbox_resirf_eoles_coupling(buildings, energy_prices, taxes, cost_heater, cost_insulation,
                                                flow_built, post_inputs, start_year_resirf, timestep_resirf,
                                                config_eoles, year_eoles, anticipated_year_eoles, 180,
                                                hourly_gas_exogeneous=hourly_exogeneous_CH4,
                                                existing_capacity=existing_capacity,
                                                existing_charging_capacity=existing_charging_capacity,
                                                existing_energy_capacity=existing_energy_capacity,
                                                maximum_capacity=maximum_capacity, method_hourly_profile="valentin",
                                                scenario_cost=scenario_cost,
                                                existing_annualized_costs_elec=existing_annualized_costs_elec,
                                                existing_annualized_costs_CH4=existing_annualized_costs_CH4,
                                                existing_annualized_costs_H2=existing_annualized_costs_H2,
                                                lifetime_renov=40, discount_rate_renov=0.045, plot=False,
                                                max_iter=max_iter, initial_design_numdata=initial_design_numdata)

    return max_iter, initial_design_numdata, optimizer


def test_convergence_2030():
    output, optimizer = resirf_eoles_coupling_dynamic(buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built,
                                  post_inputs, [2025, 2030], [180, 250], scenario_cost, config_eoles, max_iter=22, return_optimizer=True)
    return output, optimizer


if __name__ == '__main__':
    buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built, post_inputs = ini_res_irf(
        path=os.path.join("eoles", "outputs"),
        logger=None,
        config=None)

    # # simulation between start and end - flow output represents annual values for year "end"
    # output = simu_res_irf(buildings, 0.0, 0.0, 2020, 2021, energy_prices,
    #                       taxes, cost_heater, cost_insulation, flow_built, post_inputs)
    #
    # heating_consumption = output[2021 - 1]['Hourly consumption (kWh)']
    # heating_consumption = heating_consumption.sort_index(ascending=True)
    #
    # hourly_heat_elec = heating_consumption.loc["Electricity"] * 1e-6  # GWh
    # hourly_heat_gas = heating_consumption.loc["Natural gas"] * 1e-6  # GWh
    # hourly_heat_elec = hourly_heat_elec.reset_index().drop(columns=["index"]).squeeze()
    # hourly_heat_gas = hourly_heat_gas.reset_index().drop(columns=["index"]).squeeze()

    list_year = [2025, 2030, 2035, 2040, 2045]
    list_trajectory_scc = [180, 250, 350, 500, 650]  # SCC trajectory from Quinet
    config_eoles = eoles.utils.get_config(spec="greenfield")  # TODO: changer le nom de la config qu'on appelle
    scenario_cost = {
        "capex": {
            "gas_boiler": 296,  # hypothesis Zeyen
            "fuel_boiler": 300,  # hypothesis Zeyen
            "wood_boiler": 300,  # hypothesis Zeyen
        },
        "fOM": {
            "gas_boiler": 19  # hypothesis Zeyen
        },
        "conversion_efficiency": {
            "gas_boiler": 0.9,
        },
        "miscellaneous": {
            "lifetime_renov": 40
        },
        "fix_capa": {
            "h2_ccgt": 0
        }
    }
    # output = resirf_eoles_coupling_dynamic(buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built,
    #                                        post_inputs, list_year, list_trajectory_scc, scenario_cost, config_eoles=config_eoles)

    #### Study trajectory with given list of subsidies
    # list_sub_heater = [0.80, 0.20, 0.25, 0.46, 0.0]
    # list_sub_insulation = [0.0, 0.0, 0.0, 0.0, 0.0]
    # output = resirf_eoles_coupling_dynamic_no_opti(list_sub_heater, list_sub_insulation, buildings, energy_prices, taxes,
    #                                       cost_heater, cost_insulation, flow_built, post_inputs,
    #                                       list_year, list_trajectory_scc, scenario_cost, config_eoles)

    # # Save results
    # date = datetime.datetime.now().strftime("%m%d%H")
    # export_results = os.path.join("eoles", "outputs", date)
    #
    # if not os.path.isdir(export_results):
    #     os.mkdir(export_results)
    #
    # with open(os.path.join(export_results, 'coupling_results.pkl'), "wb") as file:
    #     dump(output, file)

    # objective = resirf_eoles_coupling_static(np.array([[0, 0]]), buildings, energy_prices, taxes, cost_heater,
    #                                          cost_insulation, flow_built,
    #                                          post_inputs,
    #                                          start_year_resirf=2020, timestep_resirf=1, config_eoles=config_eoles,
    #                                          year_eoles=2050, anticipated_year_eoles=2050, scc=0, hourly_gas_exogeneous=0,
    #                                          existing_capacity=None, existing_charging_capacity=None,
    #                                          existing_energy_capacity=None,
    #                                          maximum_capacity=None, method_hourly_profile="valentin",
    #                                          scenario_cost=None, existing_annualized_costs_elec=0,
    #                                          existing_annualized_costs_CH4=0,
    #                                          existing_annualized_costs_H2=0, lifetime_renov=40,
    #                                          discount_rate_renov=0.045)

    #
    ## Test convergence of the result
    # existing_capacity_historical, existing_charging_capacity_historical, existing_energy_capacity_historical, \
    # maximum_capacity_evolution, heating_gas_demand_RTE_timesteps, ECS_gas_demand_RTE_timesteps, annuity_fOM_historical, storage_annuity_historical = eoles.utils.load_evolution_data()
    #
    # existing_capacity_historical = existing_capacity_historical.drop(["heat_pump", "resistive", "gas_boiler", "fuel_boiler", "wood_boiler"], axis=0)
    #
    # year_eoles, anticipated_year_eoles = 2025, 2030
    # start_year_resirf, timestep_resirf = 2025, 5
    #
    # existing_capa_historical_y = existing_capacity_historical[
    #     [str(anticipated_year_eoles)]].squeeze()  # get historical capacity still installed for year of interest
    # existing_charging_capacity_historical_y = existing_charging_capacity_historical[
    #     [str(anticipated_year_eoles)]].squeeze()
    # existing_energy_capacity_historical_y = existing_energy_capacity_historical[[str(anticipated_year_eoles)]].squeeze()
    #
    # new_maximum_capacity_y = maximum_capacity_evolution[[str(anticipated_year_eoles)]].squeeze()  # get maximum new capacity to be built
    #
    # # Existing capacities at year y
    # existing_capacity = existing_capa_historical_y
    # existing_charging_capacity = existing_charging_capacity_historical_y
    # existing_energy_capacity = existing_energy_capacity_historical_y
    #
    # maximum_capacity = (
    #         existing_capacity + new_maximum_capacity_y).dropna()  # we drop nan values, which correspond to technologies without any upper bound
    #
    # #### Historical LCOE based on historical costs
    # annualized_costs_capacity_historical, annualized_costs_energy_capacity_historical = eoles.utils.annualized_costs_investment_historical(
    #     existing_capa_historical_y, annuity_fOM_historical,
    #     existing_energy_capacity_historical_y, storage_annuity_historical)
    # annualized_costs_new_capacity = pd.DataFrame(0, index=existing_capacity_historical.index, columns=["annualized_costs"], dtype=float)
    # annualized_costs_new_energy_capacity = pd.DataFrame(0, index=existing_energy_capacity_historical.index, columns=["annualized_costs"], dtype=float)
    #
    # ### Compile total annualized investment costs from existing capacities (both historical capacities + newly built capacities before t)
    # # Necessary for calculus of LCOE accounting for evolution of capacities
    # annualized_costs_capacity = pd.concat(
    #     [annualized_costs_capacity_historical.rename(columns={'annualized_costs': 'historical_annualized_costs'}),
    #      annualized_costs_new_capacity], axis=1)
    # annualized_costs_capacity['annualized_costs'] = annualized_costs_capacity['historical_annualized_costs'] + \
    #                                                 annualized_costs_capacity['annualized_costs']
    # annualized_costs_energy_capacity = pd.concat([annualized_costs_energy_capacity_historical.rename(
    #     columns={'annualized_costs': 'historical_annualized_costs'}), annualized_costs_new_energy_capacity], axis=1)
    # annualized_costs_energy_capacity['annualized_costs'] = annualized_costs_energy_capacity[
    #                                                            'historical_annualized_costs'] + \
    #                                                        annualized_costs_energy_capacity['annualized_costs']
    #
    # existing_annualized_costs_elec, existing_annualized_costs_CH4, existing_annualized_costs_H2 = eoles.utils.process_annualized_costs_per_vector(
    #     annualized_costs_capacity[["annualized_costs"]].squeeze(),
    #     annualized_costs_energy_capacity[["annualized_costs"]].squeeze())
    #
    # ### Create additional gas profile (tertiary heating + ECS)
    # heating_gas_demand = heating_gas_demand_RTE_timesteps[anticipated_year_eoles] * 1e3  # in TWh
    # ECS_gas_demand = ECS_gas_demand_RTE_timesteps[anticipated_year_eoles] * 1e3  # in TWh
    # ECS_demand_hourly = ECS_gas_demand / 8760
    # hourly_gas = eoles.utils.create_hourly_residential_demand_profile(total_consumption=heating_gas_demand,
    #                                                                   method=HOURLY_PROFILE_METHOD)  # value for gas heating demand in tertiary sector
    # hourly_ECS = pd.Series(ECS_demand_hourly, index=hourly_gas.index)
    # hourly_exogeneous_CH4 = hourly_gas + hourly_ECS
    #
    # # Find optimal subsidy
    # optimizer, opt_sub = \
    #     optimize_blackbox_resirf_eoles_coupling(buildings, energy_prices, taxes, cost_heater, cost_insulation,
    #                                             flow_built, post_inputs, start_year_resirf, timestep_resirf,
    #                                             config_eoles, year_eoles, anticipated_year_eoles, 180,
    #                                             hourly_gas_exogeneous=hourly_exogeneous_CH4,
    #                                             existing_capacity=existing_capacity,
    #                                             existing_charging_capacity=existing_charging_capacity,
    #                                             existing_energy_capacity=existing_energy_capacity,
    #                                             maximum_capacity=maximum_capacity, method_hourly_profile="valentin",
    #                                             scenario_cost=scenario_cost,
    #                                             existing_annualized_costs_elec=existing_annualized_costs_elec,
    #                                             existing_annualized_costs_CH4=existing_annualized_costs_CH4,
    #                                             existing_annualized_costs_H2=existing_annualized_costs_H2,
    #                                             lifetime_renov=40, discount_rate_renov=0.045, plot=False)
    #
    # optimizer.plot_acquisition()
    # optimizer.plot_convergence()

    # from pylab import grid
    # Xdata = optimizer.X
    # best_Y = optimizer.Y_best
    # n = Xdata.shape[0]
    # aux = (Xdata[1:n,:]-Xdata[0:n-1,:])**2
    # distances = np.sqrt(aux.sum(axis=1))
    #
    # ## Distances between consecutive x's
    # plt.figure(figsize=(10,5))
    # plt.subplot(1, 2, 1)
    # plt.plot(list(range(n-1)), distances, '-ro')
    # plt.xlabel('Iteration')
    # plt.ylabel('d(x[n], x[n-1])')
    # plt.title('Distance between consecutive x\'s')
    # grid(True)
    #
    # # Estimated m(x) at the proposed sampling points
    # plt.subplot(1, 2, 2)
    # plt.plot(list(range(n)),best_Y,'-o')
    # plt.title('Value of the best selected sample')
    # plt.xlabel('Iteration')
    # plt.ylabel('Best y')
    # grid(True)
    # plt.show()

    # # Tester l'impact du choix du nombre d'itérations sur la convergence
    # conv = {}
    # max_iter, initial_design_numdata, optimizer = test_convergence(max_iter=20, initial_design_numdata=3)
    # conv[20] = optimizer
    # max_iter, initial_design_numdata, optimizer = test_convergence(max_iter=30, initial_design_numdata=3)
    # conv[30] = optimizer
    # max_iter, initial_design_numdata, optimizer = test_convergence(max_iter=10, initial_design_numdata=3)
    # conv[10] = optimizer
    #
    # optimizer.plot_acquisition()
    #
    from GPyOpt.plotting.plots_bo import plot_acquisition
    #
    # sns.set_theme()
    plot_acquisition([(0, 1), (0, 0.2)],
                     optimizer.model.model.X.shape[1],
                     optimizer.model.model,
                     optimizer.model.model.X,
                     optimizer.model.model.Y,
                     optimizer.acquisition.acquisition_function,
                     optimizer.suggest_next_locations())

    output, optimizer = test_convergence_2030()
