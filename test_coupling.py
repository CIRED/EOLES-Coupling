import json
import os
from importlib import resources

import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from pickle import dump, load

from project.coupling import ini_res_irf, simu_res_irf
from project.building import AgentBuildings
from project.model import get_inputs, social_planner

from project.model import create_logger, get_config, get_inputs
from eoles.model_resirf_coupling import ModelEOLES
from eoles.utils import get_config, get_pandas, calculate_annuities_resirf
from eoles.write_output import plot_simulation
import eoles.utils
from eoles.coupling_resirf_eoles import resirf_eoles_coupling_dynamic, optimize_blackbox_resirf_eoles_coupling, resirf_eoles_coupling_dynamic_no_opti
import logging
import argparse
import copy


from matplotlib import pyplot as plt

LOG_FORMATTER = '%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# consoler handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
logger.addHandler(console_handler)

HOURLY_PROFILE_METHOD = "valentin"
DICT_CONFIG_RESIRF = {
    "classic": "eoles/inputs/config/config_resirf.json",
    "nolandlord": "eoles/inputs/config/config_resirf_nolandlord.json",
    "nomultifamily": "eoles/inputs/config/config_resirf_nomultifamily.json",
    "nolandlord_nomultifamily": "eoles/inputs/config/config_resirf_nolandlord_nomultifamily.json",
    "threshold": "eoles/inputs/config/config_resirf_threshold.json",
    "classic_simple": "eoles/inputs/config/config_resirf_simple.json",
    "nolandlord_simple": "eoles/inputs/config/config_resirf_nolandlord_simple.json",
    "nomultifamily_simple": "eoles/inputs/config/config_resirf_nomultifamily_simple.json",
    "nolandlord_nomultifamily_simple": "eoles/inputs/config/config_resirf_nolandlord_nomultifamily_simple.json",
    "threshold_simple": "eoles/inputs/config/config_resirf_threshold_simple.json",
}

DICT_CONFIG_EOLES = {
    "eoles_classic": "eoles_coupling",
    "eoles_worst_case": "eoles_coupling_worst_case"
}


def test_convergence(max_iter, initial_design_numdata, buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built,
                                            post_inputs, policies_heater, policies_insulation, add_CH4_demand=False):
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

    if add_CH4_demand:
        ### Create additional gas profile (tertiary heating + ECS)
        heating_gas_demand = heating_gas_demand_RTE_timesteps[anticipated_year_eoles] * 1e3  # in TWh
        ECS_gas_demand = ECS_gas_demand_RTE_timesteps[anticipated_year_eoles] * 1e3  # in TWh
        ECS_demand_hourly = ECS_gas_demand / 8760
        hourly_gas = eoles.utils.create_hourly_residential_demand_profile(total_consumption=heating_gas_demand,
                                                                          method=HOURLY_PROFILE_METHOD)  # value for gas heating demand in tertiary sector
        hourly_ECS = pd.Series(ECS_demand_hourly, index=hourly_gas.index)
        hourly_exogeneous_CH4 = hourly_gas + hourly_ECS
    else:
        hourly_exogeneous_CH4 = eoles.utils.create_hourly_residential_demand_profile(total_consumption=0,
                                                                          method=HOURLY_PROFILE_METHOD)

    # Find optimal subsidy
    optimizer, opt_sub = \
        optimize_blackbox_resirf_eoles_coupling(buildings, energy_prices, taxes, cost_heater, cost_insulation,
                                                flow_built, post_inputs, policies_heater, policies_insulation,
                                                start_year_resirf, timestep_resirf,
                                                config_eoles, year_eoles, anticipated_year_eoles, scc=180,
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


# def test_convergence_2030():
#     output, optimizer = resirf_eoles_coupling_dynamic(_buildings, _energy_prices, _taxes, _cost_heater, _cost_insulation, _flow_built,
#                                   _post_inputs, [2025, 2030], [180, 250], scenario_cost, config_eoles, max_iter=22, return_optimizer=True)
#     return output, optimizer

if __name__ == '__main__':

    # config_eoles = eoles.utils.get_config(spec="eoles_coupling")
    # hourly_heat_elec = eoles.utils.create_hourly_residential_demand_profile(total_consumption=45,
    #                                                      method=HOURLY_PROFILE_METHOD)
    # hourly_heat_gas = eoles.utils.create_hourly_residential_demand_profile(total_consumption=95,
    #                                                      method=HOURLY_PROFILE_METHOD)
    # m_eoles = ModelEOLES(name="trajectory", config=config_eoles, path="eoles/outputs", logger=logger, nb_years=1,
    #                      hourly_heat_elec=hourly_heat_elec, hourly_heat_gas=hourly_heat_gas,
    #                      wood_consumption=0, oil_consumption=0,
    #                      existing_capacity=None, existing_charging_capacity=None,
    #                      existing_energy_capacity=None, maximum_capacity=None,
    #                      method_hourly_profile="valentin",
    #                      social_cost_of_carbon=100, year=2050, anticipated_year=2050,
    #                      scenario_cost=None, carbon_constraint=False)
    # m_eoles.build_model()
    # solver_results, status, termination_condition = m_eoles.solve(solver_name="gurobi")

    config_coupling = {
        'config_resirf': "classic",
        'calibration_threshold': False,
        'h2ccgt': False,
        'max_iter': 1,
        'sub_design': "global_renovation_low_income",
        "health": True,
        "discount_rate": 0.045,
        "rebound": True,
        "carbon_constraint": True,
        'one_shot_setting': True,
        'fix_sub_heater': False,
        'list_year': [2025],
        'list_trajectory_scc': [775],
         'scenario_cost_eoles': {
            'biomass_potential': {
                'methanization': 0,
                'pyrogazification': 0
            },
            'maximum_capacity': {
                'offshore_g': 10,
                'offshore_f': 20,
                'nuclear': 25,
                'onshore': 70,
                'pv_g': 50,
                'pv_c': 50
            },
            'existing_capacity': {
                'offshore_g': 0,
                'offshore_f': 0,
                'nuclear': 0,
                'onshore': 0,
                'pv_g': 0,
                'pv_c': 0
            },
            'vOM': {
                'natural_gas': 0.035,
            }
        }
    }

    config_coupling = {
        'config_resirf': "classic_simple",
        "config_eoles": "eoles_classic",  # includes costs assumptions
        'calibration_threshold': False,
        'h2ccgt': False,
        'max_iter': 1,
        'sub_design': "global_renovation",
        "health": True,  # on inclut les coûts de santé
        "discount_rate": 0.032,
        "rebound": True,
        "carbon_constraint": False,
        'one_shot_setting': False,
        'fix_sub_heater': False,
        'list_year': [2025, 2030],
        'list_trajectory_scc': [250, 350],
        'scenario_cost_eoles': {  # add assumptions on available technologies
            "biomass_potential": {
                "methanization": 0,
                "pyrogazification": 0
            },
            "existing_capacity": {
                "offshore_f": 0
            },
            "maximum_capacity": {
                "offshore_f": 0
            }
        }
    }

    config_resirf, config_eoles_spec = config_coupling["config_resirf"], DICT_CONFIG_EOLES[config_coupling["config_eoles"]]
    config_resirf_path = DICT_CONFIG_RESIRF[config_resirf]

    calibration_threshold = config_coupling["calibration_threshold"]

    # Calibration: whether we use threshold or not
    name_calibration = 'calibration'
    if calibration_threshold is True:
        name_calibration = '{}_threshold'.format(name_calibration)
    print(name_calibration)

    export_calibration = os.path.join('eoles', 'outputs', 'calibration', '{}.pkl'.format(name_calibration))
    import_calibration = os.path.join('eoles', 'outputs', 'calibration', '{}.pkl'.format(name_calibration))

    # initialization
    buildings, energy_prices, taxes, cost_heater, cost_insulation, lifetime_heater, demolition_rate, flow_built, post_inputs, policies_heater, policies_insulation, technical_progress, financing_cost = ini_res_irf(
        path=os.path.join('eoles', 'outputs', 'ResIRF'),
        config=config_resirf_path)

    list_year = config_coupling["list_year"]
    list_trajectory_scc = config_coupling["list_trajectory_scc"]  # SCC trajectory
    config_eoles = eoles.utils.get_config(spec=config_eoles_spec)  # TODO: changer le nom de la config qu'on appelle

    h2ccgt = config_coupling["h2ccgt"]
    scenario_cost = config_coupling["scenario_cost_eoles"]

    if not h2ccgt:  # we do not allow h2 ccgt plants
        if "fix_capa" in scenario_cost.keys():
            scenario_cost["fix_capa"]["h2_ccgt"] = 0
        else:
            scenario_cost["fix_capa"] = {
                "h2_ccgt": 0
            }

    max_iter, one_shot_setting, fix_sub_heater = config_coupling["max_iter"], config_coupling["one_shot_setting"], config_coupling["fix_sub_heater"]

    sub_design, health, discount_rate, rebound, carbon_constraint = config_coupling["sub_design"], config_coupling["health"], \
                                                                    config_coupling["discount_rate"], config_coupling["rebound"], config_coupling["carbon_constraint"]

    output, dict_optimizer = resirf_eoles_coupling_dynamic(buildings, energy_prices, taxes, cost_heater, cost_insulation,
                                                           demolition_rate, flow_built, post_inputs, policies_heater, policies_insulation,
                                                           list_year, list_trajectory_scc, scenario_cost,
                                                           config_eoles=config_eoles, max_iter=max_iter,
                                                           add_CH4_demand=False, one_shot_setting=one_shot_setting,
                                                           fix_sub_heater=fix_sub_heater, sub_design=sub_design,
                                                           health=health, carbon_constraint=carbon_constraint,
                                                           lifetime_heater=lifetime_heater,
                                                           discount_rate=discount_rate, rebound=rebound,
                                                           technical_progress=technical_progress, financing_cost=financing_cost,
                                                           optimization=True, list_sub_heater=[1, 1, 1, 1, 1],
                                                           list_sub_insulation=[0.5, 0.5, 0.5, 0.5, 0.5])


    # output = resirf_eoles_coupling_dynamic_no_opti(list_sub_heater=[1.0, 0.68], list_sub_insulation=[0.23, 0.40],
    #                                                                buildings=buildings, energy_prices=energy_prices,
    #                                                                taxes=taxes, cost_heater=cost_heater, cost_insulation=cost_insulation,
    #                                                                flow_built=flow_built, post_inputs=post_inputs, policies_heater=policies_heater,
    #                                                                policies_insulation=policies_insulation, list_year=list_year,
    #                                                                list_trajectory_scc=list_trajectory_scc, scenario_cost=scenario_cost,
    #                                                                config_eoles=config_eoles, add_CH4_demand=False,
    #                                                                 one_shot_setting=one_shot_setting, sub_design=sub_design,
    #                                                                health=health, carbon_constraint=carbon_constraint, discount_rate=discount_rate,
    #                                                                rebound=rebound, technical_progress=technical_progress, financing_cost=None)

    # # TEST
    timestep = 2
    year = 2025
    start = year
    end = year + timestep

    sub_heater = 0.5
    sub_insulation = 0.8333

    buildings._debug_mode = False
    output, consumption = simu_res_irf(buildings, sub_heater, sub_insulation, start, end, energy_prices, taxes,
                 cost_heater, cost_insulation, lifetime_heater, flow_built, post_inputs, policies_heater, policies_insulation, climate=2006,
                 smooth=False, efficiency_hour=True, output_consumption=True, full_output=True, sub_design=None, technical_progress=technical_progress,
                                       financing_cost=financing_cost)

    #
    # list_year = [2025, 2030, 2035, 2040, 2045]
    # list_trajectory_scc = [180, 250, 350, 500, 650]  # SCC trajectory from Quinet
    # config_eoles = eoles.utils.get_config(spec="greenfield")  # TODO: changer le nom de la config qu'on appelle
    #
    # scenario_cost = {}
    # if not h2ccgt:  # we do not allow h2 ccgt plants
    #     scenario_cost["fix_capa"] = {
    #         "h2_ccgt": 0
    #     }
    # print(scenario_cost)

    # max_iter, initial_design_numdata, optimizer = test_convergence(max_iter=20, initial_design_numdata=3, buildings=_buildings, energy_prices=_energy_prices,
    #                  taxes=_taxes, cost_heater=_cost_heater, cost_insulation=_cost_insulation, flow_built=_flow_built,
    #                  post_inputs=_post_inputs, policies_heater=_policies_heater, policies_insulation=_policies_insulation,
    #                  add_CH4_demand=False)

    # output = resirf_eoles_coupling_dynamic(_buildings, _energy_prices, _taxes, _cost_heater, _cost_insulation, _flow_built,
    #                                        _post_inputs, _policies_heater, _policies_insulation,
    #                                        list_year, list_trajectory_scc, scenario_cost, config_eoles=config_eoles,
    #                                        max_iter=20, add_CH4_demand=False, return_optimizer=False)
    #
    # # Save results
    # date = datetime.datetime.now().strftime("%m%d%H%M")
    # if h2ccgt:
    #     export_results = os.path.join("eoles", "outputs", f'{date}_{config}_h2ccgt')
    # else:
    #     export_results = os.path.join("eoles", "outputs", f'{date}_{config}')
    #
    # if not os.path.isdir(export_results):
    #     os.mkdir(export_results)
    #
    # with open(os.path.join(export_results, 'coupling_results.pkl'), "wb") as file:
    #     dump(output, file)
    #
    # with open(os.path.join(export_results, 'config_eoles.json'), "w") as outfile:
    #     outfile.write(json.dumps(config_eoles, indent=4))
    #
    # # read and save the actual config file
    # with open(config_res_irf_path) as file:
    #     config_res_irf = json.load(file)
    #
    # with open(os.path.join(export_results, 'config_resirf.json'), "w") as outfile:
    #     outfile.write(json.dumps(config_res_irf, indent=4))
    #
    # with open(os.path.join(export_results, 'scenario_eoles_costs.json'), "w") as outfile:
    #     outfile.write(json.dumps(scenario_cost, indent=4))
    #
    # plot_simulation(output, save_path=export_results)

    # ### Study trajectory with given list of subsidies: to rerun quickly the result from a given scenario
    # list_sub_heater = [1.0, 1.0, 1.0, 1.0, 1.0]
    # list_sub_insulation = [0.0, 0.0, 0.0, 0.0, 0.0]
    # output = resirf_eoles_coupling_dynamic_no_opti(list_sub_heater, list_sub_insulation, _buildings, _energy_prices, _taxes,
    #                                       _cost_heater, _cost_insulation, _flow_built, _post_inputs, _policies_heater, _policies_insulation,
    #                                       list_year, list_trajectory_scc, scenario_cost, config_eoles, add_CH4_demand=False)

    #
    # Test convergence of the result
    # max_iter, initial_design_numdata, optimizer = test_convergence(max_iter=20, initial_design_numdata=3)
    #
    # optimizer.plot_acquisition()
    # optimizer.plot_convergence()


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
    # #
    # # sns.set_theme()
    # plot_acquisition([(0, 1), (0, 0.2)],
    #                  optimizer.model.model.X.shape[1],
    #                  optimizer.model.model,
    #                  optimizer.model.model.X,
    #                  optimizer.model.model.Y,
    #                  optimizer.acquisition.acquisition_function,
    #                  optimizer.suggest_next_locations())
    #
    # output, optimizer = test_convergence_2030()
    # optimizer.plot_acquisition()
    # plot_acquisition([(0.5, 1), (0, 0.5)], 2, optimizer.model.model, optimizer.model.model.X, optimizer.model.model.Y,
    #                  optimizer.acquisition.acquisition_function, optimizer.suggest_next_locations(),
    #                  filename=None, label_x=None, label_y=None, color_by_step=True)
