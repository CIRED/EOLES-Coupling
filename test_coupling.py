import json
import os
from importlib import resources

import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from pickle import dump, load

from project.coupling import ini_res_irf, simu_res_irf
from project.write_output import plot_scenario, plot_compare_scenarios
from project.building import AgentBuildings
from project.model import get_inputs, social_planner

from project.model import create_logger, get_config, get_inputs
from eoles.model_resirf_coupling import ModelEOLES
from eoles.utils import get_config, get_pandas, calculate_annuities_resirf, modif_config_resirf
from eoles.write_output import plot_simulation
from eoles.coupling_resirf_eoles import resirf_eoles_coupling_dynamic, optimize_blackbox_resirf_eoles_coupling, \
    calibration_price, get_energy_prices_and_taxes, resirf_eoles_coupling_greenfield, gradient_descent
import logging
from project.main import run
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
    "threshold": "eoles/inputs/config/config_resirf_threshold.json",
    "classic_simple": "eoles/inputs/config/config_resirf_simple.json",
    "threshold_simple": "eoles/inputs/config/config_resirf_threshold_simple.json",
    "classic_simple_premature3": "eoles/inputs/config/config_resirf_simple_premature3.json",
    "classic_simple_premature5": "eoles/inputs/config/config_resirf_simple_premature5.json",
    "classic_simple_premature10": "eoles/inputs/config/config_resirf_simple_premature10.json",
    "threshold_simple_premature3": "eoles/inputs/config/config_resirf_threshold_simple_premature3.json",
    "nolandlord": "eoles/inputs/config/config_resirf_nolandlord.json",
    "nomultifamily": "eoles/inputs/config/config_resirf_nomultifamily.json",
    "nolandlord_nomultifamily": "eoles/inputs/config/config_resirf_nolandlord_nomultifamily.json",
    "nolandlord_simple": "eoles/inputs/config/config_resirf_nolandlord_simple.json",
    "nomultifamily_simple": "eoles/inputs/config/config_resirf_nomultifamily_simple.json",
    "nolandlord_nomultifamily_simple": "eoles/inputs/config/config_resirf_nolandlord_nomultifamily_simple.json",
    "test": "eoles/inputs/config/config_coupling_test.json"
}

DICT_CONFIG_EOLES = {
    "eoles_classic": "eoles_coupling",
    "eoles_biogasS2": "eoles_coupling_biogasS2",
    "eoles_nobiogas": "eoles_coupling_nobiogas",
    "eoles_nobiogas_nohydrogen": "eoles_coupling_nobiogas_nohydrogen",
    "eoles_worst_case": "eoles_coupling_worst_case"
}


def test_convergence(max_iter, initial_design_numdata, buildings, energy_prices, taxes, cost_heater, cost_insulation,
                     flow_built,
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

    config_coupling = {
        'config_resirf': "classic_simple_premature3",
        "config_eoles": "eoles_classic",  # includes costs assumptions
        'calibration_threshold': False,
        'h2ccgt': True,
        'max_iter': 100,
        'sub_design': None,
        "health": True,  # on inclut les coûts de santé
        "discount_rate": 0.032,
        "rebound": True,
        "carbon_constraint": False,
        'one_shot_setting': False,
        'fix_sub_heater': False,
        'list_year': [2025, 2030, 2035, 2040, 2045],
        'list_trajectory_scc': [250, 350, 500, 650, 775],
        'scenario_cost_eoles': {}
    }

    config_coupling = {
        'config_resirf': "classic_simple_premature3",
        "config_eoles": "eoles_classic",  # includes costs assumptions
        'calibration_threshold': False,
        'h2ccgt': True,
        'max_iter': 30,
        'sub_design': None,
        "health": True,  # on inclut les coûts de santé
        "discount_rate": 0.032,
        "rebound": True,
        "carbon_constraint": False,
        'one_shot_setting': False,
        'fix_sub_heater': False,
        'fix_sub_insulation': True,
        'list_year': [2025, 2030, 2035, 2040, 2045],
        'list_trajectory_scc': [250, 350, 500, 650, 775],
        'acquisition_jitter': 0.03,
        'price_feedback': False,
        'scenario_cost_eoles': {}
    }

    config_coupling = {
        'config_resirf': "classic_simple",
        'calibration': None,
        "config_eoles": "eoles_classic",  # includes costs assumptions
        'supply_insulation': False,
        'supply_heater': False,
        'rational_behavior': False,
        'social': False,
        'premature_replacement': 3,
        'h2ccgt': True,
        'max_iter': 30,
        'sub_design': "global_renovation",
        "no_MF": False,
        "health": True,  # on inclut les coûts de santé
        "discount_rate": 0.032,
        "rebound": True,
        "carbon_constraint": True,
        'one_shot_setting': False,
        'fix_sub_heater': False,
        'fix_sub_insulation': False,
        'list_year': [2025, 2030, 2035, 2040, 2045],
        'list_trajectory_scc': [250, 350, 500, 650, 775],
        'acquisition_jitter': 0.03,
        'scenario_cost_eoles': {}
    }

    config_resirf_path, config_eoles_spec = DICT_CONFIG_RESIRF[config_coupling["config_resirf"]], DICT_CONFIG_EOLES[config_coupling["config_eoles"]]
    with open(config_resirf_path) as file:  # load config_resirf
        config_resirf = json.load(file).get('Reference')
    config_resirf = modif_config_resirf(config_resirf, config_coupling, calibration=config_coupling["calibration"])  # modif of this configuration file to consider coupling options
    # config_resirf = modif_config_resirf(config_resirf, config_coupling, calibration=None)  # modif of this configuration file to consider coupling options
    # print(config_resirf)

    # # Calibration: whether we use threshold or not
    # name_calibration = 'calibration'
    # if calibration_threshold is True:
    #     name_calibration = '{}_threshold'.format(name_calibration)
    # print(name_calibration)
    #
    # export_calibration = os.path.join('eoles', 'outputs', 'calibration', '{}.pkl'.format(name_calibration))
    # import_calibration = os.path.join('eoles', 'outputs', 'calibration', '{}.pkl'.format(name_calibration))

    # initialization ResIRF
    buildings, inputs_dynamics, policies_heater, policies_insulation = ini_res_irf(
        path=os.path.join('eoles', 'outputs', 'ResIRF'),
        config=config_resirf)

    # # TEST for a given time step
    # timestep = 5
    # year = 2020
    # start = year
    # end = year + timestep
    #
    # sub_heater = 0.5763
    # sub_insulation = 0.6502

    # output, stock, heating_consumption = simu_res_irf(buildings=buildings, sub_heater=None,
    #                                                   sub_insulation=None,
    #                                                   start=start,
    #                                                   end=end, energy_prices=inputs_dynamics['energy_prices'],
    #                                                   taxes=inputs_dynamics['taxes'],
    #                                                   cost_heater=inputs_dynamics['cost_heater'], cost_insulation=inputs_dynamics['cost_insulation'],
    #                                                   lifetime_heater=20, flow_built=inputs_dynamics['flow_built'],
    #                                                   post_inputs=inputs_dynamics['post_inputs'], policies_heater=policies_heater,
    #                                                   policies_insulation=policies_insulation,
    #                                                   climate=2006, smooth=False, efficiency_hour=True,
    #                                                   demolition_rate=inputs_dynamics['demolition_rate'],
    #                                                   output_consumption=True,
    #                                                   full_output=True,
    #                                                   sub_design=config_coupling["sub_design"],
    #                                                   rebound=config_coupling["rebound"],
    #                                                   technical_progress=inputs_dynamics['technical_progress'],
    #                                                   financing_cost=inputs_dynamics['financing_cost'], premature_replacement=inputs_dynamics['premature_replacement'],
    #                                                   supply=inputs_dynamics['supply'])
    #
    # elec_consumption = heating_consumption.T["Electricity"]
    #
    # buildings.path = os.path.join("eoles/outputs/test_plots/plots_resirf")
    # plot_scenario(output, stock, buildings)
    # plot_compare_scenarios(result={"Reference": output}, folder=os.path.join("eoles/outputs/test_plots"))


    list_year = config_coupling["list_year"]
    list_trajectory_scc = config_coupling["list_trajectory_scc"]  # SCC trajectory
    config_eoles = get_config(spec=config_eoles_spec)  # TODO: changer le nom de la config qu'on appelle

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
    price_feedback = False
    if "price_feedback" in config_coupling.keys():
        price_feedback = config_coupling["price_feedback"]

    energy_prices_ht, energy_taxes = get_energy_prices_and_taxes(config_resirf)
    calibration_elec_lcoe, calibration_elec_transport_distrib, calibration_gas, m_eoles = calibration_price(config_eoles, scc=100)
    config_coupling["calibration_elec_lcoe"] = calibration_elec_lcoe
    config_coupling["calibration_elec_transport_distrib"] = calibration_elec_transport_distrib
    config_coupling["calibration_naturalgas_lcoe"] = calibration_gas
    config_coupling["calibration_biogas_lcoe"] = 1.2
    # config_coupling["calibration_gas_lcoe"] = calibration_gas

    # output, buildings, dict_optimizer = resirf_eoles_coupling_dynamic(buildings, inputs_dynamics,
    #                                                                   policies_heater, policies_insulation,
    #                                                                   list_year, list_trajectory_scc, scenario_cost,
    #                                                                   config_eoles=config_eoles,
    #                                                                   config_coupling=config_coupling,
    #                                                                   add_CH4_demand=False,
    #                                                                   optimization=True,
    #                                                                   price_feedback=price_feedback,
    #                                                                   energy_prices_ht=energy_prices_ht,
    #                                                                   energy_taxes=energy_taxes,
    #                                                                   acquisition_jitter=0.03)

    # # Test sensitivity to subsidy
    # sensitivity_subsidy = {
    #     "sub_100": 1.0,
    #     "sub_80": 0.8,
    #     "sub_20": 0.2,
    #     "sub_40": 0.4,
    #     "sub_60": 0.6
    # }
    # results = {}
    # for config in list(sensitivity_subsidy.keys()):
    #     print(config)
    #     export_results = os.path.join("eoles/outputs/subsidy_sensitivity", config)
    #     os.mkdir(export_results)
    #     os.mkdir(os.path.join(export_results, "plots"))
    #     subsidy = sensitivity_subsidy[config]
    #     # initialization
    #     buildings, inputs_dynamics, policies_heater, policies_insulation = ini_res_irf(
    #         path=os.path.join('eoles', 'outputs', 'ResIRF'),
    #         config=config_resirf)
    #     output, buildings, dict_optimizer = resirf_eoles_coupling_dynamic(buildings, inputs_dynamics,
    #                                                                       policies_heater, policies_insulation,
    #                                                                       list_year, list_trajectory_scc, scenario_cost,
    #                                                                       config_eoles=config_eoles,
    #                                                                       config_coupling=config_coupling,
    #                                                                       add_CH4_demand=False,
    #                                                                       optimization=False, list_sub_heater=[0.0, 0.0],
    #                                                                       list_sub_insulation=[subsidy, subsidy],
    #                                                                       price_feedback=price_feedback,
    #                                                                       energy_prices_ht=energy_prices_ht,
    #                                                                       energy_taxes=energy_taxes,
    #                                                                       acquisition_jitter=0.03)
    #     buildings.path = os.path.join(export_results, "plots")
    #     plot_scenario(output["Output global ResIRF ()"], output["Stock global ResIRF ()"],
    #                   buildings)  # make ResIRF plots
    #     results[config] = output
    # results_resirf = {}
    # for key in results.keys():
    #     results_resirf[key] = results[key]["Output global ResIRF ()"]
    # plot_compare_scenarios(result=results_resirf, folder=os.path.join("eoles/outputs/subsidy_sensitivity"))

    # list_sub_heater = [0.0, 0.0, 0.6864, 0.6725, 0.967]
    # list_sub_insulation = [0.764599, 0.78320, 0.8085, 0.8908, 0.09]

    output, buildings, dict_optimizer = resirf_eoles_coupling_dynamic(buildings, inputs_dynamics,
                                                                      policies_heater, policies_insulation,
                                                                      list_year, list_trajectory_scc, scenario_cost,
                                                                      config_eoles=config_eoles,
                                                                      config_coupling=config_coupling,
                                                                      add_CH4_demand=False,
                                                                      optimization=False, list_sub_heater=[0.775598, 0.0, 0.84839800, 0.88300, 0.92762],
                                                                      list_sub_insulation=[0.6363841, 0.831483, 0.765298, 0.50764, 0.239],
                                                                      price_feedback=price_feedback,
                                                                      energy_prices_ht=energy_prices_ht,
                                                                      energy_taxes=energy_taxes,
                                                                      acquisition_jitter=0.03, grad_descent=False)

    # gradient_descent(x0=[0.963, 0.09], buildings=buildings, inputs_dynamics=inputs_dynamics, policies_heater=policies_heater,
    #                  policies_insulation=policies_insulation, start_year_resirf=2045, timestep_resirf=5,
    #                  config_eoles=config_eoles, year_eoles=2045, anticipated_year_eoles=2050, scc=750, hourly_gas_exogeneous,
    #                  existing_capacity, existing_charging_capacity, existing_energy_capacity, maximum_capacity,
    #                  method_hourly_profile, scenario_cost, existing_annualized_costs_elec=0, existing_annualized_costs_CH4=0,
    #                  existing_annualized_costs_H2=0, lifetime_renov=50, lifetime_heater=20, discount_rate=0.045,
    #                  max_iter=20, sub_design=None, health=True, carbon_constraint=False, rebound=True)


    # output, buildings, dict_optimizer = resirf_eoles_coupling_greenfield(buildings, energy_prices, taxes, cost_heater, cost_insulation,
    #                                                        demolition_rate, flow_built, post_inputs, policies_heater, policies_insulation,
    #                                                        scc=775, scenario_cost=scenario_cost,
    #                                                        config_eoles=config_eoles, config_coupling=config_coupling,
    #                                                        add_CH4_demand=False,
    #                                                        technical_progress=technical_progress, financing_cost=financing_cost,
    #                                                        premature_replacement=premature_replacement,
    #                                                        optimization=False, list_sub_heater=[0.2],
    #                                                        list_sub_insulation=[0.3])

    # # Test plot optimization blackbox
    # import matplotlib.tri as tri
    # evaluations = pd.read_csv(os.path.join("eoles/outputs/0309_120024_temoin_simple_premature3/evaluations_optimizer_2045.csv"), sep='\t')
    # evaluations = evaluations.loc[evaluations.Y < 40.7]
    #
    # x = evaluations["var_1"].to_numpy()
    # y = evaluations["var_2"].to_numpy()
    # z = evaluations["Y"].to_numpy()
    #
    # fig, ax = plt.subplots()
    # ax.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
    # cntr2 = ax.tricontourf(x, y, z, levels=14, cmap="RdBu_r")
    #
    # fig.colorbar(cntr2, ax=ax)
    # ax.plot(x, y, 'ko', ms=3)
    # plt.show()

    #
    # plot_simulation(output, save_path=os.path.join("eoles/outputs/test_plots", "plots"))
    # buildings.path = os.path.join("eoles/outputs/test_plots/")
    # plot_scenario(output["Output global ResIRF ()"], output["Stock global ResIRF ()"], buildings)

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

    # # TEST for a given time step
    # timestep = 10
    # year = 2020
    # start = year
    # end = year + timestep
    #
    # sub_heater = 0.9586
    # sub_insulation = 0.585141
    #
    # output, stock, heating_consumption = simu_res_irf(buildings=buildings, sub_heater=sub_heater, sub_insulation=sub_insulation,
    #                                            start=start, end=end, energy_prices=energy_prices,
    #                                            taxes=taxes, cost_heater=cost_heater, cost_insulation=cost_insulation,
    #                                            lifetime_heater=lifetime_heater, demolition_rate=demolition_rate, flow_built=flow_built,
    #                                            post_inputs=post_inputs, policies_heater=policies_heater,
    #                                            policies_insulation=policies_insulation,
    #                                            sub_design=None, financing_cost=financing_cost, climate=2006, smooth=False, efficiency_hour=True,
    #                                            output_consumption=True, full_output=True, rebound=True,
    #                                            technical_progress=technical_progress)
    # buildings.path = os.path.join("eoles/outputs/test_plots/plots_resirf")
    # plot_scenario(output, stock, buildings)
    # grouped_output(result={"Reference": output}, folder=os.path.join("eoles/outputs/test_plots"))

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

    # ################## Projection of transport and distribution costs #############
    #
    # from sklearn.linear_model import LinearRegression
    # transport_and_distrib = pd.DataFrame(
    #     data={'M0': [5, 2, 2, 0, 0, 19], 'M1': [5, 2, 2, 0, 0, 19], 'M23': [5, 3, 2, 0, 0, 17],
    #           'N1': [5, 2, 2, 0, 0, 17], 'N2': [5, 1, 1, 0, 0, 16], 'N03': [5, 1, 1, 0, 0, 16]},
    #     index=["Transport - réseau existant et renouvellement", "Transport - raccordement éolien en mer",
    #            "Transport - adaptation", "Transport - interconnexions", "Transport - autre", "Distribution"])
    # transport_and_distrib = transport_and_distrib.T
    # transport_and_distrib["transport"] = transport_and_distrib["Transport - réseau existant et renouvellement"] + \
    #                                      transport_and_distrib["Transport - adaptation"] + transport_and_distrib[
    #                                          "Transport - interconnexions"] + transport_and_distrib["Transport - autre"]
    # transport_and_distrib["transport_and_distribution"] = transport_and_distrib["transport"] + transport_and_distrib[
    #     "Distribution"]
    #
    # prod_renewable = pd.DataFrame(
    #     data={'M0': [208, 74, 62], 'M1': [214, 59, 45], 'M23': [125, 72, 60], 'N1': [118, 58, 45],
    #           'N2': [90, 52, 36], 'N03': [70, 43, 22]}, index=["solar", "onshore_wind", "offshore_wind"])
    # prod_renewable = prod_renewable.T
    #
    # y = transport_and_distrib[["transport_and_distribution"]].squeeze().to_numpy()
    # X = prod_renewable[["solar", "onshore_wind"]].to_numpy()
    #
    # reg = LinearRegression().fit(X, y)
    # print(f"Coefficients for predicting transport and distribution: {reg.coef_}")
    # print(f"Intercept for predicting transport and distribution: {reg.intercept_}")
    #
    # y_distrib = transport_and_distrib[["Distribution"]].squeeze().to_numpy()
    # reg_distrib = LinearRegression().fit(X, y_distrib)
    # print(f"Coefficients for predicting distribution: {reg_distrib.coef_}")
    # print(f"Intercept for predicting transport and distribution: {reg.intercept_}")
    #
    # y_offshore = transport_and_distrib[["Transport - raccordement éolien en mer"]].squeeze().to_numpy()
    # X_offshore = prod_renewable[["offshore_wind"]].to_numpy()
    # reg_offshore = LinearRegression().fit(X_offshore, y_offshore)
    # print(f"Coefficients for predicting offshore costs: {reg_offshore.coef_}")
    # print(f"Intercept for predicting transport and distribution: {reg.intercept_}")

    ################### Calibration LCOE ##############
    # #
    # config_eoles = eoles.utils.get_config(spec="eoles_coupling")
    # calibration_lcoe, m_eoles = calibration_price(config_eoles, scc=100)

    # calibration_lcoe, m_eoles = calibration_price(config_eoles, existing_capacity=None,
    #                                               existing_charging_capacity=None,
    #                                               existing_energy_capacity=None,
    #                                               maximum_capacity=None,
    #                                               existing_annualized_costs_elec=0,
    #                                               existing_annualized_costs_CH4=0,
    #                                               existing_annualized_costs_H2=0,
    #                                               scc=500)
