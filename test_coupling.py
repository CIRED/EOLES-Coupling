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
from eoles.utils import get_config, get_pandas, calculate_annuities_resirf, modif_config_resirf, \
    config_resirf_exogenous, create_multiple_coupling_configs, modif_config_eoles
from eoles.write_output import plot_simulation, comparison_simulations, comparison_simulations_scenarios
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


def run_resirf_alone(year, timestep, config_coupling, buildings, inputs_dynamics, policies_heater, policies_insulation,
                     sub_heater=None, sub_insulation=None):
    """Test to run resirf for a small time period."""
    start = year
    end = year + timestep

    output, stock, heating_consumption = simu_res_irf(buildings=buildings, sub_heater=sub_heater,
                                                      sub_insulation=sub_insulation,
                                                      start=start,
                                                      end=end, energy_prices=inputs_dynamics['energy_prices'],
                                                      taxes=inputs_dynamics['taxes'],
                                                      cost_heater=inputs_dynamics['cost_heater'],
                                                      cost_insulation=inputs_dynamics['cost_insulation'],
                                                      lifetime_heater=20, flow_built=inputs_dynamics['flow_built'],
                                                      post_inputs=inputs_dynamics['post_inputs'],
                                                      policies_heater=policies_heater,
                                                      policies_insulation=policies_insulation,
                                                      climate=2006, smooth=False, efficiency_hour=True,
                                                      demolition_rate=inputs_dynamics['demolition_rate'],
                                                      output_consumption=True,
                                                      full_output=True,
                                                      sub_design=config_coupling["sub_design"],
                                                      rebound=config_coupling["rebound"],
                                                      technical_progress=inputs_dynamics['technical_progress'],
                                                      financing_cost=inputs_dynamics['financing_cost'],
                                                      premature_replacement=inputs_dynamics['premature_replacement'],
                                                      supply=inputs_dynamics['supply'])
    return output, stock, heating_consumption


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
        'price_feedback': False,
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
        'sub_design': None,
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

    config_coupling = {
        "calibration": None,
        'eoles': {
            'aggregated_potential': True,
        },
        'supply_insulation': False,
        'supply_heater': False,
        'rational_behavior': False,
        'social': False,
        'premature_replacement': 3,
        'max_iter': 130,
        'sub_design': None,
        "health": True,  # on inclut les coûts de santé
        "discount_rate": 0.032,
        "rebound": True,
        "carbon_constraint": True,
        'fix_sub_heater': False,
        'fix_sub_insulation': False,
        'list_year': [2025, 2030, 2035, 2040, 2045],
        'list_trajectory_scc': [250, 350, 500, 650, 775],
        'acquisition_jitter': 0.03,
        'scenario_cost_eoles': {}
        }

    # TODO: a changer avec nouveau framework !
    # config_resirf_path = DICT_CONFIG_RESIRF["classic_simple"]
    # with open(config_resirf_path) as file:  # load config_resirf
    #     config_resirf = json.load(file).get('Reference')
    # config_resirf = modif_config_resirf(config_resirf, config_coupling, calibration=config_coupling[
    #     "calibration"])  # modif of this configuration file to consider coupling options
    #
    # config_eoles = get_config(spec="eoles_coupling")
    # config_eoles = modif_config_eoles(config_eoles, config_coupling)

    # ########## Test exogenous configurations
    # sensitivity = {
    #     # "no_policy": True,
    #     "policies": {
    #         "Ambitious": "project/input/policies/policies_ambitious.json",
    #         # "Reference": "project/input/policies/current/policies_ref.json"
    #     }
    # }
    # dict_config_resirf = config_resirf_exogenous(sensitivity=sensitivity, config_resirf=config_resirf)
    # dict_config_coupling = create_multiple_coupling_configs(dict_config_resirf=dict_config_resirf, config_coupling=config_coupling)

    # # initialization ResIRF
    # buildings, inputs_dynamics, policies_heater, policies_insulation = ini_res_irf(
    #     path=os.path.join('eoles', 'outputs', 'ResIRF'),
    #     config=config_resirf)
    #
    # # TEST for a given time step
    # timestep = 2
    # year = 2020
    # output, stock, heating_consumption = run_resirf_alone(year, timestep, config_coupling, buildings, inputs_dynamics,
    #                                                       policies_heater, policies_insulation, sub_heater=None, sub_insulation=None)

    #
    # elec_consumption = heating_consumption.T["Electricity"]
    #
    # buildings.path = os.path.join("eoles/outputs/test_plots/plots_resirf")
    # plot_scenario(output, stock, buildings)
    # plot_compare_scenarios(result={"Reference": output}, folder=os.path.join("eoles/outputs/test_plots"))

    list_year = config_coupling["list_year"]
    list_trajectory_scc = config_coupling["list_trajectory_scc"]  # SCC trajectory

    max_iter, fix_sub_heater = config_coupling["max_iter"], config_coupling["fix_sub_heater"]
    price_feedback = False
    if "price_feedback" in config_coupling.keys():
        price_feedback = config_coupling["price_feedback"]

    # energy_taxes, energy_vta = get_energy_prices_and_taxes(config_resirf)
    # calibration_elec_lcoe, calibration_elec_transport_distrib, calibration_gas, m_eoles = calibration_price(config_eoles, scc=100)
    # config_coupling["calibration_elec_lcoe"] = calibration_elec_lcoe
    # config_coupling["calibration_elec_transport_distrib"] = calibration_elec_transport_distrib
    # config_coupling["calibration_naturalgas_lcoe"] = calibration_gas
    # config_coupling["calibration_biogas_lcoe"] = 1.2
    # # config_coupling["calibration_gas_lcoe"] = calibration_gas
    #
    # output, buildings, dict_optimizer = resirf_eoles_coupling_dynamic(buildings, inputs_dynamics,
    #                                                                   policies_heater, policies_insulation,
    #                                                                   list_year, list_trajectory_scc, scenario_cost,
    #                                                                   config_eoles=config_eoles,
    #                                                                   config_coupling=config_coupling,
    #                                                                   add_CH4_demand=False,
    #                                                                   optimization=True,
    #                                                                   price_feedback=price_feedback,
    #                                                                   energy_taxes=energy_taxes,
    #                                                                   energy_vta=energy_vta,
    #                                                                   acquisition_jitter=0.03,
    #                                                                   aggregated_potential=True)

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

    # output, buildings, dict_optimizer = resirf_eoles_coupling_dynamic(buildings, inputs_dynamics,
    #                                                                   policies_heater, policies_insulation,
    #                                                                   list_year, list_trajectory_scc, scenario_cost,
    #                                                                   config_eoles=config_eoles,
    #                                                                   config_coupling=config_coupling,
    #                                                                   add_CH4_demand=False,
    #                                                                   optimization=False, list_sub_heater=[0.0, 0.0, 0.0, 0.0, 0.0],
    #                                                                   list_sub_insulation=[0.0, 0.0, 0.0, 0.0, 0.0],
    #                                                                   price_feedback=price_feedback,
    #                                                                   energy_taxes=energy_taxes,
    #                                                                   energy_vta=energy_vta,
    #                                                                   acquisition_jitter=0.03, grad_descent=False, cofp=False)

    # gradient_descent(x0=[0.963, 0.09], buildings=buildings, inputs_dynamics=inputs_dynamics, policies_heater=policies_heater,
    #                  policies_insulation=policies_insulation, start_year_resirf=2045, timestep_resirf=5,
    #                  config_eoles=config_eoles, year_eoles=2045, anticipated_year_eoles=2050, scc=750, hourly_gas_exogeneous,
    #                  existing_capacity, existing_charging_capacity, existing_energy_capacity, maximum_capacity,
    #                  method_hourly_profile, scenario_cost, existing_annualized_costs_elec=0, existing_annualized_costs_CH4=0,
    #                  existing_annualized_costs_H2=0, lifetime_renov=50, lifetime_heater=20, discount_rate=0.045,
    #                  max_iter=20, sub_design=None, health=True, carbon_constraint=False, rebound=True)


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

    dict_output1 = {"Uniform": os.path.join("eoles/outputs/0506_greenfield_S3_N1/0506_151129_uniform_greenfield_S3_N1"),
                   "Global renovation": os.path.join(
                       "eoles/outputs/0506_greenfield_S3_N1/0506_151216_GR_greenfield_S3_N1"),
                   "Global renovation FGE": os.path.join(
                       "eoles/outputs/0506_greenfield_S3_N1/0506_151213_GR_fge_greenfield_S3_N1"),
                   "Centralized": os.path.join(
                       "eoles/outputs/0506_greenfield_S3_N1/0506_064807_centralized_insulation_greenfield_S3_N1"),
                   "MWh_tCO2": os.path.join(
                       "eoles/outputs/0506_greenfield_S3_N1/0506_151211_MWh_tCO2_greenfield_S3_N1"),
                   }
    #
    # dict_output2 = {
    #     "Uniform": os.path.join("eoles/outputs/0507_greenfield_S2p_N1/0507_230755_uniform_greenfield_S2p_N1"),
    #     "Global renovation": os.path.join("eoles/outputs/0507_greenfield_S2p_N1/0507_230000_GR_greenfield_S2p_N1"),
    #     "Global renovation FGE": os.path.join(
    #         "eoles/outputs/0507_greenfield_S2p_N1/0507_224644_GR_fge_greenfield_S2p_N1"),
    #     "Centralized": os.path.join(
    #         "eoles/outputs/0507_greenfield_S2p_N1/0507_175056_centralized_insulation_greenfield_S2p_N1"),
    #     "tCO2": os.path.join("eoles/outputs/0507_greenfield_S2p_N1/0507_231124_MWh_tCO2_greenfield_S2p_N1")
    #     }
    #
    # comparison_simulations_scenarios(dict_output1, dict_output2, greenfield=False, health=False, x_min=0,
    #                                  x_max=None, y_min=0, y_max=None,
    #                                  rotation=90, save_path=None, pdf=False, carbon_constraint=True, percent=False,
    #                                  eoles=True)

    results_resirf = {}
    for path, name_config in zip(dict_output1.values(), [n for n in dict_output1.keys()]):
        with open(os.path.join(path, 'coupling_results.pkl'), "rb") as file:
            output = load(file)
            results_resirf[name_config] = output["Output global ResIRF ()"]

    folder_comparison = os.path.join("eoles/outputs/comparison")
    # date = datetime.datetime.now().strftime("%m%d_%H%M%S")
    # folder = os.path.join(folder_comparison, f'{date}_greenfield_S3_N1_prices')
    # if not os.path.isdir(folder):
    #     os.mkdir(folder)

    # Plots coupling
    total_system_costs_df, consumption_savings_tot_df, complete_system_costs_2050_df = comparison_simulations(
        dict_output1, ref="Uniform", greenfield=True, health=True, save_path=folder_comparison, carbon_constraint=True)



