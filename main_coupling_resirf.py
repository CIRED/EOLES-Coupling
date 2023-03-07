import json
import os
from importlib import resources

import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from pickle import dump, load
from multiprocessing import Pool

from project.coupling import ini_res_irf, simu_res_irf
from project.write_output import plot_scenario
from project.model import get_inputs, social_planner

from project.model import create_logger, get_config, get_inputs
from eoles.model_resirf_coupling import ModelEOLES
from eoles.utils import get_config, get_pandas, calculate_annuities_resirf
from eoles.write_output import plot_simulation, plot_blackbox_optimization, save_summary_pdf
import eoles.utils
from eoles.coupling_resirf_eoles import resirf_eoles_coupling_dynamic, optimize_blackbox_resirf_eoles_coupling, \
    calibration_price, get_energy_prices_and_taxes
import logging
import argparse


from matplotlib import pyplot as plt

LOG_FORMATTER = '%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# consoler handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
logger.addHandler(console_handler)

DICT_CONFIG_RESIRF = {
    "classic": "eoles/inputs/config/config_resirf.json",
    "classic_premature3": "eoles/inputs/config/config_resirf_premature3.json",
    "threshold": "eoles/inputs/config/config_resirf_threshold.json",
    "classic_simple": "eoles/inputs/config/config_resirf_simple.json",
    "threshold_simple": "eoles/inputs/config/config_resirf_threshold_simple.json",
    "classic_simple_premature3": "eoles/inputs/config/config_resirf_simple_premature3.json",
    "classic_simple_premature5": "eoles/inputs/config/config_resirf_simple_premature5.json",
    "classic_simple_premature10": "eoles/inputs/config/config_resirf_simple_premature10.json",
    "threshold_simple_premature3": "eoles/inputs/config/config_resirf_threshold_simple_premature3.json",
    "threshold_simple_premature10": "eoles/inputs/config/config_resirf_threshold_simple_premature3.json",
    "nolandlord": "eoles/inputs/config/config_resirf_nolandlord.json",
    "nomultifamily": "eoles/inputs/config/config_resirf_nomultifamily.json",
    "nolandlord_nomultifamily": "eoles/inputs/config/config_resirf_nolandlord_nomultifamily.json",
    "nolandlord_simple": "eoles/inputs/config/config_resirf_nolandlord_simple.json",
    "nomultifamily_simple": "eoles/inputs/config/config_resirf_nomultifamily_simple.json",
    "nolandlord_nomultifamily_simple": "eoles/inputs/config/config_resirf_nolandlord_nomultifamily_simple.json",
}

DICT_CONFIG_EOLES = {
    "eoles_classic": "eoles_coupling",
    "eoles_worst_case": "eoles_coupling_worst_case"
}


def run_optimization_scenario(config_coupling, name_config_coupling="default"):
    """

    :param config_coupling: dict
        Dictionary containing different configurations. For the time being, includes:
        configuration for resirf (market failures, threshold), consideration of H2 CCGT in EOLES, and number of iterations for optimization
    :return:
    Saves the output of the optimization + save plots
    """

    config_resirf, config_eoles_spec = config_coupling["config_resirf"], DICT_CONFIG_EOLES[config_coupling["config_eoles"]]
    config_resirf_path = DICT_CONFIG_RESIRF[config_resirf]
    # print(config_resirf_path)

    calibration_threshold = config_coupling["calibration_threshold"]

    # Calibration: whether we use threshold or not
    name_calibration = 'calibration'
    if calibration_threshold is True:
        name_calibration = '{}_threshold'.format(name_calibration)
    # print(name_calibration)

    export_calibration = os.path.join('eoles', 'outputs', 'calibration', '{}.pkl'.format(name_calibration))
    import_calibration = os.path.join('eoles', 'outputs', 'calibration', '{}.pkl'.format(name_calibration))

    # initialization
    buildings, energy_prices, taxes, cost_heater, cost_insulation, lifetime_heater, demolition_rate, flow_built, post_inputs, policies_heater, policies_insulation, technical_progress, financing_cost, premature_replacement = ini_res_irf(
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

    one_shot_setting = config_coupling["one_shot_setting"]

    anticipated_demand_t10 = False
    if "anticipated_demand_t10" in config_coupling.keys():
        anticipated_demand_t10 = config_coupling["anticipated_demand_t10"]

    anticipated_scc = False
    if "anticipated_scc" in config_coupling.keys():
        anticipated_scc = config_coupling["anticipated_scc"]

    price_feedback = False
    if "price_feedback" in config_coupling.keys():
        price_feedback = config_coupling["price_feedback"]

    energy_prices_ht, energy_taxes = get_energy_prices_and_taxes(config_resirf_path)
    calibration_elec_lcoe, calibration_elec_transport_distrib, calibration_gas, m_eoles = calibration_price(config_eoles, scc=100)
    config_coupling["calibration_elec_lcoe"] = calibration_elec_lcoe
    config_coupling["calibration_elec_transport_distrib"] = calibration_elec_transport_distrib
    config_coupling["calibration_gas_lcoe"] = calibration_gas

    output, buildings, dict_optimizer = resirf_eoles_coupling_dynamic(buildings, energy_prices, taxes, cost_heater, cost_insulation,
                                                           demolition_rate, flow_built, post_inputs, policies_heater, policies_insulation,
                                                           list_year, list_trajectory_scc, scenario_cost,
                                                           config_eoles=config_eoles, config_coupling=config_coupling,
                                                           add_CH4_demand=False, one_shot_setting=one_shot_setting,
                                                           technical_progress=technical_progress, financing_cost=financing_cost,
                                                            premature_replacement=premature_replacement,
                                                           anticipated_scc=anticipated_scc, anticipated_demand_t10=anticipated_demand_t10, optimization=True,
                                                           price_feedback=price_feedback,
                                                           energy_prices_ht=energy_prices_ht, energy_taxes=energy_taxes)

    # Save results
    date = datetime.datetime.now().strftime("%m%d_%H%M%S")
    export_results = os.path.join("eoles", "outputs", f'{date}_{name_config_coupling}')

    # Create directories
    if not os.path.isdir(export_results):
        os.mkdir(export_results)

    if not os.path.isdir(os.path.join(export_results, "config")):
        os.mkdir(os.path.join(export_results, "config"))

    if not os.path.isdir(os.path.join(export_results, "dataframes")):
        os.mkdir(os.path.join(export_results, "dataframes"))

    if not os.path.isdir(os.path.join(export_results, "plots")):
        os.mkdir(os.path.join(export_results, "plots"))

    # if not os.path.isdir(os.path.join(export_results, "plots", "plots_resirf")):
    #     os.mkdir(os.path.join(export_results, "plots", "plots_resirf"))

    with open(os.path.join(export_results, "config", 'config_eoles.json'), "w") as outfile:
        outfile.write(json.dumps(config_eoles, indent=4))

    # read and save the actual config file
    with open(config_resirf_path) as file:
        config_res_irf = json.load(file)

    with open(os.path.join(export_results, "config", 'config_resirf.json'), "w") as outfile:
        outfile.write(json.dumps(config_res_irf, indent=4))

    with open(os.path.join(export_results, "config", 'config_coupling.json'), "w") as outfile:
        outfile.write(json.dumps(config_coupling, indent=4))

    with open(os.path.join(export_results, 'coupling_results.pkl'), "wb") as file:
        dump(output, file)

    if output is not None:  # we exclude the case where we did not find a way to not violate the carbon budget
        for key in output.keys():
            if key != 'Subvention heater' and key != 'Subvention insulation' and key != 'max_iter':
                key_save = '_'.join(key.split('(')[0].lower().split(' ')[:-1])
                output[key].to_csv(os.path.join(export_results, 'dataframes', f'{key_save}.csv'))

        plot_blackbox_optimization(dict_optimizer, save_path=os.path.join(export_results))
        plot_simulation(output, save_path=os.path.join(export_results, "plots"))
        buildings.path = os.path.join(export_results, "plots")
        plot_scenario(output["Output global ResIRF ()"], output["Stock global ResIRF ()"], buildings)  # make ResIRF plots
        save_summary_pdf(path=export_results)  # saving summary as pdf

    return name_config_coupling


def run_scenario_no_opti(name_folder, name_config_coupling="default"):
    """

    :param config_coupling: dict
        Dictionary containing different configurations. For the time being, includes:
        configuration for resirf (market failures, threshold), consideration of H2 CCGT in EOLES, and number of iterations for optimization
    :return:
    Saves the output of the optimization + save plots
    """

    # Load config coupling from name folder
    with open(os.path.join(name_folder, "config", "config_coupling.json")) as file:
        config_coupling = json.load(file)

    # Load subsidies path
    subsidies_df = pd.read_csv(os.path.join(name_folder, "dataframes", "subsidies.csv"), index_col=0)
    list_subsidies_heater = list(subsidies_df[["Heater"]])
    list_subsidies_insulation = list(subsidies_df[["Insulation"]])

    config_resirf, config_eoles_spec = config_coupling["config_resirf"], DICT_CONFIG_EOLES[config_coupling["config_eoles"]]
    config_resirf_path = DICT_CONFIG_RESIRF[config_resirf]
    # print(config_resirf_path)

    calibration_threshold = config_coupling["calibration_threshold"]

    # Calibration: whether we use threshold or not
    name_calibration = 'calibration'
    if calibration_threshold is True:
        name_calibration = '{}_threshold'.format(name_calibration)
    # print(name_calibration)

    export_calibration = os.path.join('eoles', 'outputs', 'calibration', '{}.pkl'.format(name_calibration))
    import_calibration = os.path.join('eoles', 'outputs', 'calibration', '{}.pkl'.format(name_calibration))

    # initialization
    buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built, post_inputs, policies_heater, policies_insulation, technical_progress, financing_cost = ini_res_irf(
        path=os.path.join('eoles', 'outputs', 'ResIRF'),
        logger=None,
        config=config_resirf_path,
        import_calibration=None,
        export_calibration=None, cost_factor=1)

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
    t10 = None
    if "t10" in config_coupling.keys():
        t10 = config_coupling["t10"]
    output, dict_optimizer = resirf_eoles_coupling_dynamic_no_opti(list_sub_heater=list_subsidies_heater, list_sub_insulation=list_subsidies_insulation,
                                                                   buildings=buildings, energy_prices=energy_prices, taxes=taxes,
                                                                   cost_heater=cost_heater, cost_insulation=cost_insulation,
                                                                   flow_built=flow_built, post_inputs=post_inputs, policies_heater=policies_heater,
                                                                   policies_insulation=policies_insulation, list_year=list_year, list_trajectory_scc=list_trajectory_scc,
                                                                   scenario_cost=scenario_cost, config_eoles=config_eoles,
                                                                   add_CH4_demand=False, one_shot_setting=one_shot_setting, sub_design=sub_design,
                                                      health=health, carbon_constraint=carbon_constraint, discount_rate=discount_rate,
                                                           rebound=rebound, technical_progress=technical_progress, financing_cost=financing_cost)

    # Save results
    date = datetime.datetime.now().strftime("%m%d_%H%M%S")
    export_results = os.path.join("eoles", "outputs", f'{date}_{name_config_coupling}')

    # Create directories
    if not os.path.isdir(export_results):
        os.mkdir(export_results)

    if not os.path.isdir(os.path.join(export_results, "config")):
        os.mkdir(os.path.join(export_results, "config"))

    if not os.path.isdir(os.path.join(export_results, "dataframes")):
        os.mkdir(os.path.join(export_results, "dataframes"))

    if not os.path.isdir(os.path.join(export_results, "plots")):
        os.mkdir(os.path.join(export_results, "plots"))

    with open(os.path.join(export_results, "config", 'config_eoles.json'), "w") as outfile:
        outfile.write(json.dumps(config_eoles, indent=4))

    # read and save the actual config file
    with open(config_resirf_path) as file:
        config_res_irf = json.load(file)

    with open(os.path.join(export_results, "config", 'config_resirf.json'), "w") as outfile:
        outfile.write(json.dumps(config_res_irf, indent=4))

    with open(os.path.join(export_results, "config", 'config_coupling.json'), "w") as outfile:
        outfile.write(json.dumps(config_coupling, indent=4))

    with open(os.path.join(export_results, 'coupling_results.pkl'), "wb") as file:
        dump(output, file)

    if output is not None:  # we exclude the case where we did not find a way to not violate the carbon budget
        for key in output.keys():
            if key != 'Subvention heater' and key != 'Subvention insulation' and key != 'max_iter':
                key_save = '_'.join(key.split('(')[0].lower().split(' ')[:-1])
                output[key].to_csv(os.path.join(export_results, 'dataframes', f'{key_save}.csv'))

        plot_blackbox_optimization(dict_optimizer, save_path=os.path.join(export_results))
        plot_simulation(output, save_path=os.path.join(export_results, "plots"))

    return name_config_coupling


def run_scenario(list_sub_heater, list_sub_insulation, config_coupling, name_config_coupling="default"):
    """
    Runs without optimization, with specific path for subsidies
    :param config_coupling: dict
        Dictionary containing different configurations. For the time being, includes:
        configuration for resirf (market failures, threshold), consideration of H2 CCGT in EOLES, and number of iterations for optimization
    :return:
    Saves the output of the optimization + save plots
    """

    config_resirf, config_eoles = config_coupling["config_resirf"], DICT_CONFIG_EOLES[config_coupling["config_eoles"]]
    config_resirf_path = DICT_CONFIG_RESIRF[config_resirf]
    # print(config_resirf_path)

    calibration_threshold = config_coupling["calibration_threshold"]

    # Calibration: whether we use threshold or not
    name_calibration = 'calibration'
    if calibration_threshold is True:
        name_calibration = '{}_threshold'.format(name_calibration)
    # print(name_calibration)

    export_calibration = os.path.join('eoles', 'outputs', 'calibration', '{}.pkl'.format(name_calibration))
    import_calibration = os.path.join('eoles', 'outputs', 'calibration', '{}.pkl'.format(name_calibration))

    # initialization
    buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built, post_inputs, policies_heater, policies_insulation = ini_res_irf(
        path=os.path.join('eoles', 'outputs', 'ResIRF'),
        logger=None,
        config=config_resirf_path,
        import_calibration=None,
        export_calibration=export_calibration, cost_factor=1)

    list_year = config_coupling["list_year"]
    list_trajectory_scc = config_coupling["list_trajectory_scc"]  # SCC trajectory
    config_eoles = eoles.utils.get_config(spec="greenfield")  # TODO: changer le nom de la config qu'on appelle

    h2ccgt = config_coupling["h2ccgt"]
    scenario_cost = config_coupling["scenario_cost_eoles"]
    if not h2ccgt:  # we do not allow h2 ccgt plants
        if "fix_capa" in scenario_cost.keys():
            scenario_cost["fix_capa"]["h2_ccgt"] = 0
        else:
            scenario_cost["fix_capa"] = {
                "h2_ccgt": 0
            }

    one_shot_setting = config_coupling["one_shot_setting"]
    sub_design, annuity_health = config_coupling["sub_design"], config_coupling["annuity_health"]
    output = resirf_eoles_coupling_dynamic_no_opti(list_sub_heater, list_sub_insulation, buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built,
                                           post_inputs, policies_heater, policies_insulation,
                                           list_year, list_trajectory_scc, scenario_cost, config_eoles=config_eoles,
                                           add_CH4_demand=False, one_shot_setting=one_shot_setting, sub_design=sub_design,
                                                   annuity_health=annuity_health)

    # Save results
    date = datetime.datetime.now().strftime("%m%d_%H%M%S")
    if h2ccgt:
        export_results = os.path.join("eoles", "outputs", f'{date}_{name_config_coupling}_maxiter{max_iter}_h2ccgt')
    else:
        export_results = os.path.join("eoles", "outputs", f'{date}_{name_config_coupling}_maxiter{max_iter}')

    # Create directories
    if not os.path.isdir(export_results):
        os.mkdir(export_results)

    if not os.path.isdir(os.path.join(export_results, "config")):
        os.mkdir(os.path.join(export_results, "config"))

    if not os.path.isdir(os.path.join(export_results, "dataframes")):
        os.mkdir(os.path.join(export_results, "dataframes"))

    #
    # if not os.path.isdir(os.path.join(export_results, "plots")):
    #     os.mkdir(os.path.join(export_results, "plots"))

    with open(os.path.join(export_results, 'coupling_results_no_opti.pkl'), "wb") as file:
        dump(output, file)

    for key in output.keys():
        if key != 'Subvention heater' and key != 'Subvention insulation' and key != 'max_iter':
            key_save = '_'.join(key.split('(')[0].lower().split(' ')[:-1])
            output[key].to_csv(os.path.join(export_results, 'dataframes', f'{key_save}.csv'))

    # with open(os.path.join(export_results, "config", 'config_eoles.json'), "w") as outfile:
    #     outfile.write(json.dumps(config_eoles, indent=4))
    #
    # # read and save the actual config file
    # with open(config_resirf_path) as file:
    #     config_res_irf = json.load(file)
    #
    # with open(os.path.join(export_results, "config", 'config_resirf.json'), "w") as outfile:
    #     outfile.write(json.dumps(config_res_irf, indent=4))

    with open(os.path.join(export_results, "config", 'config_coupling.json'), "w") as outfile:
        outfile.write(json.dumps(config_coupling, indent=4))

    # plot_simulation(output, save_path=os.path.join(export_results, "plots"))

    return output, name_config_coupling


def run_multiple_configs(dict_config, cpu:int):
    """Run multiple configs in parallel"""
    logger.info('Scenarios: {}'.format(', '.join(dict_config.keys())))
    try:
        logger.info('Launching processes')
        with Pool(cpu) as pool:
            results = pool.starmap(run_optimization_scenario, zip(dict_config.values(), [n for n in dict_config.keys()]))
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simulate coupling.')
    parser.add_argument("-c", "--config", type=str, default="classic", help="configuration for resirf")
    parser.add_argument("-i", "--maxiter", type=int, default=20, help="maximum iterations for blackbox optimization")
    parser.add_argument('--threshold', action=argparse.BooleanOptionalAction, default=False, help="whether we activate the threshold setting")
    parser.add_argument("--h2ccgt", action=argparse.BooleanOptionalAction, default=False, help="whether we allow H2-CCGT plants are not")
    parser.add_argument("--cpu", type=int, default=40, help="CPUs for multiprocessing")

    args = parser.parse_args()
    config_resirf, max_iter, calibration_threshold, h2ccgt, cpu = args.config, args.maxiter, args.threshold, args.h2ccgt, args.cpu  # we select the config we are interested in
    # config_resirf, max_iter, calibration_threshold, h2ccgt = "classic", 15, False, False

    # config_coupling = {
    #     'config_resirf': config_resirf,
    #     'h2ccgt': h2ccgt,
    #     'calibration_threshold': calibration_threshold,
    #     'max_iter': max_iter,
    #     'sub_design': 'natural_gas',
    #     "annuity_health": False,
    #     "carbon_constraint": False,
    #     'one_shot_setting': True,
    #     'fix_sub_heater': False,
    #     'list_year': [2025],
    #     'list_trajectory_scc': [650],
    #     'scenario_cost_eoles': {
    #         'biomass_potential': {
    #             'methanization': 0,
    #             'pyrogazification': 0
    #         },
    #         'maximum_capacity': {
    #             'offshore_g': 10,
    #             'offshore_f': 20,
    #             'nuclear': 25,
    #             'onshore': 70,
    #             'pv_g': 50,
    #             'pv_c': 50
    #         },
    #         'existing_capacity': {
    #             'offshore_g': 0,
    #             'offshore_f': 0,
    #             'nuclear': 0,
    #             'onshore': 0,
    #             'pv_g': 0,
    #             'pv_c': 0
    #         },
    #         'vOM': {
    #             'natural_gas': 0.035,
    #         }
    #     }
    # }


    DICT_CONFIGS_BATCH1 = {
        "noHC_temoin_simple_anticipatedSCCt5": {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': False,
            'max_iter': 18,
            'sub_design': None,
            "health": False,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'anticipated_scc': "t5",
            'scenario_cost_eoles': {}
        },
        "noHC_global_renovation_simple_anticipatedSCCt5": {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': False,
            'max_iter': 18,
            'sub_design': "global_renovation",
            "health": False,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'anticipated_scc': "t5",
            'scenario_cost_eoles': {}
        },
        "noHC_threshold_simple_anticipatedSCCt5": {
            'config_resirf': "threshold_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': True,
            'h2ccgt': False,
            'max_iter': 18,
            'sub_design': None,
            "health": False,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'anticipated_scc': "t5",
            'scenario_cost_eoles': {}
        },
        "noHC_efficiency100_simple_anticipatedSCCt5": {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': False,
            'max_iter': 18,
            'sub_design': "efficiency_100",
            "health": False,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'anticipated_scc': "t5",
            'scenario_cost_eoles': {}
        },
        "global_renovation_simple_anticipatedSCCt5": {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': False,
            'max_iter': 18,
            'sub_design': "global_renovation",
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'anticipated_scc': "t5",
            'scenario_cost_eoles': {}
        },
        "efficiency100_simple_anticipatedSCCt5": {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': False,
            'max_iter': 18,
            'sub_design': "efficiency_100",
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'anticipated_scc': "t5",
            'scenario_cost_eoles': {}
        },

    }

    DICT_CONFIGS_BATCH1 = {
        "temoin_simple": {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 22,
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
        },
        "threshold_simple": {
            'config_resirf': "threshold_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': True,
            'h2ccgt': True,
            'max_iter': 22,
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
        },
        "global_renovation_simple": {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 22,
            'sub_design': "global_renovation",
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'scenario_cost_eoles': {}
        },
        "efficiency100_simple": {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 22,
            'sub_design': "efficiency_100",
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'scenario_cost_eoles': {}
        },
        "temoin_simple_premature10": {
            'config_resirf': "classic_simple_premature10",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 22,
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
        },
        "threshold_simple_premature10": {
            'config_resirf': "threshold_simple_premature10",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': True,
            'h2ccgt': True,
            'max_iter': 22,
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
        },
    }

    DICT_CONFIGS_2 = {
        "noHC_temoin_simple_premature3": {
            'config_resirf': "classic_simple_premature3",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 27,
            'sub_design': None,
            "health": False,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'scenario_cost_eoles': {}
        },
        "noHC_threshold_simple_premature3": {
            'config_resirf': "threshold_simple_premature3",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': True,
            'h2ccgt': True,
            'max_iter': 27,
            'sub_design': None,
            "health": False,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'scenario_cost_eoles': {}
        },
        "noHC_global_renovation_simple_premature3": {
            'config_resirf': "classic_simple_premature3",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 27,
            'sub_design': "global_renovation",
            "health": False,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'scenario_cost_eoles': {}
        },
        "noHC_efficiency100_simple_premature3": {
            'config_resirf': "classic_simple_premature3",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 27,
            'sub_design': "efficiency_100",
            "health": False,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'scenario_cost_eoles': {}
        },
        "temoin_simple_premature5": {
            'config_resirf': "classic_simple_premature5",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 27,
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
        },
        "global_renovation_simple_premature5": {
            'config_resirf': "classic_simple_premature5",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 27,
            'sub_design': "global_renovation",
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'scenario_cost_eoles': {}
        },
        "temoin_simple_elecworst_premature3": {
            'config_resirf': "classic_simple_premature3",
            "config_eoles": "eoles_worst_case",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 22,
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
        },
    }

    DICT_CONFIGS = {
        "efficiency100_simple_premature5": {
            'config_resirf': "classic_simple_premature5",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 27,
            'sub_design': "efficiency_100",
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'scenario_cost_eoles': {}
        },
        "temoin_simple_premature3_carbonbudget": {
            'config_resirf': "classic_simple_premature3",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 27,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'scenario_cost_eoles': {}
        },
        "global_renovation_simple_premature3_carbonbudget": {
            'config_resirf': "classic_simple_premature3",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 27,
            'sub_design': "global_renovation",
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'scenario_cost_eoles': {}
        },
        "efficiency100_simple_premature3_carbonbudget": {
            'config_resirf': "classic_simple_premature3",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 27,
            'sub_design': "efficiency_100",
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'scenario_cost_eoles': {}
        },
        "temoin_simple_premature3_anticipatedSCC": {
            'config_resirf': "classic_simple_premature3",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 27,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'anticipated_scc': 'average',
            'scenario_cost_eoles': {}
        },
        "global_renovation_simple_premature3_anticipatedSCC": {
            'config_resirf': "classic_simple_premature3",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 27,
            'sub_design': "global_renovation",
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'anticipated_scc': 'average',
            'scenario_cost_eoles': {}
        },
        "temoin_simple_elecworst_premature10": {
            'config_resirf': "classic_simple_premature10",
            "config_eoles": "eoles_worst_case",  # includes costs assumptions
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 22,
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
        },
    }

    # name_config_coupling, optimizer = run_optimization_scenario(config_coupling, name_config_coupling="classic_oneshot_scc650_nobiogas_subdesignGas_annuityFalse")
    run_multiple_configs(DICT_CONFIGS_BATCH1, cpu=cpu)

    # If need to rerun without doing the optimization
    # dict_simu_rerun = {
    #     "noHC_temoin": os.path.join("eoles/outputs/0130_210326_noHC_temoin"),
    #     "noHC_global_renovation": os.path.join("eoles/outputs/0130_205702_noHC_global_renovation"),
    #     "noHC_global_renovation_nolandlord_nomultifamily": os.path.join("eoles/outputs/0130_210711_noHC_global_renovation_nolandlord_nomultifamily"),
    #     "noHC_threshold": os.path.join("eoles/outputs/0128_235752_noHC_threshold"),
    #     "noHC-GR-FG": os.path.join("eoles/outputs/0130_205612_noHC_global_renov_FG"),
    #     "noHC_best_eff": os.path.join("eoles/outputs/0130_205757_noHC_best_eff"),
    #     "noHC_best_eff_FG": os.path.join("eoles/outputs/0130_210645_noHC_best_eff_FG"),
    #     "global_renovation": os.path.join("eoles/outputs/0130_210542_global_renovation"),
    #     "global_renovation_nolandlord_nomultifamily": os.path.join("eoles/outputs/0130_205638_global_renovation_nolandlord_nomultifamily"),
    #     "temoin": os.path.join("eoles/outputs/0130_205740_temoin"),
    # }
    #
    # for simu in dict_simu_rerun:
    #     run_scenario_no_opti(name_folder=dict_simu_rerun[simu], name_config_coupling=simu)

    # output, name_config_coupling = run_scenario(list_sub_heater=[0], list_sub_insulation=[0],
    #                                             config_coupling=config_coupling, name_config_coupling="classic_oneshot_scc2000_nobiogas_subheaterf")

