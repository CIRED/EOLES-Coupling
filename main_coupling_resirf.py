import json
import os
from importlib import resources

import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from pickle import dump, load
from multiprocessing import Pool

from project.sensitivity import ini_res_irf, simu_res_irf
from project.model import get_inputs, social_planner

from project.model import create_logger, get_config, get_inputs
from eoles.model_resirf_coupling import ModelEOLES
from eoles.utils import get_config, get_pandas, calculate_annuities_resirf
from eoles.write_output import plot_simulation, plot_blackbox_optimization
import eoles.utils
from eoles.coupling_resirf_eoles import resirf_eoles_coupling_dynamic, optimize_blackbox_resirf_eoles_coupling, resirf_eoles_coupling_dynamic_no_opti
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

DICT_CONFIG = {
    "classic": "eoles/inputs/config/config_resirf.json",
    "landlord": "eoles/inputs/config/config_resirf_nolandlord.json",
    "multifamily": "eoles/inputs/config/config_resirf_nomultifamily.json",
    "threshold": "eoles/inputs/config/config_resirf_threshold.json",
}


def run_optimization_scenario(config_coupling, name_config_coupling="default"):
    """

    :param config_coupling: dict
        Dictionary containing different configurations. For the time being, includes:
        configuration for resirf (market failures, threshold), consideration of H2 CCGT in EOLES, and number of iterations for optimization
    :return:
    Saves the output of the optimization + save plots
    """

    config_resirf = config_coupling["config_resirf"]
    config_resirf_path = DICT_CONFIG[config_resirf]
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

    max_iter, one_shot_setting, fix_sub_heater = config_coupling["max_iter"], config_coupling["one_shot_setting"], config_coupling["fix_sub_heater"]

    sub_design, annuity_health = config_coupling["sub_design"], config_coupling["annuity_health"]

    output, dict_optimizer = resirf_eoles_coupling_dynamic(buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built,
                                                       post_inputs, policies_heater, policies_insulation,
                                                       list_year, list_trajectory_scc, scenario_cost, config_eoles=config_eoles,
                                                       max_iter=max_iter, add_CH4_demand=False,
                                                       one_shot_setting=one_shot_setting, fix_sub_heater=fix_sub_heater, sub_design=sub_design,
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

    if not os.path.isdir(os.path.join(export_results, "plots")):
        os.mkdir(os.path.join(export_results, "plots"))

    with open(os.path.join(export_results, 'coupling_results.pkl'), "wb") as file:
        dump(output, file)

    for key in output.keys():
        if key != 'Subvention heater' and key != 'Subvention insulation' and key != 'max_iter':
            key_save = '_'.join(key.split('(')[0].lower().split(' ')[:-1])
            output[key].to_csv(os.path.join(export_results, 'dataframes', f'{key_save}.csv'))

    with open(os.path.join(export_results, "config", 'config_eoles.json'), "w") as outfile:
        outfile.write(json.dumps(config_eoles, indent=4))

    # read and save the actual config file
    with open(config_resirf_path) as file:
        config_res_irf = json.load(file)

    with open(os.path.join(export_results, "config", 'config_resirf.json'), "w") as outfile:
        outfile.write(json.dumps(config_res_irf, indent=4))

    with open(os.path.join(export_results, "config", 'config_coupling.json'), "w") as outfile:
        outfile.write(json.dumps(config_coupling, indent=4))

    plot_simulation(output, save_path=os.path.join(export_results, "plots"))
    plot_blackbox_optimization(dict_optimizer, save_path=os.path.join(export_results))

    # with open(os.path.join(export_results, 'dict_optimizer.pkl'), "wb") as file:
    #     dump(dict_optimizer, file)

    return name_config_coupling, dict_optimizer


def run_scenario(list_sub_heater, list_sub_insulation, config_coupling, name_config_coupling="default"):
    """
    Runs without optimization, with specific path for subsidies
    :param config_coupling: dict
        Dictionary containing different configurations. For the time being, includes:
        configuration for resirf (market failures, threshold), consideration of H2 CCGT in EOLES, and number of iterations for optimization
    :return:
    Saves the output of the optimization + save plots
    """

    config_resirf = config_coupling["config_resirf"]
    config_resirf_path = DICT_CONFIG[config_resirf]
    print(config_resirf_path)

    calibration_threshold = config_coupling["calibration_threshold"]

    # Calibration: whether we use threshold or not
    name_calibration = 'calibration'
    if calibration_threshold is True:
        name_calibration = '{}_threshold'.format(name_calibration)
    print(name_calibration)

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

    # parser = argparse.ArgumentParser(description='Simulate coupling.')
    # parser.add_argument("-c", "--config", type=str, default="classic", help="configuration for resirf")
    # parser.add_argument("-i", "--maxiter", type=int, default=20, help="maximum iterations for blackbox optimization")
    # parser.add_argument('--threshold', action=argparse.BooleanOptionalAction, default=False, help="whether we activate the threshold setting")
    # parser.add_argument("--h2ccgt", action=argparse.BooleanOptionalAction, default=False, help="whether we allow H2-CCGT plants are not")
    # parser.add_argument("--cpu", type=int, default=40, help="CPUs for multiprocessing")
    #
    # args = parser.parse_args()
    # config_resirf, max_iter, calibration_threshold, h2ccgt, cpu = args.config, args.maxiter, args.threshold, args.h2ccgt, args.cpu  # we select the config we are interested in
    config_resirf, max_iter, calibration_threshold, h2ccgt = "classic", 15, False, False

    config_coupling = {
        'config_resirf': config_resirf,
        'h2ccgt': h2ccgt,
        'calibration_threshold': calibration_threshold,
        'max_iter': max_iter,
        'sub_design': 'natural_gas',
        "annuity_health": False,
        'one_shot_setting': True,
        'fix_sub_heater': False,
        'list_year': [2025],
        'list_trajectory_scc': [650],
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

    dict_config_trajectory = {
        "classic": {
            'config_resirf': "classic",
            'calibration_threshold': False,
            'h2ccgt': False,
            'max_iter': 14,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [180, 250, 350, 500, 650],
            'scenario_cost_eoles': {},
            'one_shot_setting': False
        },
        "classic_h2ccgt": {
            'config_resirf': "classic",
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 14,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [180, 250, 350, 500, 650],
            'scenario_cost_eoles': {},
            'one_shot_setting': False
        },
        "threshold": {
            'config_resirf': "threshold",
            'calibration_threshold': True,
            'h2ccgt': False,
            'max_iter': 14,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [180, 250, 350, 500, 650],
            'scenario_cost_eoles': {},
            'one_shot_setting': False
        },
        "landlord": {
            'config_resirf': "landlord",
            'calibration_threshold': False,
            'h2ccgt': False,
            'max_iter': 14,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [180, 250, 350, 500, 650],
            'scenario_cost_eoles': {},
            'one_shot_setting': False
        },
        "multifamily": {
            'config_resirf': "multifamily",
            'calibration_threshold': False,
            'h2ccgt': False,
            'max_iter': 14,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [180, 250, 350, 500, 650],
            'scenario_cost_eoles': {},
            'one_shot_setting': False
        }
    }

    list_design = ["very_low_income", "low_income", "natural_gas", "fossil", "electricity", "global_renovation",
                   "global_renovation_low_income", "best_option"]
    dict_config_subdesign = {f'classic_design_{design}': {
            'config_resirf': "classic",
            'calibration_threshold': False,
            'h2ccgt': False,
            'max_iter': 17,
            'sub_design': design,
            "annuity_health": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [180, 250, 350, 500, 650],
            'scenario_cost_eoles': {}
        } for design in list_design}
    dict_config_subdesign.update({f'classic_nobiogas_design_{design}': {
            'config_resirf': "classic",
            'calibration_threshold': False,
            'h2ccgt': True,
            'max_iter': 17,
            'sub_design': design,
            "annuity_health": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [180, 250, 350, 500, 650],
            'scenario_cost_eoles': {
                'biomass_potential': {
                    'methanization': 0,
                    'pyrogazification': 0
                }
            }
        } for design in list_design})

    list_failure = ["landlord", "multifamily"]
    dict_config_failure = {f'{failure}_design_{design}': {
            'config_resirf': failure,
            'calibration_threshold': False,
            'h2ccgt': False,
            'max_iter': 17,
            'sub_design': design,
            "annuity_health": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'scenario_cost_eoles': {}
        } for design in list_design for failure in list_failure}
    dict_config_failure.update({'threshold': {
            'config_resirf': "threshold",
            'calibration_threshold': True,
            'h2ccgt': False,
            'max_iter': 17,
            'sub_design': None,
            "annuity_health": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [180, 250, 350, 500, 650],
            'scenario_cost_eoles': {}
        }})

    # print(calibration_threshold, h2ccgt)

    name_config_coupling, optimizer = run_optimization_scenario(config_coupling, name_config_coupling="classic_oneshot_scc650_nobiogas_subdesignGas_annuityFalse")
    # run_multiple_configs(dict_config_subdesign, cpu=cpu)

    # output, name_config_coupling = run_scenario(list_sub_heater=[0], list_sub_insulation=[0],
    #                                             config_coupling=config_coupling, name_config_coupling="classic_oneshot_scc2000_nobiogas_subheaterf")

