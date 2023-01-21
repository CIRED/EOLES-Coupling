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
from eoles.write_output import plot_simulation
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


def run_optimization_scenario(config:str, calibration_threshold:bool, h2ccgt:bool):

    config_resirf_path = DICT_CONFIG[config]

    # Calibration: whether we use threshold or not
    name_calibration = 'calibration'
    if calibration_threshold is True:
        name_calibration = '{}_threshold'.format(name_calibration)

    export_calibration = os.path.join('eoles', 'outputs', 'calibration', '{}.pkl'.format(name_calibration))
    import_calibration = os.path.join('eoles', 'outputs', 'calibration', '{}.pkl'.format(name_calibration))

    # initialization
    buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built, post_inputs, policies_heater, policies_insulation = ini_res_irf(
        path=os.path.join('eoles', 'outputs', 'ResIRF'),
        logger=None,
        config=config_resirf_path,
        import_calibration=import_calibration,
        export_calibration=export_calibration)

    list_year = [2025, 2030, 2035, 2040, 2045]
    list_trajectory_scc = [180, 250, 350, 500, 650]  # SCC trajectory from Quinet
    config_eoles = eoles.utils.get_config(spec="greenfield")  # TODO: changer le nom de la config qu'on appelle

    scenario_cost = {}
    if not h2ccgt:  # we do not allow h2 ccgt plants
        scenario_cost["fix_capa"] = {
            "h2_ccgt": 0
        }

    output = resirf_eoles_coupling_dynamic(buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built,
                                           post_inputs, policies_heater, policies_insulation,
                                           list_year, list_trajectory_scc, scenario_cost, config_eoles=config_eoles,
                                           max_iter=20, add_CH4_demand=False, return_optimizer=False)

    # Save results
    date = datetime.datetime.now().strftime("%m%d%H%M")
    if h2ccgt:
        export_results = os.path.join("eoles", "outputs", f'{date}_{config}_h2ccgt')
    else:
        export_results = os.path.join("eoles", "outputs", f'{date}_{config}')

    # Create directories
    if not os.path.isdir(export_results):
        os.mkdir(export_results)

    if not os.path.isdir(os.path.join(export_results, "config")):
        os.mkdir(os.path.join(export_results, "config"))

    if not os.path.isdir(os.path.join(export_results, "plots")):
        os.mkdir(os.path.join(export_results, "plots"))

    with open(os.path.join(export_results, 'coupling_results.pkl'), "wb") as file:
        dump(output, file)

    with open(os.path.join(export_results, "config", 'config_eoles.json'), "w") as outfile:
        outfile.write(json.dumps(config_eoles, indent=4))

    # read and save the actual config file
    with open(config_resirf_path) as file:
        config_res_irf = json.load(file)

    with open(os.path.join(export_results, "config", 'config_resirf.json'), "w") as outfile:
        outfile.write(json.dumps(config_res_irf, indent=4))

    with open(os.path.join(export_results, "config", 'scenario_eoles_costs.json'), "w") as outfile:
        outfile.write(json.dumps(scenario_cost, indent=4))

    plot_simulation(output, save_path=os.path.join(export_results, "plots"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simulate resirf.')
    parser.add_argument("-c", "--config", type=str, help="configuration for resirf")
    parser.add_argument('--threshold', action=argparse.BooleanOptionalAction, help="whether we activate the threshold setting")
    parser.add_argument("--h2ccgt", action=argparse.BooleanOptionalAction, help="whether we allow H2-CCGT plants are not")

    args = parser.parse_args()
    config, calibration_threshold, h2ccgt = args.config, args.threshold, args.h2ccgt  # we select the config we are interested in
    # config, h2ccgt, calibration_threshold = os.path.join('eoles', 'inputs', 'config', 'config_resirf.json'), False, False

    run_optimization_scenario(config=config, calibration_threshold=calibration_threshold, h2ccgt=h2ccgt)

