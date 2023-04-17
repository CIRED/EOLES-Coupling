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
from project.utils import get_json
from project.write_output import plot_scenario, plot_compare_scenarios, indicator_policies
from project.model import get_inputs, social_planner

from project.model import create_logger, get_config, get_inputs
from eoles.model_resirf_coupling import ModelEOLES
from eoles.utils import get_config, get_pandas, calculate_annuities_resirf, modif_config_resirf, \
    config_resirf_exogenous, create_multiple_coupling_configs, create_multiple_coupling_configs2, modif_config_eoles
from eoles.write_output import plot_simulation, plot_blackbox_optimization, save_summary_pdf, comparison_simulations
import eoles.utils
from eoles.coupling_resirf_eoles import resirf_eoles_coupling_dynamic, optimize_blackbox_resirf_eoles_coupling, \
    calibration_price, get_energy_prices_and_taxes, resirf_eoles_coupling_greenfield
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
    "threshold_simple_premature10": "eoles/inputs/config/config_resirf_threshold_simple_premature10.json",
    "nolandlord": "eoles/inputs/config/config_resirf_nolandlord.json",
    "nomultifamily": "eoles/inputs/config/config_resirf_nomultifamily.json",
    "nolandlord_nomultifamily": "eoles/inputs/config/config_resirf_nolandlord_nomultifamily.json",
    "nolandlord_simple": "eoles/inputs/config/config_resirf_nolandlord_simple.json",
    "nomultifamily_simple": "eoles/inputs/config/config_resirf_nomultifamily_simple.json",
    "nolandlord_nomultifamily_simple": "eoles/inputs/config/config_resirf_nolandlord_nomultifamily_simple.json",
}

DICT_CONFIG_EOLES = {
    "eoles_classic": "eoles_coupling",
    "eoles_biogasS2": "eoles_coupling_biogasS2",
    "eoles_nobiogas": "eoles_coupling_nobiogas",
    "eoles_nobiogas_nohydrogen": "eoles_coupling_nobiogas_nohydrogen",
    "eoles_worst_case": "eoles_coupling_worst_case"
}


def save_simulation_results(output, buildings, name_config_coupling, config_coupling, config_eoles, config_resirf,
                            dict_optimizer, optimization=True):
    """Save simulation results."""
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

    with open(os.path.join(export_results, "config", 'config_resirf.json'), "w") as outfile:
        outfile.write(json.dumps(config_resirf, indent=4))

    with open(os.path.join(export_results, "config", 'config_coupling.json'), "w") as outfile:
        outfile.write(json.dumps(config_coupling, indent=4))

    with open(os.path.join(export_results, 'coupling_results.pkl'), "wb") as file:
        dump(output, file)

    if output is not None:  # we exclude the case where we did not find a way to not violate the carbon budget
        for key in output.keys():
            if key != 'Subvention heater' and key != 'Subvention insulation' and key != 'max_iter':
                key_save = '_'.join(key.split('(')[0].lower().split(' ')[:-1])
                output[key].to_csv(os.path.join(export_results, 'dataframes', f'{key_save}.csv'))

        buildings.path = os.path.join(export_results, "plots")
        plot_scenario(output["Output global ResIRF ()"], output["Stock global ResIRF ()"],
                      buildings)  # make ResIRF plots
        if optimization:
            plot_blackbox_optimization(dict_optimizer, save_path=os.path.join(export_results))

        if not "greenfield" in config_coupling.keys():  # si greenfield, on ne veut pas plotter l'évolution des quantités, car pas d'optimisation dynamique
            plot_simulation(output, save_path=os.path.join(export_results, "plots"))
            save_summary_pdf(path=export_results)  # saving summary as pdf
    return export_results, output["Output global ResIRF ()"]


def run_exogenous_scenario(config_coupling, name_config_coupling="default"):
    config_resirf = config_coupling["config_resirf"]
    config_eoles = eoles.utils.get_config(spec="eoles_coupling")
    config_eoles = modif_config_eoles(config_eoles, config_coupling)

    list_year = config_coupling["list_year"]
    list_trajectory_scc = config_coupling["list_trajectory_scc"]  # SCC trajectory

    h2ccgt = config_coupling["h2ccgt"]
    scenario_cost = config_coupling["scenario_cost_eoles"]
    if not h2ccgt:  # we do not allow h2 ccgt plants
        if "fix_capa" in scenario_cost.keys():
            scenario_cost["fix_capa"]["h2_ccgt"] = 0
        else:
            scenario_cost["fix_capa"] = {
                "h2_ccgt": 0
            }

    anticipated_demand_t10 = False
    if "anticipated_demand_t10" in config_coupling.keys():
        anticipated_demand_t10 = config_coupling["anticipated_demand_t10"]

    anticipated_scc = False
    if "anticipated_scc" in config_coupling.keys():
        anticipated_scc = config_coupling["anticipated_scc"]

    price_feedback = False
    if "price_feedback" in config_coupling.keys():
        price_feedback = config_coupling["price_feedback"]

    aggregated_potential = False
    if "aggregated_potential" in config_coupling.keys():
        aggregated_potential = config_coupling["aggregated_potential"]

    cofp = False
    if "cofp" in config_coupling.keys():
        cofp = config_coupling["cofp"]

    # initialization ResIRF
    buildings, inputs_dynamics, policies_heater, policies_insulation = ini_res_irf(
        path=os.path.join('eoles', 'outputs', 'ResIRF'),
        config=config_resirf)

    energy_taxes, energy_vta = get_energy_prices_and_taxes(config_resirf)
    calibration_elec_lcoe, calibration_elec_transport_distrib, calibration_gas, m_eoles = calibration_price(
        config_eoles, scc=100)
    config_coupling["calibration_elec_lcoe"] = calibration_elec_lcoe
    config_coupling["calibration_elec_transport_distrib"] = calibration_elec_transport_distrib
    config_coupling["calibration_naturalgas_lcoe"] = calibration_gas
    config_coupling["calibration_biogas_lcoe"] = 1.2

    if "greenfield" in config_coupling.keys():  # we optimize in a greenfield manner
        assert config_coupling[
            "greenfield"], "Parameter greenfield can only be True for the time being, when specified in config."
        print("Greenfield")
        output, buildings, dict_optimizer = resirf_eoles_coupling_greenfield(buildings, inputs_dynamics,
                                                                             policies_heater, policies_insulation,
                                                                             scc=775, scenario_cost=scenario_cost,
                                                                             config_eoles=config_eoles,
                                                                             config_coupling=config_coupling,
                                                                             add_CH4_demand=False,
                                                                             optimization=False,
                                                                             list_sub_heater=[0.0],
                                                                             list_sub_insulation=[0.0]
                                                                             )
    else:
        output, buildings, dict_optimizer = resirf_eoles_coupling_dynamic(buildings, inputs_dynamics,
                                                                          policies_heater, policies_insulation,
                                                                          list_year, list_trajectory_scc,
                                                                          scenario_cost,
                                                                          config_eoles=config_eoles,
                                                                          config_coupling=config_coupling,
                                                                          add_CH4_demand=False,
                                                                          anticipated_scc=anticipated_scc,
                                                                          anticipated_demand_t10=anticipated_demand_t10,
                                                                          optimization=False,
                                                                          list_sub_heater=[0.0, 0.0, 0.0, 0.0, 0.0],
                                                                          list_sub_insulation=[0.0, 0.0, 0.0, 0.0, 0.0],
                                                                          price_feedback=price_feedback,
                                                                          energy_taxes=energy_taxes,
                                                                          energy_vta=energy_vta,
                                                                          aggregated_potential=aggregated_potential,
                                                                          cofp=cofp)

    # Save results
    export_results, output_resirf = save_simulation_results(output, buildings, name_config_coupling, config_coupling, config_eoles, config_resirf,
                            dict_optimizer, optimization=True)
    return name_config_coupling, output_resirf, export_results


def run_optimization_scenario(config_coupling, name_config_coupling="default"):
    """

    :param config_coupling: dict
        Dictionary containing different configurations. For the time being, includes:
        configuration for resirf (market failures, threshold), consideration of H2 CCGT in EOLES, and number of iterations for optimization
    :return:
    Saves the output of the optimization + save plots
    """
    print(name_config_coupling)
    config_resirf_path = DICT_CONFIG_RESIRF["classic_simple"]
    with open(config_resirf_path) as file:  # load config_resirf
        config_resirf = json.load(file).get('Reference')
    config_resirf = modif_config_resirf(config_resirf, config_coupling, calibration=config_coupling[
        "calibration"])  # modif of this configuration file to consider coupling options

    config_eoles = eoles.utils.get_config(spec="eoles_coupling")
    config_eoles = modif_config_eoles(config_eoles, config_coupling)

    list_year = config_coupling["list_year"]
    list_trajectory_scc = config_coupling["list_trajectory_scc"]  # SCC trajectory

    acquisition_jitter = 0.01
    if "acquisition_jitter" in config_coupling.keys():
        acquisition_jitter = config_coupling["acquisition_jitter"]

    grid_initialize = False
    if "grid_initialize" in config_coupling.keys():
        grid_initialize = config_coupling["grid_initialize"]

    normalize_Y = True
    if "normalize_Y" in config_coupling.keys():
        normalize_Y = config_coupling["normalize_Y"]

    anticipated_demand_t10 = False
    if "anticipated_demand_t10" in config_coupling.keys():
        anticipated_demand_t10 = config_coupling["anticipated_demand_t10"]

    anticipated_scc = False
    if "anticipated_scc" in config_coupling.keys():
        anticipated_scc = config_coupling["anticipated_scc"]

    price_feedback = False
    if "price_feedback" in config_coupling.keys():
        price_feedback = config_coupling["price_feedback"]

    aggregated_potential = False
    if "aggregated_potential" in config_coupling["eoles"].keys():
        aggregated_potential = config_coupling["eoles"]["aggregated_potential"]

    cofp = False
    if "cofp" in config_coupling.keys():
        cofp = config_coupling["cofp"]

    # initialization ResIRF
    buildings, inputs_dynamics, policies_heater, policies_insulation = ini_res_irf(
        path=os.path.join('eoles', 'outputs', 'ResIRF'),
        config=config_resirf)

    energy_taxes, energy_vta = get_energy_prices_and_taxes(config_resirf)
    calibration_elec_lcoe, calibration_elec_transport_distrib, calibration_gas, m_eoles = calibration_price(
        config_eoles, scc=100)
    config_coupling["calibration_elec_lcoe"] = calibration_elec_lcoe
    config_coupling["calibration_elec_transport_distrib"] = calibration_elec_transport_distrib
    config_coupling["calibration_naturalgas_lcoe"] = calibration_gas
    config_coupling["calibration_biogas_lcoe"] = 1.2

    if "no_subsidies" in config_coupling.keys():  # we do not want to have any isolation and electrification
        assert config_coupling["no_subsidies"], "Parameter no_subsidies can only be True for the time being"
        print("no optimized subsidies")
        if "greenfield" in config_coupling.keys():  # we optimize in a greenfield manner
            assert config_coupling[
                "greenfield"], "Parameter greenfield can only be True for the time being, when specified in config."
            print("Greenfield")
            output, buildings, dict_optimizer = resirf_eoles_coupling_greenfield(buildings, inputs_dynamics,
                                                                                 policies_heater, policies_insulation,
                                                                                 scc=775, scenario_cost=config_coupling["scenario_cost_eoles"],
                                                                                 config_eoles=config_eoles,
                                                                                 config_coupling=config_coupling,
                                                                                 add_CH4_demand=False,
                                                                                 optimization=False,
                                                                                 list_sub_heater=[0.0],
                                                                                 list_sub_insulation=[0.0]
                                                                                 )
        else:
            output, buildings, dict_optimizer = resirf_eoles_coupling_dynamic(buildings, inputs_dynamics,
                                                                              policies_heater, policies_insulation,
                                                                              list_year, list_trajectory_scc,
                                                                              config_coupling["scenario_cost_eoles"],
                                                                              config_eoles=config_eoles,
                                                                              config_coupling=config_coupling,
                                                                              add_CH4_demand=False,
                                                                              anticipated_scc=anticipated_scc,
                                                                              anticipated_demand_t10=anticipated_demand_t10,
                                                                              optimization=False,
                                                                              list_sub_heater=[0.0, 0.0, 0.0, 0.0, 0.0],
                                                                              list_sub_insulation=[0.0, 0.0, 0.0, 0.0,
                                                                                                   0.0],
                                                                              price_feedback=price_feedback,
                                                                              energy_taxes=energy_taxes,
                                                                              energy_vta=energy_vta,
                                                                              aggregated_potential=aggregated_potential,
                                                                              cofp=cofp)
    elif "greenfield" in config_coupling.keys():  # we optimize in a greenfield manner
        assert config_coupling[
            "greenfield"], "Parameter greenfield can only be True for the time being, when specified in config."
        print("Greenfield")
        output, buildings, dict_optimizer = resirf_eoles_coupling_greenfield(buildings, inputs_dynamics,
                                                                             policies_heater, policies_insulation,
                                                                             scc=775, scenario_cost=config_coupling["scenario_cost_eoles"],
                                                                             config_eoles=config_eoles,
                                                                             config_coupling=config_coupling,
                                                                             add_CH4_demand=False,
                                                                             optimization=True,
                                                                             acquisition_jitter=acquisition_jitter,
                                                                             grid_initialize=grid_initialize,
                                                                             normalize_Y=normalize_Y
                                                                             )
    else:  # we optimize the value of subsidy
        output, buildings, dict_optimizer = resirf_eoles_coupling_dynamic(buildings, inputs_dynamics,
                                                                          policies_heater, policies_insulation,
                                                                          list_year, list_trajectory_scc, config_coupling["scenario_cost_eoles"],
                                                                          config_eoles=config_eoles,
                                                                          config_coupling=config_coupling,
                                                                          add_CH4_demand=False,
                                                                          anticipated_scc=anticipated_scc,
                                                                          anticipated_demand_t10=anticipated_demand_t10,
                                                                          optimization=True,
                                                                          price_feedback=price_feedback,
                                                                          energy_taxes=energy_taxes,
                                                                          energy_vta=energy_vta,
                                                                          acquisition_jitter=acquisition_jitter,
                                                                          grid_initialize=grid_initialize,
                                                                          normalize_Y=normalize_Y,
                                                                          aggregated_potential=aggregated_potential,
                                                                          cofp=cofp)

    # Save results
    export_results, output_resirf = save_simulation_results(output, buildings, name_config_coupling, config_coupling, config_eoles, config_resirf,
                            dict_optimizer, optimization=True)

    return name_config_coupling, output_resirf, export_results


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

    config_resirf, config_eoles_spec = config_coupling["config_resirf"], DICT_CONFIG_EOLES[
        config_coupling["config_eoles"]]
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

    max_iter, one_shot_setting, fix_sub_heater = config_coupling["max_iter"], config_coupling["one_shot_setting"], \
                                                 config_coupling["fix_sub_heater"]

    sub_design, health, discount_rate, rebound, carbon_constraint = config_coupling["sub_design"], config_coupling[
        "health"], \
                                                                    config_coupling["discount_rate"], config_coupling[
                                                                        "rebound"], config_coupling["carbon_constraint"]
    t10 = None
    if "t10" in config_coupling.keys():
        t10 = config_coupling["t10"]
    output, dict_optimizer = resirf_eoles_coupling_dynamic_no_opti(list_sub_heater=list_subsidies_heater,
                                                                   list_sub_insulation=list_subsidies_insulation,
                                                                   buildings=buildings, energy_prices=energy_prices,
                                                                   taxes=taxes,
                                                                   cost_heater=cost_heater,
                                                                   cost_insulation=cost_insulation,
                                                                   flow_built=flow_built, post_inputs=post_inputs,
                                                                   policies_heater=policies_heater,
                                                                   policies_insulation=policies_insulation,
                                                                   list_year=list_year,
                                                                   list_trajectory_scc=list_trajectory_scc,
                                                                   scenario_cost=scenario_cost,
                                                                   config_eoles=config_eoles,
                                                                   add_CH4_demand=False,
                                                                   one_shot_setting=one_shot_setting,
                                                                   sub_design=sub_design,
                                                                   health=health, carbon_constraint=carbon_constraint,
                                                                   discount_rate=discount_rate,
                                                                   rebound=rebound,
                                                                   technical_progress=technical_progress,
                                                                   financing_cost=financing_cost)

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


def run_multiple_configs(dict_config, cpu: int, exogenous=True, reference=None, greenfield=False, health=True,
                         carbon_constraint=True, folder_comparison=os.path.join("eoles/outputs/comparison")):
    """Run multiple configs in parallel"""
    logger.info('Scenarios: {}'.format(', '.join(dict_config.keys())))
    try:
        logger.info('Launching processes')
        if not exogenous:
            with Pool(cpu) as pool:
                results = pool.starmap(run_optimization_scenario,
                                       zip(dict_config.values(), [n for n in dict_config.keys()]))
        else:
            with Pool(cpu) as pool:
                results = pool.starmap(run_exogenous_scenario,
                                       zip(dict_config.values(), [n for n in dict_config.keys()]))
        results_resirf = {i[0]: i[1] for i in results}
        results_general = {i[0]: i[2] for i in results}

        # Plots ResIRF
        date = datetime.datetime.now().strftime("%m%d_%H%M%S")
        folder = os.path.join(folder_comparison, f'{date}')
        if not os.path.isdir(folder):
            os.mkdir(folder)
        plot_compare_scenarios(results_resirf, folder=folder)

        # config_policies = get_json('project/input/policies/cba_inputs.json')
        # if 'Reference' in results_resirf.keys() and len(results_resirf.keys()) > 1 and config_policies is not None:
        #     indicator_policies(results_resirf, folder, config_policies, policy_name=None)

        # Plots coupling
        if reference is not None:
            assert reference in results_general.keys(), "Name of reference simulation should be one of the simulations."
            annualized_system_costs_df, total_system_costs_df, consumption_savings_tot_df, complete_system_costs_2050_df = comparison_simulations(
                results_general, ref=reference, greenfield=greenfield, health=health, save_path=folder, carbon_constraint=carbon_constraint)
    except Exception as e:
        logger.exception(e)
        raise e
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate coupling.')
    # parser.add_argument("-c", "--config", type=str, default="classic", help="configuration for resirf")
    # parser.add_argument("-i", "--maxiter", type=int, default=20, help="maximum iterations for blackbox optimization")
    # parser.add_argument('--threshold', action=argparse.BooleanOptionalAction, default=False,
    #                     help="whether we activate the threshold setting")
    # parser.add_argument("--h2ccgt", action=argparse.BooleanOptionalAction, default=False,
    #                     help="whether we allow H2-CCGT plants are not")
    parser.add_argument("--cpu", type=int, default=3, help="CPUs for multiprocessing")

    args = parser.parse_args()
    cpu = args.cpu  # we select the config we are interested in

    # no HC
    DICT_CONFIGS_3 = {
        'noHC_no_subsidy_heater_centralized_private': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": False,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'noHC_centralized': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 100,
            'sub_design': None,
            "health": False,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.03,
            'scenario_cost_eoles': {}
        },
        'noHC_centralized_social': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': True,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 100,
            'sub_design': None,
            "health": False,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.03,
            'scenario_cost_eoles': {}
        },
        'noHC_centralized_supply': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': True,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 100,
            'sub_design': None,
            "health": False,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.03,
            'scenario_cost_eoles': {}
        },
        'noHC_no_subsidy_heater_centralized_private_carbonbudget': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": False,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'noHC_centralized_carbonbudget': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 100,
            'sub_design': None,
            "health": False,  # on inclut les coûts de santé
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
        },
        'noHC_no_subsidy_heater_centralized_private_carbonbudget_greenfield': {
            'greenfield': True,
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": False,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_centralized_private_carbonbudget': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
    }

    # carbon budget S2
    DICT_CONFIGS_4 = {
        'centralized_carbonbudget': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 100,
            'sub_design': None,
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
        },
        'no_subsidy_insulation_carbonbudget_S2': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_biogasS2",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': True,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_carbonbudget_S2': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_biogasS2",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_centralized_private_carbonbudget_S2': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_biogasS2",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'centralized_carbonbudget_S2': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_biogasS2",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 100,
            'sub_design': None,
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
        },
        'uniform_carbonbudget_S2': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_biogasS2",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 100,
            'sub_design': None,
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
        },
        'uniform_carbonbudget_sensitivity': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 50,
            'sub_design': None,
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
        },
        'noHC_uniform_sensitivity': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 50,
            'sub_design': None,
            "health": False,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.03,
            'scenario_cost_eoles': {}
        },
    }

    DICT_CONFIGS_5 = {
        'noHC_centralized_carbonbudget_greenfield': {
            'greenfield': True,
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 100,
            'sub_design': None,
            "health": False,  # on inclut les coûts de santé
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
        },
        'no_subsidy': {
            'no_renovation': True,
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 30,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': True,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.03,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_insulation': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
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
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_centralized_private': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'uniform': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 100,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.03,
            'scenario_cost_eoles': {}
        },
        'GR': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 100,
            'sub_design': "global_renovation",
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.03,
            'scenario_cost_eoles': {}
        },
        'centralized': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 100,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": False,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.03,
            'scenario_cost_eoles': {}
        },
    }

    DICT_CONFIGS_SUPPLY = {
        'GR_carbonbudget_supply': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': True,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 60,
            'sub_design': "global_renovation",
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
        },
        'GR_carbonbudget_sensitivity': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': "global_renovation",
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
        },
        'noHC_uniform_carbonbudget': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': True,
            'supply_heater': False,
            'rational_behavior': False,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 60,
            'sub_design': None,
            "health": False,  # on inclut les coûts de santé
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
        },
        'noHC_GR_carbonbudget': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': True,
            'supply_heater': False,
            'rational_behavior': False,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 60,
            'sub_design': "global_renovation",
            "health": False,  # on inclut les coûts de santé
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
        },
        'GR_carbonbudget_S2': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_biogasS2",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 60,
            'sub_design': "global_renovation",
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'GR_carbonbudget_S2_supply': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_biogasS2",  # includes costs assumptions
            'supply_insulation': True,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 60,
            'sub_design': "global_renovation",
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'uniform_carbonbudget_S2_supply': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_biogasS2",  # includes costs assumptions
            'supply_insulation': True,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 60,
            'sub_design': None,
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
    }

    # carbon budget
    DICT_CONFIGS_6 = {
        'no_subsidy_insulation_carbonbudget': {
            'config_resirf': "classic_simple",
            'calibration': os.path.join("eoles/inputs/calibration_20230321.pkl"),
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': True,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_carbonbudget': {
            'config_resirf': "classic_simple",
            'calibration': os.path.join("eoles/inputs/calibration_20230321.pkl"),
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_centralized_carbonbudget': {
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'uniform_carbonbudget': {
            'config_resirf': "classic_simple",
            'calibration': os.path.join("eoles/inputs/calibration_20230321.pkl"),
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': None,
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
        },
        'GR_carbonbudget': {
            'config_resirf': "classic_simple",
            'calibration': os.path.join("eoles/inputs/calibration_20230321.pkl"),
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': "global_renovation",
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
        },
        'GR_noMF_carbonbudget': {
            'config_resirf': "classic_simple",
            'calibration': os.path.join("eoles/inputs/calibration_20230321.pkl"),
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': "global_renovation",
            'no_MF': True,
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
        },
        'GR_FGE_carbonbudget': {
            'config_resirf': "classic_simple",
            'calibration': os.path.join("eoles/inputs/calibration_20230321.pkl"),
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': "global_renovation_fge",
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
        },
        'efficiency100_carbonbudget': {
            'config_resirf': "classic_simple",
            'calibration': os.path.join("eoles/inputs/calibration_20230321.pkl"),
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': "efficiency_100",
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
    }

    # carbon budget greenfield
    DICT_CONFIGS_8 = {
        'centralized_social_carbonbudget_greenfield': {
            'greenfield': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': True,
            'premature_replacement': 3,
            'h2ccgt': True,
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
        },
        'centralized_GR_carbonbudget_greenfield': {
            'greenfield': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': "global_renovation",
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.02,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.03,
            'scenario_cost_eoles': {}
        },
        'centralized_carbonbudget_greenfield_dr2': {
            'greenfield': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.02,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.03,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_insulation_carbonbudget_greenfield_dr2': {
            'greenfield': True,
            'config_resirf': "classic_simple",
            'calibration': os.path.join("eoles/inputs/calibration_20230321.pkl"),
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.02,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': True,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_carbonbudget_greenfield_dr2': {
            'greenfield': True,
            'config_resirf': "classic_simple",
            'calibration': os.path.join("eoles/inputs/calibration_20230321.pkl"),
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.02,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_centralized_carbonbudget_greenfield_dr2': {
            'greenfield': True,
            'config_resirf': "classic_simple",
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.02,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'uniform_carbonbudget_greenfield_dr2': {
            'greenfield': True,
            'config_resirf': "classic_simple",
            'calibration': os.path.join("eoles/inputs/calibration_20230321.pkl"),
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.02,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.03,
            'scenario_cost_eoles': {}
        },
        'GR_carbonbudget_greenfield_dr2': {
            'greenfield': True,
            'config_resirf': "classic_simple",
            'calibration': os.path.join("eoles/inputs/calibration_20230321.pkl"),
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': "global_renovation",
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.02,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.03,
            'scenario_cost_eoles': {}
        },
    }

    # classic

    # classic with prices non constant
    DICT_CONFIGS_9 = {
        'uniform_carbonbudget': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": True,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'max_iter': 150,
            'sub_design': None,
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
        },
        'centralized_social_carbonbudget': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": True
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': True,
            'premature_replacement': 3,
            'max_iter': 150,
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
        },
        'centralized_GR_carbonbudget': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": True,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'max_iter': 150,
            'sub_design': "global_renovation",
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
        },
        'centralized_carbonbudget': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": True,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'max_iter': 130,
            'sub_design': None,
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
        },
        'no_subsidy_insulation_carbonbudget': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": True,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'max_iter': 50,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': True,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_carbonbudget': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": True,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'max_iter': 50,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_centralized_carbonbudget': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": True,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'max_iter': 50,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'GR_carbonbudget': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": True,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'max_iter': 150,
            'sub_design': "global_renovation",
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
    }

    DICT_CONFIGS_9_noagg = {
        'uniform_carbonbudget_noagg': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": False,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'max_iter': 150,
            'sub_design': None,
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
        },
        'centralized_social_carbonbudget_noagg': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": False
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': True,
            'premature_replacement': 3,
            'max_iter': 150,
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
        },
        'centralized_GR_carbonbudget_noagg': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": False,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'max_iter': 150,
            'sub_design': "global_renovation",
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
        },
        'centralized_carbonbudget_noagg': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": False,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'max_iter': 130,
            'sub_design': None,
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
        },
        'no_subsidy_insulation_carbonbudget_noagg': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": False,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'max_iter': 50,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': True,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_carbonbudget_noagg': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": False,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'max_iter': 50,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_centralized_carbonbudget_noagg': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": False,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'max_iter': 50,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'GR_carbonbudget_noagg': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": False,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'max_iter': 150,
            'sub_design': "global_renovation",
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
    }

    DICT_CONFIGS_9_agg_N1 = {
        'uniform_carbonbudget_agg_N1': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": True,
                'N1': True,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'max_iter': 150,
            'sub_design': None,
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
        },
        'centralized_social_carbonbudget_agg_N1': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": True,
                'N1': True,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': True,
            'premature_replacement': 3,
            'max_iter': 150,
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
        },
        'centralized_GR_carbonbudget_agg_N1': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": True,
                'N1': True,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'max_iter': 150,
            'sub_design': "global_renovation",
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
        },
        'centralized_carbonbudget_agg_N1': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": True,
                'N1': True,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'max_iter': 130,
            'sub_design': None,
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
        },
        'no_subsidy_insulation_carbonbudget_agg_N1': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": True,
                'N1': True,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'max_iter': 50,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': True,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_carbonbudget_agg_N1': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": True,
                'N1': True,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'max_iter': 50,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_centralized_carbonbudget_agg_N1': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": True,
                'N1': True,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'max_iter': 50,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'GR_carbonbudget_agg_N1': {
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": True,
                'N1': True
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'max_iter': 150,
            'sub_design': "global_renovation",
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
    }

    DICT_CONFIGS_9_greenfield = {
        'uniform_carbonbudget_greenfield': {
            'greenfield': True,
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": False,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'max_iter': 150,
            'sub_design': None,
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
        },
        'centralized_social_carbonbudget_greenfield': {
            'greenfield': True,
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": False
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': True,
            'premature_replacement': 3,
            'max_iter': 150,
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
        },
        'centralized_GR_carbonbudget_greenfield': {
            'greenfield': True,
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": False,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'max_iter': 150,
            'sub_design': "global_renovation",
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
        },
        'centralized_carbonbudget_greenfield': {
            'greenfield': True,
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": False,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'max_iter': 130,
            'sub_design': None,
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
        },
        'no_subsidy_insulation_carbonbudget_greenfield': {
            'greenfield': True,
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": False,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'max_iter': 50,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': True,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_carbonbudget_greenfield': {
            'greenfield': True,
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": False,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'max_iter': 50,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_centralized_carbonbudget_greenfield': {
            'greenfield': True,
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": False,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'max_iter': 50,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'GR_carbonbudget_greenfield': {
            'greenfield': True,
            "calibration": None,
            "eoles": {
                "biomass_potential": "S3",
                "aggregated_potential": False,
            },
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'max_iter': 150,
            'sub_design': "global_renovation",
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
    }

    # classic
    DICT_CONFIGS_9_prices_greenfield = {
        'centralized_social_carbonbudget_prices_greenfield': {
            'prices_constant': False,
            'greenfield': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': True,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 150,
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
        },
        'centralized_GR_carbonbudget_prices_greenfield': {
            'prices_constant': False,
            'greenfield': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 150,
            'sub_design': "global_renovation",
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
        },
        'centralized_carbonbudget_prices_greenfield': {
            'prices_constant': False,
            'greenfield': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': None,
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
        },
        'no_subsidy_insulation_carbonbudget_prices_greenfield': {
            'prices_constant': False,
            'greenfield': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 50,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': True,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_carbonbudge_prices_greenfieldt': {
            'prices_constant': False,
            'greenfield': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 50,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_centralized_carbonbudget_prices_greenfield': {
            'prices_constant': False,
            'greenfield': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 50,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'uniform_carbonbudget_prices_greenfield': {
            'prices_constant': False,
            'greenfield': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 150,
            'sub_design': None,
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
        },
        'GR_carbonbudge_prices_greenfieldt': {
            'prices_constant': False,
            'greenfield': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 150,
            'sub_design': "global_renovation",
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
    }

    DICT_CONFIGS_10 = {
        'centralized_social_carbonbudget_S2': {
            'aggregated_potential': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_biogasS2",
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': True,
            'premature_replacement': 3,
            'h2ccgt': True,
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
        },
        'centralized_GR_carbonbudget_S2': {
            'aggregated_potential': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_biogasS2",
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': "global_renovation",
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
        },
        'centralized_carbonbudget_S2': {
            'aggregated_potential': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_biogasS2",
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': None,
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
        },
        'no_subsidy_insulation_carbonbudget_S2': {
            'aggregated_potential': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_biogasS2",
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': True,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_carbonbudget_S2': {
            'aggregated_potential': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_biogasS2",
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_centralized_carbonbudget_S2': {
            'aggregated_potential': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_biogasS2",
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'uniform_carbonbudget_S2': {
            'aggregated_potential': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_biogasS2",
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': None,
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
        },
        'GR_carbonbudget_S2': {
            'aggregated_potential': True,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_biogasS2",
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': "global_renovation",
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
    }

    DICT_CONFIGS_11 = {
        'centralized_social_carbonbudget_greenfield_S2': {
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_biogasS2",
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': True,
            'premature_replacement': 3,
            'h2ccgt': True,
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
        },
        'centralized_GR_carbonbudget_greenfield_S2': {
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_biogasS2",
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': "global_renovation",
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
        },
        'centralized_carbonbudget_greenfield_S2': {
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_biogasS2",
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': None,
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
        },
        'no_subsidy_insulation_carbonbudget_greenfield_S2': {
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_biogasS2",
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': False,
            'fix_sub_insulation': True,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_carbonbudget_greenfield_S2': {
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_biogasS2",
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'no_subsidy_heater_centralized_carbonbudget_greenfield_S2': {
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_biogasS2",
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 40,
            'sub_design': None,
            "health": True,  # on inclut les coûts de santé
            "discount_rate": 0.032,
            "rebound": True,
            "carbon_constraint": True,
            'one_shot_setting': False,
            'fix_sub_heater': True,
            'fix_sub_insulation': False,
            'list_year': [2025, 2030, 2035, 2040, 2045],
            'list_trajectory_scc': [250, 350, 500, 650, 775],
            'acquisition_jitter': 0.01,
            'scenario_cost_eoles': {}
        },
        'uniform_carbonbudget_greenfield_S2': {
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_biogasS2",
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': None,
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
        },
        'GR_carbonbudget_greenfield_S2': {
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_biogasS2",
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': "global_renovation",
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
        },
        'uniform_carbonbudget_oldpotential': {
            'aggregated_potential': False,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': None,
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
        },
        'centralized_carbonbudget_oldpotential': {
            'aggregated_potential': False,
            'config_resirf': "classic_simple",
            "calibration": None,
            "config_eoles": "eoles_classic",  # includes costs assumptions
            'supply_insulation': False,
            'supply_heater': False,
            'rational_behavior': True,
            'social': False,
            'premature_replacement': 3,
            'h2ccgt': True,
            'max_iter': 130,
            'sub_design': None,
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
        },
    }

    ########## Test exogenous configurations
    config_coupling = {
        'no_subsidies': True,
        'aggregated_potential': True,
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
        "health": True,  # on inclut les coûts de santé
        "discount_rate": 0.032,
        "rebound": True,
        "carbon_constraint": False,
        'fix_sub_heater': False,
        'fix_sub_insulation': False,
        'list_year': [2025, 2030, 2035, 2040, 2045],
        'list_trajectory_scc': [250, 350, 500, 650, 775],
        'scenario_cost_eoles': {}
    }

    # config_resirf_path, config_eoles_spec = DICT_CONFIG_RESIRF[config_coupling["config_resirf"]], DICT_CONFIG_EOLES[config_coupling["config_eoles"]]
    # with open(config_resirf_path) as file:  # load config_resirf
    #     config_resirf = json.load(file).get('Reference')
    # config_resirf = modif_config_resirf(config_resirf, config_coupling, calibration=config_coupling["calibration"])  # modif of this configuration file to consider coupling options
    #
    # sensitivity = {
    #     'Reference': {
    #         "policies": "project/input/policies/current/policies_ref.json"
    #     },
    #     'No policy': {
    #         'no_policy': True
    #     },
    #     "Ambitious": {
    #         "policies": "project/input/policies/policies_ambitious.json"
    #     },
    #     # "Ambitious Price feedback": {
    #     #     "policies": "project/input/policies/policies_ambitious.json",
    #     #     "price_feedback": True
    #     # }
    # }
    # dict_config_coupling = create_multiple_coupling_configs2(sensitivity, config_resirf, config_coupling)
    # # dict_config_resirf = config_resirf_exogenous(sensitivity=sensitivity, config_resirf=config_resirf)
    # # dict_config_coupling = create_multiple_coupling_configs(dict_config_resirf=dict_config_resirf, config_coupling=config_coupling)

    results = run_multiple_configs(DICT_CONFIGS_9_agg_N1, cpu=cpu, exogenous=False, reference=None, greenfield=False,
                                   health=True, carbon_constraint=True)

    # dict_output = {"Reference": os.path.join("eoles/outputs/0407_134331_Reference"),
    #                "Ambitious": os.path.join("eoles/outputs/0407_134347_Ambitious"),
    #                }
    # annualized_system_costs_df, total_system_costs_df, consumption_savings_tot_df, complete_system_costs_2050_df = comparison_simulations(
    #     dict_output, ref="Reference", greenfield=False, health=True,
    #     save_path=os.path.join("eoles/outputs/comparison/test"), carbon_constraint=False)
