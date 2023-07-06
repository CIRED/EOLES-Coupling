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
    modif_config_eoles, modif_config_coupling, create_configs_coupling, create_default_options, create_optimization_param, \
    create_coupling_param, extract_subsidy_value
from eoles.write_output import plot_simulation, plot_blackbox_optimization, save_summary_pdf, comparison_simulations
import eoles.utils
from eoles.coupling_resirf_eoles import resirf_eoles_coupling_dynamic, optimize_blackbox_resirf_eoles_coupling, \
    calibration_price, get_energy_prices_and_taxes, resirf_eoles_coupling_greenfield
import logging
import argparse
from pathlib import Path

from matplotlib import pyplot as plt

LOG_FORMATTER = '%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # replace NOTSET with INFO
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

        if optimization:
            plot_blackbox_optimization(dict_optimizer, save_path=os.path.join(export_results))

        buildings.path = os.path.join(export_results, "plots")
        plot_scenario(output["Output global ResIRF ()"], output["Stock global ResIRF ()"],
                      buildings)  # make ResIRF plots

        if not "greenfield" in config_coupling.keys():  # si greenfield, on ne veut pas plotter l'évolution des quantités, car pas d'optimisation dynamique
            plot_simulation(output, save_path=os.path.join(export_results, "plots"))
            save_summary_pdf(path=export_results)  # saving summary as pdf
    return export_results, output["Output global ResIRF ()"]


def run_exogenous_scenario(config_coupling, name_config_coupling="default"):
    """Function used to run the scenario without optimization. We assume here that config_resirf is already a
    dictionary which was modified before."""
    config_resirf = config_coupling["config_resirf"]
    config_eoles = eoles.utils.get_config(spec="eoles_coupling")
    config_eoles, config_coupling = modif_config_eoles(config_eoles, config_coupling)

    list_year = config_coupling["list_year"]
    list_trajectory_scc = config_coupling["list_trajectory_scc"]  # SCC trajectory

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
        config=config_resirf,
        level_logger=logging.NOTSET
    )

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
                                                                          scenario_cost=config_coupling["scenario_cost_eoles"],
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
    config_resirf_path = DICT_CONFIG_RESIRF["classic_simple"]  # classical ResIRF configuration
    with open(config_resirf_path) as file:  # load config_resirf
        config_resirf = json.load(file).get('Reference')
    config_resirf = modif_config_resirf(config_resirf, config_coupling)  # modif of this configuration file to take into account options specified in coupling configuration

    config_eoles = eoles.utils.get_config(spec="eoles_coupling")
    config_eoles, config_coupling = modif_config_eoles(config_eoles, config_coupling)  # modif of this configuration file to take into account options specified in coupling configuration

    list_year = config_coupling["list_year"]
    list_trajectory_scc = config_coupling["list_trajectory_scc"]  # SCC trajectory

    default_config = create_default_options(config_coupling)  # update of default options for the optimization framework
    optimparam = create_optimization_param(default_config)
    couplingparam = create_coupling_param(default_config)

    # initialization ResIRF
    buildings, inputs_dynamics, policies_heater, policies_insulation = ini_res_irf(
        path=os.path.join('eoles', 'outputs', 'ResIRF'),
        config=config_resirf,
        level_logger=logging.NOTSET
        )

    energy_taxes, energy_vta = get_energy_prices_and_taxes(config_resirf)
    calibration_elec_lcoe, calibration_elec_transport_distrib, calibration_gas, m_eoles = calibration_price(
        config_eoles, scc=100)
    config_coupling["calibration_elec_lcoe"] = calibration_elec_lcoe
    config_coupling["calibration_elec_transport_distrib"] = calibration_elec_transport_distrib
    config_coupling["calibration_naturalgas_lcoe"] = calibration_gas
    config_coupling["calibration_biogas_lcoe"] = 1.2

    # Specification of options for optimization
    list_sub_heater, list_sub_insulation = None, None
    optim_eoles = config_coupling['optim_eoles']
    optimization = True

    if config_coupling["no_subsidies"]:
        print("no optimized subsidies")
        optimization = False
        if config_coupling["greenfield"]:
            list_sub_heater, list_sub_insulation = [0.0], [1.0]
        else:
            list_sub_heater, list_sub_insulation = [0.0 for i in range(5)], [1.0 for i in range(5)]

    if config_coupling['subsidies_specified']:
        print('Subsidies specified')
        optimization = False
        if config_coupling["greenfield"]:
            assert len(config_coupling['subsidies_heater']) == 1, "Subsidies are not correctly specified in the greenfield setting."
        else:
            assert len(config_coupling['subsidies_heater']) == len(
                config_coupling['list_year']), "Subsidies are not correctly specified in the multistep setting."
        list_sub_heater, list_sub_insulation = config_coupling['subsidies_heater'], config_coupling['subsidies_insulation']

    if config_coupling["greenfield"]:  # we optimize in a greenfield manner
        print("Greenfield")
        output, buildings, dict_optimizer = resirf_eoles_coupling_greenfield(buildings, inputs_dynamics,
                                                                             policies_heater, policies_insulation,
                                                                             scc=775, scenario_cost=config_coupling["scenario_cost_eoles"],
                                                                             config_eoles=config_eoles,
                                                                             config_coupling=config_coupling,
                                                                             add_CH4_demand=False,
                                                                             optimization=optimization,
                                                                             list_sub_heater=list_sub_heater,
                                                                             list_sub_insulation=list_sub_insulation,
                                                                             optimparam=optimparam,
                                                                             couplingparam=couplingparam,
                                                                             optim_eoles=optim_eoles
                                                                             )
    else:  # we optimize the value of subsidy
        output, buildings, dict_optimizer = resirf_eoles_coupling_dynamic(buildings, inputs_dynamics,
                                                                          policies_heater, policies_insulation,
                                                                          list_year, list_trajectory_scc, config_coupling["scenario_cost_eoles"],
                                                                          config_eoles=config_eoles,
                                                                          config_coupling=config_coupling,
                                                                          add_CH4_demand=False,
                                                                          optimparam=optimparam,
                                                                          couplingparam=couplingparam,
                                                                          optimization=optimization,
                                                                          list_sub_heater=list_sub_heater,
                                                                          list_sub_insulation=list_sub_insulation,
                                                                          energy_taxes=energy_taxes,
                                                                          energy_vta=energy_vta,
                                                                          optim_eoles=optim_eoles)


    # if "no_subsidies" in config_coupling.keys():  # we do not want to have any insolation and electrification
    #     assert config_coupling["no_subsidies"], "Parameter no_subsidies can only be True for the time being"
    #     print("no optimized subsidies")
    #     if "greenfield" in config_coupling.keys():  # we optimize in a greenfield manner
    #         assert config_coupling[
    #             "greenfield"], "Parameter greenfield can only be True for the time being, when specified in config."
    #         print("Greenfield")
    #         output, buildings, dict_optimizer = resirf_eoles_coupling_greenfield(buildings, inputs_dynamics,
    #                                                                              policies_heater, policies_insulation,
    #                                                                              scc=775, scenario_cost=config_coupling["scenario_cost_eoles"],
    #                                                                              config_eoles=config_eoles,
    #                                                                              config_coupling=config_coupling,
    #                                                                              add_CH4_demand=False,
    #                                                                              optimization=False,
    #                                                                              list_sub_heater=[0.0],
    #                                                                              list_sub_insulation=[1.0],
    #                                                                              optimparam=optimparam,
    #                                                                              couplingparam=couplingparam
    #                                                                              )
    #     else:
    #         output, buildings, dict_optimizer = resirf_eoles_coupling_dynamic(buildings, inputs_dynamics,
    #                                                                           policies_heater, policies_insulation,
    #                                                                           list_year, list_trajectory_scc,
    #                                                                           config_coupling["scenario_cost_eoles"],
    #                                                                           config_eoles=config_eoles,
    #                                                                           config_coupling=config_coupling,
    #                                                                           add_CH4_demand=False,
    #                                                                           couplingparam=couplingparam,
    #                                                                           optimization=False,
    #                                                                           list_sub_heater=[0.0 for i in range(5)],
    #                                                                           list_sub_insulation=[1.0 for i in range(5)],
    #                                                                           energy_taxes=energy_taxes,
    #                                                                           energy_vta=energy_vta)
    # elif 'subsidies_specified' in config_coupling.keys():
    #     print('Subsidies specified')
    #     assert config_coupling['subsidies_specified'], "Parameter subsidies_specified can only be True for the time being, when specified in config."
    #     assert 'subsidies_insulation' in config_coupling.keys()
    #     assert 'subsidies_heater' in config_coupling.keys()
    #     if "greenfield" in config_coupling.keys():  # we optimize in a greenfield manner
    #         assert config_coupling["greenfield"], "Parameter greenfield can only be True for the time being, when specified in config."
    #         print("Greenfield")
    #         assert len(config_coupling['subsidies_heater']) == 1, "Subsidies are not correctly specified in the greenfield setting."
    #         output, buildings, dict_optimizer = resirf_eoles_coupling_greenfield(buildings, inputs_dynamics,
    #                                                                              policies_heater, policies_insulation,
    #                                                                              scc=775, scenario_cost=config_coupling["scenario_cost_eoles"],
    #                                                                              config_eoles=config_eoles,
    #                                                                              config_coupling=config_coupling,
    #                                                                              add_CH4_demand=False,
    #                                                                              optimization=False,
    #                                                                              list_sub_heater=config_coupling['subsidies_heater'],
    #                                                                              list_sub_insulation=config_coupling['subsidies_insulation'],
    #                                                                              couplingparam=couplingparam
    #                                                                              )
    #     else:
    #         assert len(config_coupling['subsidies_heater']) == len(config_coupling['list_year']), "Subsidies are not correctly specified in the multistep setting."
    #         output, buildings, dict_optimizer = resirf_eoles_coupling_dynamic(buildings, inputs_dynamics,
    #                                                                           policies_heater, policies_insulation,
    #                                                                           list_year, list_trajectory_scc,
    #                                                                           config_coupling["scenario_cost_eoles"],
    #                                                                           config_eoles=config_eoles,
    #                                                                           config_coupling=config_coupling,
    #                                                                           add_CH4_demand=False,
    #                                                                           couplingparam=couplingparam,
    #                                                                           optimization=False,
    #                                                                           list_sub_heater=config_coupling['subsidies_heater'],
    #                                                                           list_sub_insulation=config_coupling['subsidies_insulation'],
    #                                                                           energy_taxes=energy_taxes,
    #                                                                           energy_vta=energy_vta)
    # elif 'optim_eoles' in config_coupling.keys():
    #     print('Optimization ResIRF - no optimization EOLES')
    #     assert not config_coupling['optim_eoles'], "Parameter optim_eoles can only be False for the time being, when specified in config."
    #     if "greenfield" in config_coupling.keys():  # we optimize in a greenfield manner
    #         assert config_coupling["greenfield"], "Parameter greenfield can only be True for the time being, when specified in config."
    #         print("Greenfield")
    #         output, buildings, dict_optimizer = resirf_eoles_coupling_greenfield(buildings, inputs_dynamics,
    #                                                                              policies_heater, policies_insulation,
    #                                                                              scc=775, scenario_cost=config_coupling["scenario_cost_eoles"],
    #                                                                              config_eoles=config_eoles,
    #                                                                              config_coupling=config_coupling,
    #                                                                              add_CH4_demand=False,
    #                                                                              optimization=True,
    #                                                                              optimparam=optimparam,
    #                                                                              couplingparam=couplingparam, optim_eoles=False
    #                                                                              )
    #     else:
    #         output, buildings, dict_optimizer = resirf_eoles_coupling_dynamic(buildings, inputs_dynamics,
    #                                                                           policies_heater, policies_insulation,
    #                                                                           list_year, list_trajectory_scc,
    #                                                                           config_coupling["scenario_cost_eoles"],
    #                                                                           config_eoles=config_eoles,
    #                                                                           config_coupling=config_coupling,
    #                                                                           add_CH4_demand=False,
    #                                                                           couplingparam=couplingparam,
    #                                                                           optimparam=optimparam,
    #                                                                           optimization=True,
    #                                                                           energy_taxes=energy_taxes,
    #                                                                           energy_vta=energy_vta,
    #                                                                           optim_eoles=False)
    # elif "greenfield" in config_coupling.keys():  # we optimize in a greenfield manner
    #     assert config_coupling[
    #         "greenfield"], "Parameter greenfield can only be True for the time being, when specified in config."
    #     print("Greenfield")
    #     output, buildings, dict_optimizer = resirf_eoles_coupling_greenfield(buildings, inputs_dynamics,
    #                                                                          policies_heater, policies_insulation,
    #                                                                          scc=775, scenario_cost=config_coupling["scenario_cost_eoles"],
    #                                                                          config_eoles=config_eoles,
    #                                                                          config_coupling=config_coupling,
    #                                                                          add_CH4_demand=False,
    #                                                                          optimization=True,
    #                                                                          optimparam=optimparam
    #                                                                          )
    # else:  # we optimize the value of subsidy
    #     output, buildings, dict_optimizer = resirf_eoles_coupling_dynamic(buildings, inputs_dynamics,
    #                                                                       policies_heater, policies_insulation,
    #                                                                       list_year, list_trajectory_scc, config_coupling["scenario_cost_eoles"],
    #                                                                       config_eoles=config_eoles,
    #                                                                       config_coupling=config_coupling,
    #                                                                       add_CH4_demand=False,
    #                                                                       optimparam=optimparam,
    #                                                                       couplingparam=couplingparam,
    #                                                                       optimization=True,
    #                                                                       energy_taxes=energy_taxes,
    #                                                                       energy_vta=energy_vta)

    # Save results
    export_results, output_resirf = save_simulation_results(output, buildings, name_config_coupling, config_coupling, config_eoles, config_resirf,
                            dict_optimizer, optimization=True)

    return name_config_coupling, output_resirf, export_results


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

        # Plots
        date = datetime.datetime.now().strftime("%m%d_%H%M%S")
        folder = os.path.join(folder_comparison, f'{date}')
        if not os.path.isdir(folder):
            os.mkdir(folder)

        # Plots coupling
        if reference is not None:
            assert reference in results_general.keys(), "Name of reference simulation should be one of the simulations."
            annualized_system_costs_df, total_system_costs_df, consumption_savings_tot_df, complete_system_costs_2050_df = comparison_simulations(
                results_general, ref=reference, greenfield=greenfield, health=health, save_path=folder, carbon_constraint=carbon_constraint)

            results_resirf["Reference"] = results_resirf.pop(reference)
            plot_compare_scenarios(results_resirf, folder=folder)

        # config_policies = get_json('project/input/policies/cba_inputs.json')
        # if 'Reference' in results_resirf.keys() and len(results_resirf.keys()) > 1 and config_policies is not None:
        #     indicator_policies(results_resirf, folder, config_policies, policy_name=None)

    except Exception as e:
        logger.exception(e)
        raise e
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate coupling.')
    parser.add_argument("--cpu", type=int, default=3, help="CPUs for multiprocessing")
    parser.add_argument("--configpath", type=str, help="config")

    args = parser.parse_args()
    cpu = args.cpu  # we select the config we are interested in
    configpath = Path(args.configpath)
    configpath = Path('eoles') / Path('inputs') / Path('xps') / configpath

    assert configpath.is_file(), "configpath argument does not correspond to a real file"
    # assert os.path.isfile(configpath)
    assert (configpath.resolve().parent / Path("base.json")).is_file(), "Directory does not contain the reference configuration file"

    with open(configpath) as file:  # load additional configuration
        config_additional = json.load(file)

    with open(configpath.resolve().parent / Path("base.json")) as file:  # load reference configuration for coupling
        config_coupling = json.load(file)

    list_design = ['uniform', 'centralized_insulation', 'GR', 'GR_fge', 'MWh_insulation']
    list_design = ['uniform']
    DICT_CONFIGS_no_optim_s3 = create_configs_coupling(list_design=list_design, config_coupling=config_coupling,
                                                       config_additional=config_additional)

    # DICT_CONFIGS_no_optim_s3_constant = create_configs_coupling(list_design=list_design, name_design='no_optim_s3_constant', config_coupling=config_coupling, cap_MWh=1000,
    #                                              cap_tCO2=2000, greenfield=False, prices_constant=True, biomass_potential_scenario="S3",
    #                                              aggregated_potential=True, maximum_capacity_scenario='N1', max_iter=100, lifetime_insulation=5,
    #                                                    optim_eoles=False, carbon_emissions_resirf='carbon_emission_s3', electricity_constant=True)
    #
    # DICT_CONFIGS_S3_N1_tight = create_configs_coupling(list_design=list_design, name_design='S3_N1_tight', config_coupling=config_coupling, cap_MWh=1000,
    #                                              cap_tCO2=2000, greenfield=False, prices_constant=True, biomass_potential_scenario="S3",
    #                                              aggregated_potential=True, maximum_capacity_scenario='N1', max_iter=100, lifetime_insulation=5,
    #                                                    carbon_budget='carbon_budget_tight')
    #
    # DICT_CONFIGS_greenfield_S3_N1_dh = create_configs_coupling(list_design=list_design, name_design='greenfield_S3_N1_dh',
    #                                                            config_coupling=config_coupling, cap_MWh=1000,
    #                                              cap_tCO2=2000, greenfield=True, prices_constant=True, biomass_potential_scenario="S3",
    #                                              aggregated_potential=True, maximum_capacity_scenario='N1', max_iter=100, lifetime_insulation=5,
    #                                                    district_heating=True)
    #
    # DICT_CONFIGS_S3_N1_dh = create_configs_coupling(list_design=list_design, name_design='S3_N1_dh', config_coupling=config_coupling, cap_MWh=1000,
    #                                              cap_tCO2=2000, greenfield=False, prices_constant=True, biomass_potential_scenario="S3",
    #                                              aggregated_potential=True, maximum_capacity_scenario='N1', max_iter=100, lifetime_insulation=5,
    #                                                    district_heating=True)
    #
    # DICT_CONFIGS_S3_N1nuc = create_configs_coupling(list_design=list_design,
    #                                              name_design='S3_N1nuc', config_coupling=config_coupling, cap_MWh=1000,
    #                                              cap_tCO2=2000, greenfield=False, prices_constant=True, biomass_potential_scenario="S3",
    #                                              aggregated_potential=True, maximum_capacity_scenario='N1nuc', max_iter=100, lifetime_insulation=5)
    #
    # DICT_CONFIGS_S2p_N1nuc = create_configs_coupling(list_design=list_design,
    #                                              name_design='S2p_N1nuc', config_coupling=config_coupling, cap_MWh=1000,
    #                                              cap_tCO2=2000, greenfield=False, prices_constant=True, biomass_potential_scenario="S2p",
    #                                              aggregated_potential=True, maximum_capacity_scenario='N1nuc', max_iter=100, lifetime_insulation=5)
    #
    # DICT_CONFIGS_greenfield_S2p_N1 = create_configs_coupling(list_design=list_design, name_design='greenfield_S2p_N1', config_coupling=config_coupling, cap_MWh=1000,
    #                                              cap_tCO2=2000, greenfield=True, prices_constant=True, biomass_potential_scenario="S2p",
    #                                              aggregated_potential=True, maximum_capacity_scenario='N1', max_iter=100, lifetime_insulation=5)
    #
    # DICT_CONFIGS_greenfield_S3_Opt = create_configs_coupling(list_design=list_design, name_design='greenfield_S3_Opt', config_coupling=config_coupling, cap_MWh=1000,
    #                                              cap_tCO2=2000, greenfield=True, prices_constant=True, biomass_potential_scenario="S3",
    #                                              aggregated_potential=True, maximum_capacity_scenario='Opt', max_iter=100, lifetime_insulation=5)

    list_design = ['uniform', 'centralized_insulation', 'GR', 'MWh_insulation']

    subsidies_heater = {
        'uniform': [1.0],
        'centralized_insulation': [0.5, 0.16, 0.5, 0.5, 0.28],
        'GR': [0.5, 0.16, 0.5, 0.5, 0.5],
        'MWh_insulation': [0.75],
    }
    subsidies_insulation = {
        'uniform': [0.38],
        'centralized_insulation': [0.16, 0.83, 0.5, 0.16, 0.95],
        'GR': [0.16, 0.83, 0.5, 0.16, 0.16],
        'MWh_insulation': [0.14],
    }
    subsidies_heater, subsidies_insulation = extract_subsidy_value(os.path.join('eoles/outputs', '0511_no_optim_s3'), name_config='greenfield_no_optim_s3')
    config_additional["subsidies_heater"] = subsidies_heater
    config_additional["subsidies_insulation"] = subsidies_insulation
    # list_design = ['centralized_insulation']
    # list_design = ['uniform', 'GR', 'MWh_insulation']
    DICT_CONFIGS_greenfield_S3_N1_no_optim_rerun = create_configs_coupling(list_design=list_design,
                                                                     config_coupling=config_coupling, config_additional=config_additional)

    results = run_multiple_configs(DICT_CONFIGS_no_optim_s3, cpu=cpu, exogenous=False, reference=None, greenfield=True,
                                   health=True, carbon_constraint=True)

    # config_coupling_no_optim['carbon_emissions_resirf'] = "project/input/technical/carbon_emission_s3.csv"
    # DICT_CONFIGS_greenfield_S3_N1_no_optim = create_configs_coupling(list_design=['centralized_insulation'],
    #                                                         name_design='greenfield_no_optim_s3', config_coupling=config_coupling_no_optim, cap_MWh=1000,
    #                                                         cap_tCO2=2000, greenfield=True, prices_constant=True, biomass_potential_scenario="S3",
    #                                                         aggregated_potential=True, maximum_capacity_scenario='N1', max_iter=110, lifetime_insulation=5)
    # results = run_multiple_configs(DICT_CONFIGS_greenfield_S3_N1_no_optim, cpu=cpu, exogenous=False, reference=None, greenfield=True,
    #                                health=True, carbon_constraint=True)

    # results = run_multiple_configs(DICT_CONFIGS_greenfield_S2_N1, cpu=cpu, exogenous=False, reference=None, greenfield=True,
    #                                health=True, carbon_constraint=True)

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

    # dict_output = {"Reference": os.path.join("eoles/outputs/0407_134331_Reference"),
    #                "Ambitious": os.path.join("eoles/outputs/0407_134347_Ambitious"),
    #                }
    # annualized_system_costs_df, total_system_costs_df, consumption_savings_tot_df, complete_system_costs_2050_df = comparison_simulations(
    #     dict_output, ref="Reference", greenfield=False, health=True,
    #     save_path=os.path.join("eoles/outputs/comparison/test"), carbon_constraint=False)
