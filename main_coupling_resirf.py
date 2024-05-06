import json
import os
from importlib import resources

import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from pickle import dump, load
from multiprocessing import Pool
import glob
import shutil

from project.coupling import ini_res_irf, simu_res_irf
from project.utils import get_json
from project.write_output import plot_scenario, plot_compare_scenarios, indicator_policies
from project.model import get_inputs

from project.model import create_logger, get_config, get_inputs
from eoles.model_resirf_coupling import ModelEOLES
from eoles.utils import get_config, get_pandas, calculate_annuities_resirf, modif_config_resirf, \
    modif_config_eoles, modif_config_coupling, create_configs_coupling, create_default_options, create_optimization_param, \
    create_coupling_param, extract_subsidy_value, find_folders
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
    "classic_simple": "eoles/inputs/config/config_resirf_simple.json",
}

DICT_CONFIG_EOLES = {
    "eoles_classic": "eoles_coupling"
}


def save_simulation_results(output, buildings, name_config_coupling, config_coupling, config_eoles, config_resirf,
                            dict_optimizer, optimization=True, save_folder=None, plot=False):
    """Save simulation results."""
    date = datetime.datetime.now().strftime("%m%d%H%M%S")

    if save_folder is not None:
        export_results = Path(save_folder) / Path(f'{date}_{name_config_coupling}')
    else:
        export_results = Path('eoles') / Path('outputs') / Path(f'{date}_{name_config_coupling}')
    # export_results = os.path.join("eoles", "outputs", f'{date}_{name_config_coupling}')

    # Create directories
    if not export_results.is_dir():
        os.mkdir(export_results)

    if not (Path(export_results) / Path("config")).is_dir():
        os.mkdir(Path(export_results) / Path("config"))

    if not (Path(export_results) / Path("dataframes")).is_dir():
        os.mkdir(Path(export_results) / Path("dataframes"))

    if not (Path(export_results) / Path("plots")).is_dir():
        os.mkdir(Path(export_results) / Path("plots"))

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

        if plot:
            if optimization:
                plot_blackbox_optimization(dict_optimizer, save_path=os.path.join(export_results))

            buildings.path = os.path.join(export_results, "plots")
            try:
                plot_scenario(output["Output global ResIRF ()"], output["Stock global ResIRF ()"],
                              buildings)  # make ResIRF plots
            except:
                logger.info(f'Config {name_config_coupling} plots not working')
                pass

            if not config_coupling["greenfield"]:  # si greenfield, on ne veut pas plotter l'évolution des quantités, car pas d'optimisation dynamique
                try:
                    plot_simulation(output, save_path=os.path.join(export_results, "plots"))
                except:
                    pass
            # save_summary_pdf(path=export_results)  # saving summary as pdf
    return export_results, output["Output global ResIRF ()"]


def run_scenario(config_coupling, name_config_coupling="default", save_folder=None, plot=True):
    """
    Runs an optimization scenario.
    :param config_coupling: dict
        Dictionary containing different configurations. For the time being, includes:
        configuration for resirf (market failures, threshold), consideration of H2 CCGT in EOLES, and number of iterations for optimization
    :param name_config_coupling: str
        Name of the configuration used to save the results.
    :return:
    Saves the output of the optimization + save plots
    """
    print(name_config_coupling)

    # We update the different configuration dictionaries
    config_resirf_path = DICT_CONFIG_RESIRF["classic_simple"]  # classical ResIRF configuration
    with open(config_resirf_path) as file:  # load config_resirf
        config_resirf = json.load(file).get('Reference')
    config_resirf = modif_config_resirf(config_resirf, config_coupling)  # modif of this configuration file to take into account options specified in coupling configuration

    config_eoles = eoles.utils.get_config(spec="eoles_coupling")
    config_eoles, config_coupling = modif_config_eoles(config_eoles, config_coupling)  # modif of this configuration file to take into account options specified in coupling configuration

    # update of default options based on the config_coupling dictionary.
    default_config = create_default_options(config_coupling)
    optimparam = create_optimization_param(default_config)
    couplingparam = create_coupling_param(default_config)

    # initialization ResIRF
    buildings, inputs_dynamics, policies_heater, policies_insulation = ini_res_irf(
        path=os.path.join('eoles', 'outputs', 'ResIRF'),
        config=config_resirf,
        level_logger=logging.NOTSET
        )

    energy_taxes, energy_vat = get_energy_prices_and_taxes(config_resirf)
    calibration_elec_lcoe, calibration_elec_transport_distrib, calibration_gas, m_eoles = calibration_price(
        config_eoles, scc=100)
    config_coupling["calibration_elec_lcoe"] = calibration_elec_lcoe
    config_coupling["calibration_elec_transport_distrib"] = calibration_elec_transport_distrib
    config_coupling["calibration_naturalgas_lcoe"] = calibration_gas
    config_coupling["calibration_biogas_lcoe"] = 1.2
    # print(config_coupling["calibration_elec_lcoe"], config_coupling["calibration_elec_transport_distrib"], config_coupling["calibration_naturalgas_lcoe"])

    # Specification of options for optimization
    list_sub_heater, list_sub_insulation = None, None
    optimization = True

    if config_coupling["no_subsidies"]:  # scenario particulier où on fixe les subventions à zéro (quand on fait tourner un scénario Res-IRF spécifique)
        print("no optimized subsidies")
        optimization = False
        if config_coupling["greenfield"]:
            list_sub_heater, list_sub_insulation = [0.0], [0.0]
        else:
            list_sub_heater, list_sub_insulation = [0.0 for i in range(len(config_coupling["list_year"]))], [0.0 for i in range(len(config_coupling["list_year"]))]

    if config_coupling['subsidies_specified']:  # on donne des subventions spécifiées
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
                                                                             couplingparam=couplingparam
                                                                             )
    else:  # we optimize the value of subsidy
        output, buildings, dict_optimizer = resirf_eoles_coupling_dynamic(buildings, inputs_dynamics,
                                                                          policies_heater, policies_insulation,
                                                                          config_coupling["scenario_cost_eoles"],
                                                                          config_eoles=config_eoles,
                                                                          config_coupling=config_coupling,
                                                                          add_CH4_demand=False,
                                                                          optimparam=optimparam,
                                                                          couplingparam=couplingparam,
                                                                          optimization=optimization,
                                                                          list_sub_heater=list_sub_heater,
                                                                          list_sub_insulation=list_sub_insulation,
                                                                          energy_taxes=energy_taxes,
                                                                          energy_vat=energy_vat)

    # Save results
    export_results, output_resirf = save_simulation_results(output, buildings, name_config_coupling, config_coupling, config_eoles, config_resirf,
                            dict_optimizer, optimization=True, save_folder=save_folder, plot=plot)

    return name_config_coupling, output_resirf, export_results


def run_multiple_configs(dict_config, cpu: int, folder_to_save=None, plot=True):
    """Run multiple configs in parallel"""
    logger.info('Scenarios: {}'.format(', '.join(dict_config.keys())))

    if not (Path('eoles') / Path('outputs')).is_dir():  # creation of outputs folder if nonexisting
        os.mkdir(Path('eoles') / Path('outputs'))

    if folder_to_save is not None: # we create the folder to save the results
        folder_to_save = Path('eoles') / Path('outputs') / Path(folder_to_save)
        if not folder_to_save.is_dir():
            folder_to_save.mkdir()

        try:  # saving the scenarios.csv file in the folder to save
            shutil.copy2(Path('eoles/inputs/xps') / Path(folder_to_save.name) / Path('scenarios.csv'), Path('eoles') / Path('outputs') / Path(folder_to_save.name) / Path('scenarios.csv'))
        except:
            pass
    try:
        logger.info('Launching processes')
        with Pool(cpu) as pool:

            results = pool.starmap(run_scenario,
                                   zip(dict_config.values(), [n for n in dict_config.keys()], [folder_to_save] * len(dict_config), [plot] * len(dict_config)))
        results_resirf = {i[0]: i[1] for i in results}
        results_general = {i[0]: i[2] for i in results}

        # Compress the folder
        shutil.make_archive(folder_to_save, 'zip', root_dir=folder_to_save.parent, base_dir=folder_to_save.name)
        # Now, remove the original 'folder_to_save' directory to save space
        # shutil.rmtree(folder_to_save)

    except Exception as e:
        logger.exception(e)
        raise e
    return folder_to_save


if __name__ == '__main__':
    # Main code
    parser = argparse.ArgumentParser(description='Simulate coupling.')
    parser.add_argument("--cpu", type=int, default=3, help="CPUs for multiprocessing")
    parser.add_argument("--configpath", type=str, help="config json file", default=None)
    parser.add_argument("--configdir", type=str, help="config directory", default=None)
    parser.add_argument("--configref", type=str, help="base.json", default=None)
    parser.add_argument("--patterns", nargs="+", type=str, default=["*.json"], help="Patterns to filter files in the directory.")
    parser.add_argument("--exclude-patterns", nargs="+", type=str, default=["base.json", "settings_framework.json"],help="Patterns to exclude files.")
    parser.add_argument("--cluster", type=int, default=None, help="Cluster assignment to process Monte-Carlo simulations.")

    args = parser.parse_args()
    cpu = args.cpu  # we select the config we are interested in
    assert (args.configpath is not None) or (args.configdir is not None), "Parameters are not correctly specified"

    configref = Path(args.configref)
    # assert configref.is_file(), "configref argument does not correspond to an existing file, reference configuration is not specified correctly."

    if args.configpath is not None:  # we have specified a json file
        configpath = Path(args.configpath)
        assert configpath.is_file(), "configpath argument does not correspond to an existing file"
        # assert os.path.isfile(configpath)
        assert (configpath.resolve().parent / configref).is_file(), "Directory does not contain the reference configuration file"

        with open(configpath) as file:  # load additional configuration
            config_additional = json.load(file)

        with open(configpath.resolve().parent / configref) as file:  # load reference configuration for coupling
            config_coupling = json.load(file)

        list_design = ['uniform', 'centralized_insulation', 'DR', 'proportional']
        list_design = None

        # # Cas spécifique où on vient extraire la valeur de subventions qui ont été optimisées au préalable
        # config_coupling['subsidies_specified'] = True  # we specify that subsidies are given
        # subsidies_heater, subsidies_insulation = extract_subsidy_value(Path('eoles') / Path('outputs') / Path('0910_S3_N1'), name_config='S3_N1')
        # config_additional["subsidies_heater"] = subsidies_heater
        # config_additional["subsidies_insulation"] = subsidies_insulation

        DICT_CONFIGS = create_configs_coupling(list_design=list_design, config_coupling=config_coupling,
                                               config_additional=config_additional)

    if args.configdir is not None:  # we have specified a directory which contains multiple json files
        configdir = Path(args.configdir)
        assert configdir.is_dir(), "configdir argument does not correspond to an existing directory."

        if args.cluster is None:
            patterns = args.patterns
        else:  # we have a specific assignment of scenarios to clusters
            cluster_assignments_df = pd.read_csv(configdir / Path('cluster_assignments.csv'), index_col=0)
            patterns = cluster_assignments_df.loc[cluster_assignments_df.Cluster == f'Cluster{int(args.cluster)}'].index.to_list()
            patterns = [f'{pattern}.json' for pattern in patterns]

        config_files = []
        for pattern in patterns:
            pattern_path = configdir / pattern
            matching_files = glob.glob(str(pattern_path))

            # Loop through the matching files and exclude those that match any exclude pattern, notably the reference config file
            for file in matching_files:
                if all(file_match not in file for file_match in args.exclude_patterns):
                    config_files.append(file)

        DICT_CONFIGS = {}
        for configpath in config_files:
            configpath = Path(configpath)
            with open(configpath) as file:  # load additional configuration
                config_additional = json.load(file)

            with open(configpath.resolve().parent / configref) as file:  # load reference configuration for coupling
                config_coupling = json.load(file)

            list_design = ['uniform', 'centralized_insulation', 'DR', 'proportional']
            list_design = None

            DICT_CONFIGS = create_configs_coupling(list_design=list_design, config_coupling=config_coupling,
                                                   config_additional=config_additional, dict_configs=DICT_CONFIGS)

    # capture the name of the folder to save from configdir
    if configdir is not None:
        folder_date = configdir.name
    else:
        folder_date = datetime.datetime.now().strftime("%Y%m%d")
    folder_to_save = run_multiple_configs(DICT_CONFIGS, cpu=cpu, folder_to_save=folder_date, plot=False)

    # CODE to test specific subsidies
    # to add if I want to run stuff again with specific subsidies. Maybe to adapt depending on what i want to test
    # list_folder = find_folders(base_folder="eoles/outputs/1110_optim_pricefeedback", target_string="centralized_insulation_S2_N1_pricefeedback_hcDPE")
    # subsidies_heater_dict, subsidies_insulation_dict = extract_subsidy_value(list_folder,
    #                                                                          name_config="S2_N1_pricefeedback_hcDPE")
    # config_coupling['subsidies_specified'] = True  # we specify that subsidies are given
    # config_additional['subsidies_heater'] = subsidies_heater_dict
    # config_additional['subsidies_insulation'] = subsidies_insulation_dict

    # config_coupling['subsidies_specified'] = True  # we specify that subsidies are given
    # config_additional['subsidies_heater'] = {'centralized_insulation': [0.5, 0.2, 0.6, 0.8, 0.6],
    #                                          'uniform': [0.9, 0.9, 0.9, 0.9, 0.9]}
    # config_additional['subsidies_insulation'] = {'centralized_insulation': [0.97, 0.8, 0.2, 0.2, 0.97],
    #                                              'uniform': [0.5, 0.8, 0.2, 0.5, 0.6]}

    # configpath = Path('eoles') / Path('inputs') / Path('xps') / configpath





