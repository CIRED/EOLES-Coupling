import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta
from shutil import copy2
from itertools import product
import json
from copy import deepcopy
import glob
import random
import argparse

from settings_scenarios import map_values, map_scenarios_to_configs


def creation_scenarios(file=Path('eoles/inputs/config/scenarios/scenarios.json'), N=100, method='marginal', n_cluster=None):
    assert method in ['marginal', 'montecarlo', 'exhaustive', 'debug']
    prefix = 'marginal'
    if method == 'montecarlo':
        prefix = 'montecarlo_{}'.format(N)
    elif method == 'exhaustive':
        prefix = 'exhaustive'
    elif method == 'debug':
        prefix = 'debug'

    folder_simu = Path('eoles') / Path('inputs') / Path('xps')

    if not folder_simu.is_dir():
        folder_simu.mkdir()

    time = datetime.today().strftime('%Y%m%d_%H%M%S')
    folder_simu = folder_simu / Path('{}_{}'.format(prefix, time))

    folder_simu.mkdir(parents=True, exist_ok=True)
    [os.remove(f) for f in glob.glob(os.path.join(folder_simu, '*'))]

    path_file_settings = Path('eoles') / Path('inputs') / Path('config') / Path('settings_framework.json')
    copy2(path_file_settings, folder_simu)

    with open(file, 'r') as file:
        scenarios = json.load(file)

    scenarios = {**scenarios['demand'], **scenarios['supply'], **scenarios['prices']}

    if method == 'marginal':
        temp, k = {}, 1
        temp.update({'S0': {'biogas': 'reference'}})
        for key, value in scenarios.items():
            for v in value:
                if v != 'reference':
                    temp['S{}'.format(k)] = {key: v}
                    k += 1
        scenarios_counterfactual = deepcopy(temp)
        scenarios_ban = deepcopy(temp)
        for k, v in temp.items():
            scenarios_ban['{}-ban'.format(k)] = {**v, 'ban': 'Ban'}

        scenarios = {**scenarios_counterfactual, **scenarios_ban}
    elif method == 'debug':
        temp, k = {}, 1
        for key, value in scenarios.items():
            for v in value:
                if v != 'reference':
                    temp['S{}'.format(k)] = {key: v}
                    k += 1
        scenarios = deepcopy(temp)
    else:
        name_scenarios, values_scenarios = zip(*scenarios.items())
        scenarios = [dict(zip(name_scenarios, v)) for v in product(*values_scenarios)]
        scenarios = {'S{}'.format(n): v for n, v in enumerate(scenarios)}

        scenarios_counterfactual = deepcopy(scenarios)
        for k, v in scenarios_counterfactual.items():
            v.update({'ban': 'reference'})

        scenarios_ban = deepcopy(scenarios)

        for k, v in scenarios_ban.items():
            v.update({'ban': 'Ban'})

        if method == 'montecarlo':
            assert N is not None, 'N should be provided with montecarlo method'
            # Randomly select N keys (knowing that 2 * N scenarios will be run)
            selected_keys = random.sample(list(scenarios_counterfactual), N)

            # If you need the key-value pairs
            scenarios_counterfactual = {key: scenarios_counterfactual[key] for key in selected_keys}
            scenarios_ban = {'{}-ban'.format(key): scenarios_ban[key] for key in selected_keys}
            scenarios = {**scenarios_counterfactual, **scenarios_ban}
        else:  # method is 'exhaustive
            scenarios_ban = {'{}-ban'.format(key): scenarios_ban[key] for key in scenarios_ban.keys()}
            scenarios = {**scenarios_counterfactual, **scenarios_ban}

        if n_cluster is not None:
            n = len(scenarios)
            scenarios_per_cluster = n // n_cluster

            # Additional scenarios to distribute if n is not divisible by n_cluster
            additional_scenarios = n % n_cluster

            # Assign scenarios to clusters
            cluster_assignments = {}
            start = 0
            for i in range(n_cluster):
                # Calculate end index for slicing; distribute additional scenarios among the first few clusters
                end = start + scenarios_per_cluster + (1 if i < additional_scenarios else 0)
                for key in list(scenarios.keys())[start:end]:
                    cluster_assignments[key] = f'Cluster{i + 1}'
                start = end

            # Create DataFrame from cluster assignments
            cluster_assignments_df = pd.DataFrame(list(cluster_assignments.items()), columns=['Scenario', 'Cluster'])
            cluster_assignments_df.set_index('Scenario', inplace=True)
            assert set(cluster_assignments_df.index) == set(scenarios.keys()), "Problem when assigning scenarios to clusters"
            cluster_assignments_df.to_csv(folder_simu / Path('cluster_assignments.csv'))


    path_file_config_reference = Path('eoles') / Path('inputs') / Path('config') / Path('config_coupling_reference.json')
    with open(path_file_config_reference, 'r') as file:
        config_reference = json.load(file)

    for name_scenario, values_scenarios in scenarios.items():
        new_config = deepcopy(config_reference)
        new_config['name_config'] = name_scenario
        for name_variable, value_variable in values_scenarios.items():
            if value_variable == 'reference':  # no modification to the configuration
                pass
            else:
                if map_scenarios_to_configs[name_variable][0] == 'supply':
                    new_config[map_scenarios_to_configs[name_variable][1]] = deepcopy(map_values[value_variable])
                elif map_scenarios_to_configs[name_variable][0] == 'prices':
                    if map_scenarios_to_configs[name_variable][1] in new_config.keys():
                        new_config[map_scenarios_to_configs[name_variable][1]]['eoles']['rate'].update(map_values[value_variable]['eoles']['rate'])  # we just add a prices argument to the dictionary
                        new_config[map_scenarios_to_configs[name_variable][1]]['resirf']['rate'].update(map_values[value_variable]['resirf']['rate'])  # we just add a prices argument to the dictionary
                    else:
                        new_config[map_scenarios_to_configs[name_variable][1]] = deepcopy(map_values[value_variable])  # we just add a prices argument to the dictionary
                elif map_scenarios_to_configs[name_variable][0] == 'demand':
                    if map_scenarios_to_configs[name_variable][1] in new_config.keys():
                        new_config[map_scenarios_to_configs[name_variable][1]].update(deepcopy(map_values[value_variable]))
                    else:
                        new_config[map_scenarios_to_configs[name_variable][1]] = deepcopy(map_values[value_variable])
                elif map_scenarios_to_configs[name_variable][0] == 'policies':
                    temp = deepcopy(new_config['policies'])
                    temp.update(deepcopy(map_values[value_variable]))  # we add new policy information to the existing one
                    new_config['policies'] = deepcopy(temp)
                else:
                    raise KeyError('Key not found')
        folder_additional = folder_simu / Path(name_scenario + '.json')
        with open(folder_additional, "w") as outfile:
            outfile.write(json.dumps(new_config, indent=4))

    scenarios = pd.DataFrame.from_dict(scenarios, orient='index')
    if scenarios.duplicated().any():
        raise ValueError('Duplicated scenarios')
    scenarios.to_csv(folder_simu / Path('scenarios.csv'))

    return folder_simu


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create scenarios.')
    parser.add_argument("--N", type=int, default=100, help="Number of scenarios if created here.")
    parser.add_argument("--method", type=str, default='marginal', help="Whether to use marginal, MonteCarlo or exhaustive for the creation of scenarios.")
    parser.add_argument("--ncluster", type=int, default=None, help="Number of clusters to process the Montecarlo")
    args = parser.parse_args()

    N = int(args.N)
    method = str(args.method)
    n_cluster = None
    if args.ncluster is not None:
        n_cluster = int(args.ncluster)
    folder_simu = creation_scenarios(method=method, N=N, n_cluster=n_cluster)