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

from settings_scenarios import map_values, map_scenarios_to_configs


def creation_scenarios(file=Path('eoles/inputs/config/scenarios/scenarios.json'), N=100, montecarlo=False):

    prefix = 'marginal'
    if montecarlo:
        prefix = 'montecarlo_{}'.format(N)

    folder_simu = Path('eoles') / Path('inputs') / Path('xps')

    time = datetime.today().strftime('%Y%m%d_%H%M%S')
    folder_simu = folder_simu / Path('{}_{}'.format(prefix, time))

    folder_simu.mkdir(parents=True, exist_ok=True)
    [os.remove(f) for f in glob.glob(os.path.join(folder_simu, '*'))]

    path_file_settings = Path('eoles') / Path('inputs') / Path('config') / Path('settings_framework.json')
    copy2(path_file_settings, folder_simu)

    with open(file, 'r') as file:
        scenarios = json.load(file)

    scenarios = {**scenarios['demand'], **scenarios['supply'], **scenarios['prices']}

    if montecarlo:
        name_scenarios, values_scenarios = zip(*scenarios.items())
        scenarios = [dict(zip(name_scenarios, v)) for v in product(*values_scenarios)]
        scenarios = {'S{}'.format(n): v for n, v in enumerate(scenarios)}
        if N is not None:
            scenarios_counterfactual = deepcopy(scenarios)
            for k, v in scenarios_counterfactual.items():
                v.update({'ban': 'reference'})

            scenarios_ban = deepcopy(scenarios)

            for k, v in scenarios_ban.items():
                v.update({'ban': 'Ban'})

            # Randomly select N keys (knowing that 2 * N scenarios will be run)
            selected_keys = random.sample(list(scenarios_counterfactual), N)

            # If you need the key-value pairs
            scenarios_counterfactual = {key: scenarios_counterfactual[key] for key in selected_keys}
            scenarios_ban = {'{}-ban'.format(key): scenarios_ban[key] for key in selected_keys}
            scenarios = {**scenarios_counterfactual, **scenarios_ban}
    else:
        temp, k = {}, 1
        temp.update({'S0': {'insulation': 'reference'}})
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
                    new_config[map_scenarios_to_configs[name_variable][1]] = deepcopy(map_values[value_variable])  # we just add a prices argument to the dictionary
                elif map_scenarios_to_configs[name_variable][0] == 'demand':
                    # if map_scenarios_to_configs[name_variable][1] == 'energy':  # we have to modify prices, which requires a specific handling of this case
                    #     assert 'energy' in new_config.keys(), 'Energy should be a key of the configuration'
                    #     new_config['energy']['energy_prices']['rate'].update(deepcopy(map_values[value_variable]['resirf']['rate']))  # we only modify the rate for the given scenario
                    # else:
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
    folder_simu = creation_scenarios()