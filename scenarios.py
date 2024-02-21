import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta
from shutil import copy2
from itertools import product
import json
from copy import deepcopy
import glob

from settings_scenarios import map_values, map_scenarios_to_configs


folder_simu = Path('eoles') / Path('inputs') / Path('xps')

date = datetime.now()
date = date.strftime("%Y%m%d")  # formatting
folder_simu = folder_simu / Path(str(date))

folder_simu.mkdir(parents=True, exist_ok=True)
[os.remove(f) for f in glob.glob(os.path.join(folder_simu, '*'))]

path_file_settings = Path('eoles') / Path('inputs') / Path('config') / Path('settings_framework.json')
copy2(path_file_settings, folder_simu)


scenarios_supply = {
    'biogas': ['reference', 'Biogas-'],
    'capacity_nuc': ['reference', 'Nuc-'],
    'capacity_ren': ['reference', 'Ren-'],
    'demand': ['reference', 'Sufficiency', 'Reindustrialisation']
}
scenarios_demand = {
    'ban': ['reference', 'Ban'],
    'insulation': ['reference', 'NoPolicy'],
    'learning': ['reference', 'Learning+', 'Learning-'],
    # 'profile': ['reference', 'ProfileFlat'],
    'elasticity': ['reference', 'Elasticity+', 'Elasticity-']
}
scenarios = {**scenarios_supply, **scenarios_demand}

name_scenarios, values_scenarios = zip(*scenarios.items())
scenarios = [dict(zip(name_scenarios, v)) for v in product(*values_scenarios)]
scenarios = {'S{}'.format(n): v for n, v in enumerate(scenarios)}

path_file_config_reference = Path('eoles') / Path('inputs') / Path('config') / Path('config_coupling_reference.json')
with open(path_file_config_reference, 'r') as file:
    config_reference = json.load(file)

for name_scenario, values_scenarios in scenarios.items():  #key: 'S0', values= 'biogas'
    new_config = config_reference.copy()
    new_config['name_config'] = name_scenario
    for name_variable, value_variable in values_scenarios.items():  # i: 'biogas', v: 'Biogas-'
        if value_variable == 'reference':  # no modification to the configuration
            pass
        else:
            if map_scenarios_to_configs[name_variable][0] == 'supply':
                new_config[map_scenarios_to_configs[name_variable][1]] = deepcopy(map_values[value_variable])
                # print(new_config)
            elif map_scenarios_to_configs[name_variable][0] == 'demand':
                if map_scenarios_to_configs[name_variable][1] in new_config.keys():
                    new_config[map_scenarios_to_configs[name_variable][1]].update(deepcopy(map_values[value_variable]))
                else:
                    new_config[map_scenarios_to_configs[name_variable][1]] = deepcopy(map_values[value_variable])
            elif map_scenarios_to_configs[name_variable][0] == 'policies':
                temp = deepcopy(new_config['policies'])
                temp.update(deepcopy(map_values[value_variable]))
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