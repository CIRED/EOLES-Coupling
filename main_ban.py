# This script is used to assess the impact of a ban on gas boiler script with the correct arguments.

import json
from scenarios import creation_scenarios
import subprocess
import argparse
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate coupling.')
    parser.add_argument("--cpu", type=int, default=3, help="CPUs for multiprocessing")
    parser.add_argument("--N", type=int, default=100, help="Number of scenarios if created here.")
    parser.add_argument("--montecarlo", type=bool, default=False, help="Whether to use MonteCarlo for the creation of scenarios.")
    parser.add_argument("--scenarios", type=str, default=None, help="If provided, specifies json file to use to create scenarios.")
    parser.add_argument("--folder", type=str, default=None, help="If folder is provided, scenarios are not created.")
    args = parser.parse_args()
    cpu = int(args.cpu)
    montecarlo = bool(args.montecarlo)
    N = int(args.N)

    if args.folder is not None:
        folder_simu = str(args.folder)
    else:
        if args.scenarios is not None:
            scenarios = Path(args.scenarios)
            folder_simu = creation_scenarios(file=scenarios, N=N, montecarlo=montecarlo)
        else:
            folder_simu = creation_scenarios(N=N, montecarlo=montecarlo)

    # run main_coupling_resirf.py
    command = f'python main_coupling_resirf.py --configdir {folder_simu} --cpu {cpu} --configref settings_framework.json'
    subprocess.run(command.split())
