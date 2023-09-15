#!/bin/bash

# List of config paths you want to iterate over
config_paths=("230908/optim_noHC/S3_N1.json" "another_path.json" "yet_another_path.json")

config_dir="eoles/inputs/xps/230915/policy_resirf_scc"

# Iterate over each JSON file in the directory
for config_path in "$config_dir"/*.json; do
  if [ -f "$config_path" ] && [ "$(basename "$config_path")" != "base.json" ]; then
    echo $config_path
    python main_coupling_resirf.py --cpu 1 --configpath "$config_path"
  fi
done

## Iterate over each config path
#for config_path in "${config_paths[@]}"; do
#  python main_coupling_resirf.py --cpu 1 --configpath "$config_path"
#done