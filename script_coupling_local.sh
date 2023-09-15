#!/bin/bash

# List of config paths you want to iterate over

config_dir="eoles/inputs/xps/230915/policy_resirf_cc"

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