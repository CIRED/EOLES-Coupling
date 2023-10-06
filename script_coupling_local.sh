#!/bin/bash

# List of config paths you want to iterate over
config_dirs=("eoles/inputs/xps/231003/policies_exogenous_cc_pricefeedback" "eoles/inputs/xps/231003/policies_exogenous_scc_pricefeedback")

# Iterate over each config directory
for config_dir in "${config_dirs[@]}"; do
  # Iterate over each JSON file in the directory
  for config_path in "$config_dir"/*.json; do
    if [ -f "$config_path" ] && [ "$(basename "$config_path")" != "base.json" ]; then
      echo "$config_path"
      python main_coupling_resirf.py --cpu 1 --configpath "$config_path"
    fi
  done
done

## Iterate over each config path
#for config_path in "${config_paths[@]}"; do
#  python main_coupling_resirf.py --cpu 1 --configpath "$config_path"
#done