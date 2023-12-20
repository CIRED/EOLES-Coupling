[![DOI](https://zenodo.org/badge/544807204.svg)](https://zenodo.org/doi/10.5281/zenodo.10409265)

# EOLES-ResIRF Coupling

The EOLES-ResIRF Coupling model is a tool for studying integrated decarbonization pathways for the residential sector. It relies on coupling Res-IRF4, a technically- and behaviorally-rich model of household energy demand in France, and EOLES, a technology-explicit model of the French energy system. The coupling framework can be used in two different ways. First, it can be used to test exogenous policy portfolio assumptions in the residential sector. Second, it can be used to optimize the value of energy efficiency subsidies, in combination with the energy system. 

## Installation

### Step 1: Git clone the folder in your computer.

```bash
git clone https://github.com/celiaescribe/eoles2.git
```

### Step 2: Create a conda environment from the environment.yml file:
- The environment.yml file is in the eoles2 folder.
- Use the terminal and go to the eoles2 folder stored on your computer.
- Type the following command:
```bash
conda env create -f env.yml
```

Note that this step may take a few minutes, as the `pyomo` package takes time to install.

### Step 3: Activate the conda environment:

```bash
conda activate eoles
```
## Getting started

### Step 1: Installing ResIRF
The coupling requires to have installed the package Res-IRF as well in your environment. To do so, first refer to the [Res-IRF project](https://github.com/CIRED/Res-IRF) to clone the project in your own computer. Then, go to the corresponding folder, and run the following command (make sure that the eoles environment is activated)
```bash
pip install -e .
```

This allows you to have access to all the functionalities from ResIRF in your environment.

### Step 2: Install a solver
Our package relies on the Gurobi solver, used through the `pyomo` interface. This requires dowloading Gurobi and installing a license. It is also possible to rely on open-source solvers such as CPLEX. 

### Step 3: Running the coupling
The standard way to run the coupling is to launch the script main_coupling_resirf.py. This requires providing the configurations which you want to run. Examples of configurations can be found in `eoles/inputs/xps`. Files called `base.json` provide general configuration parameters. Other json files in a given folder provide specifications for varying configurations. For example, the file `eoles/inputs/xps/20231205/biogasS2_capacityN1_demandReference_policyambition.json` specifies different parameters for biogas potential, renewable potential, demand scenario and residential policies scenario.

You can run the script as follows:

```bash
python main_coupling_resirf.py --cpu 1 --configdir "eoles/inputs/xps/20231205
```

It is possible to create a folder with different configurations using the `scenarios_creation.ipynb` notebook.

The command specifies the configuration to use. There are two ways to specify that:
- `--configdir "eoles/inputs/xps/20231205`: the folder where the configurations are stored
- `--configfile`: the name of the configuration file to run if you do not want to run all configurations in a given folder (ex: `eoles/inputs/xps/20231205/biogasS3_capacityN1_demandReference_policyreference.json`)

Note that other parameters are allowed:
- `--cpu 1`: the number of CPUs to use
- `--patterns`: specify patterns to select configurations (default: `"*.json"`)
- `--exclude-patterns`: specify patterns to exclude configurations (default: `"base.json"`)

### Step 3: explore outputs
Output files are stored in `eoles/outputs`. 

## Results
Lucas Vivier, & Célia Escribe. (2023). Result from How to allocate mitigation efforts between home insulation, fuel switch and fuel decarbonization? Insights from the French residential sector. [Data set]. Zenodo. [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10409404.svg)](https://doi.org/10.5281/zenodo.10409404)

## Contributing

The development of the EOLES model was initiated by Behrang Shirizadeh and Philippe Quirion. The development of the Res-IRF package was originated by Louis-Gaëtan Giraudet. We rely on the latest version of Res-IRF, developed by Lucas Vivier. The coupling of the two models was developed by Célia Escribe and Lucas Vivier.