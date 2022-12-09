import logging
import pandas as pd

from project.model import social_planner

from eoles.model_heat_coupling import ModelEOLES
from eoles.utils import get_config, heating_hourly_profile, process_heating_need, get_pandas, \
    create_hourly_residential_demand_profile
from eoles.process_cost_efficiency import piecewise_linearization_cost_efficiency


if __name__ == '__main__':

    config = get_config(spec="greenfield")

    LOG_FORMATTER = '%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # consoler handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
    logger.addHandler(console_handler)

    hourly_profile_test = heating_hourly_profile(method="valentin", percentage=0)

    # TODO: attention au choix de hourly profile !!
    # ['Occupancy status', 'Income owner', 'Income tenant', 'Housing type', 'Heating system']
    dict_cost, dict_heat = social_planner(aggregation_archetype=['Wall class', "Housing type"], climate=2006,
                                          smooth=False, building_stock="medium_3", hourly_profile=hourly_profile_test)
    dict_heat = process_heating_need(dict_heat, climate=2006)
    dict_heat_modif = {key: dict_heat[key] * 1.4 for key in
                       dict_heat.keys()}  # TODO: permet de voir l'impact d'augmenter le besoin en chaleur
    linearized_renovation_costs, threshold_linearized_renovation_costs = piecewise_linearization_cost_efficiency(
        dict_cost, number_of_segments=3, plot=True, plot_tot=True)

    threshold_linearized_renovation_costs[
        threshold_linearized_renovation_costs < 0] = 0  # handles cases where the linearization was not perfect

    # importing evolution of historical capacity
    existing_capacity_historical = get_pandas("eoles/inputs/existing_capa_historical.csv",
                                              lambda x: pd.read_csv(x, index_col=0))  # GW
    existing_charging_capacity_historical = get_pandas("eoles/inputs/existing_charging_capacity_historical.csv",
                                              lambda x: pd.read_csv(x, index_col=0))  # GW
    existing_energy_capacity_historical = get_pandas("eoles/inputs/existing_energy_capacity_historical.csv",
                                              lambda x: pd.read_csv(x, index_col=0))  # GW
    maximum_capacity_evolution = get_pandas("eoles/inputs/maximum_capacity_evolution.csv",
                                              lambda x: pd.read_csv(x, index_col=0))  # GW

    # importing evolution of tertiary and ECS gas demand
    heating_gas_demand_RTE_timesteps = get_pandas("eoles/inputs/heating_gas_demand_tertiary_timesteps.csv",
                                            lambda x: pd.read_csv(x, index_col=0).squeeze())
    ECS_gas_demand_RTE_timesteps = get_pandas("eoles/inputs/ECS_gas_demand_timesteps.csv",
                                                          lambda x: pd.read_csv(x, index_col=0).squeeze())

    # TODO: il faudra ajouter les capacités pour charging et discharging
    year = [2025, 2030, 2035, 2040, 2045, 2050]
    trajectory_scc = [180, 250, 350, 500, 650, 776]  # from Quinet

    # Saving all new installed capacity
    new_capacity_tot = pd.Series(0, index=existing_capacity_historical.index, dtype=float)
    new_charging_capacity_tot = pd.Series(0, index=existing_charging_capacity_historical.index, dtype=float)
    new_energy_capacity_tot = pd.Series(0, index=existing_energy_capacity_historical.index, dtype=float)

    # Initialize renovation rate to 0 at first
    existing_renovation_rate = pd.Series(0, index=linearized_renovation_costs.index, dtype=float)  # no renovation has been done

    capacity_df = pd.DataFrame(index=existing_capacity_historical.index, dtype=float)
    new_capacity_df = pd.DataFrame(index=existing_capacity_historical.index, dtype=float)
    charging_capacity_df = pd.DataFrame(index=existing_charging_capacity_historical.index, dtype=float)
    energy_capacity_df = pd.DataFrame(index=existing_energy_capacity_historical.index, dtype=float)
    renovation_rates_df = pd.DataFrame(index=list(dict_heat.keys()), dtype=float)  # renovation rates per archetype

    for y, scc in zip(year, trajectory_scc):
        existing_capa_historical_y = existing_capacity_historical[[str(y)]].squeeze()  # get historical capacity still installed for year of interest
        existing_charging_capacity_historical_y = existing_charging_capacity_historical[[str(y)]].squeeze()
        existing_energy_capacity_historical_y = existing_energy_capacity_historical[[str(y)]].squeeze()

        new_maximum_capacity_y = maximum_capacity_evolution[[str(y)]].squeeze()  # get maximum new capacity to be built

        # Existing capacities at year y
        existing_capacity = existing_capa_historical_y + new_capacity_tot  # existing capacity are equal to newly built
        # capacities over the whole time horizon + existing capacity (from before 2020)
        existing_charging_capacity = existing_charging_capacity_historical_y + new_charging_capacity_tot
        existing_energy_capacity = existing_energy_capacity_historical_y + new_energy_capacity_tot

        maximum_capacity = (existing_capacity + new_maximum_capacity_y).dropna()  # we drop nan values, which correspond to technologies without any upper bound

        # creating gas profile
        heating_gas_demand = heating_gas_demand_RTE_timesteps[year] * 1e3  # in TWh
        ECS_gas_demand = ECS_gas_demand_RTE_timesteps[year] * 1e3  # in TWh
        ECS_demand_hourly = ECS_gas_demand/8760
        hourly_gas = create_hourly_residential_demand_profile(total_consumption=heating_gas_demand,
                                                              method="valentin")  # value for gas heating demand in tertiary sector
        hourly_ECS = pd.Series(ECS_demand_hourly, index=hourly_gas.index)
        hourly_CH4_gas = hourly_gas + hourly_ECS

        m_scc = ModelEOLES(name="trajectory", config=config, path="eoles/outputs", logger=logger, nb_years=1,
                           heating_demand=dict_heat_modif, nb_linearize=3,
                           linearized_renovation_costs=linearized_renovation_costs,
                           threshold_linearized_renovation_costs=threshold_linearized_renovation_costs,
                           existing_capacity=existing_capacity, existing_charging_capacity=existing_charging_capacity,
                           existing_energy_capacity=existing_energy_capacity,
                           existing_renovation_rate=existing_renovation_rate, maximum_capacity=maximum_capacity,
                           social_cost_of_carbon=scc, year=y, scenario_cost=None, hp_hourly=True,
                           renov=None, hourly_heat_gas=None)
        m_scc.build_model()
        solver_results, status, termination_condition = m_scc.solve(solver_name="gurobi")

        new_capacity = m_scc.capacities - existing_capacity  # we get the newly installed capacities at year y
        new_charging_capacity = m_scc.charging_capacity - existing_charging_capacity
        new_energy_capacity = m_scc.energy_capacity - existing_energy_capacity

        new_capacity_tot = new_capacity_tot + new_capacity  # total newly built capacity over the time horizon
        new_charging_capacity_tot = new_charging_capacity_tot + new_charging_capacity
        new_energy_capacity_tot = new_energy_capacity_tot + new_energy_capacity
        existing_renovation_rate = m_scc.renovation_rates_detailed  # we get new renovation decisions

        capacity_df = pd.concat([capacity_df, m_scc.capacities.to_frame().rename(columns={0: y})], axis=1)
        charging_capacity_df = pd.concat([charging_capacity_df, m_scc.charging_capacity.to_frame().rename(columns={0: y})], axis=1)
        energy_capacity_df = pd.concat([energy_capacity_df, m_scc.energy_capacity.to_frame().rename(columns={0: y})], axis=1)
        new_capacity_df = pd.concat([new_capacity_df, new_capacity.to_frame().rename(columns={0: y})], axis=1)
        renovation_rates_df = pd.concat([renovation_rates_df, m_scc.renovation_rates.to_frame().rename(columns={0: y})], axis=1)