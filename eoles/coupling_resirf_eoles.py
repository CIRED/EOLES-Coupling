from copy import deepcopy
from project.sensitivity import ini_res_irf, simu_res_irf
from GPyOpt.methods import BayesianOptimization
import numpy as np
import logging
import pandas as pd

from eoles.model_resirf_coupling import ModelEOLES
from eoles.utils import get_config, get_pandas, calculate_annuities_resirf
import eoles.utils

LOG_FORMATTER = '%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# consoler handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
logger.addHandler(console_handler)

HOURLY_PROFILE_METHOD = "valentin"
LIST_REPLACEMENT_HEATER = ['Replacement Electricity-Heat pump water (Thousand)',
                           'Replacement Electricity-Heat pump air (Thousand)',
                           'Replacement Electricity-Performance boiler (Thousand)',
                           'Replacement Natural gas-Performance boiler (Thousand)',
                           'Replacement Oil fuel-Performance boiler (Thousand)',
                           'Replacement Wood fuel-Performance boiler (Thousand)']

LIST_STOCK_HEATER = ['Stock Electricity-Heat pump air (Thousand)',
                     'Stock Electricity-Heat pump water (Thousand)', 'Stock Electricity-Performance boiler (Thousand)',
                     'Stock Natural gas-Performance boiler (Thousand)', 'Stock Natural gas-Standard boiler (Thousand)',
                     'Stock Oil fuel-Performance boiler (Thousand)', 'Stock Oil fuel-Standard boiler (Thousand)',
                     'Stock Wood fuel-Performance boiler (Thousand)', 'Stock Wood fuel-Standard boiler (Thousand)']


def run_resirf(sub_heater, sub_insulation, buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built,
               post_inputs, start_year_resirf, timestep_resirf):
    endyear_resirf = start_year_resirf + timestep_resirf

    buildings_copy = deepcopy(buildings)

    # simulation between start and end - flow output represents annual values for year "end"
    output = simu_res_irf(buildings_copy, sub_heater, sub_insulation, start_year_resirf, endyear_resirf, energy_prices,
                          taxes, cost_heater, cost_insulation, flow_built, post_inputs, output_resirf=False, climate=2006,
                          efficiency_hour=True)

    investment_heater_cost = sum(
        [output[y]["Investment heater (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €
    investment_insulation_cost = sum(
        [output[y]["Investment insulation (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €
    subsidies_heater_cost = sum(
        [output[y]["Subsidies heater (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €
    subsidies_insulation_cost = sum(
        [output[y]["Subsidies insulation (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €
    health_cost = sum(
        [output[y]["Health cost (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €
    # TODO: a changer ici !!
    heating_consumption = output[endyear_resirf - 1]['Consumption (kWh/h)']

    electricity_consumption = heating_consumption.sum(axis=1)["Electricity"]*1e-6  # in GWh
    gas_consumption = heating_consumption.sum(axis=1)["Natural gas"]*1e-6
    wood_consumption = heating_consumption.sum(axis=1)["Wood fuel"]*1e-6
    oil_consumption = heating_consumption.sum(axis=1)["Oil fuel"]*1e-6

    replacement_output = {key: sum(output[y][key] for y in range(start_year_resirf, endyear_resirf)) for key in list(output[endyear_resirf-1].keys()) if "Replacement" in key}
    stock_output = {key: output[endyear_resirf-1][key] for key in list(output[endyear_resirf-1].keys()) if "Stock" in key}

    o = dict()
    o["Electricity (TWh)"] = electricity_consumption
    o["Natural gas (TWh)"] = gas_consumption
    o["Wood fuel (TWh)"] = wood_consumption
    o["Oil fuel (TWh)"] = oil_consumption
    o["Investment heater (Billion euro)"] = investment_heater_cost
    o["Investment insulation (Billion euro)"] = investment_insulation_cost
    o["Subsidies heater (Billion euro)"] = subsidies_heater_cost
    o["Subsidies insulation (Billion euro)"] = subsidies_insulation_cost
    o["Health cost (Billion euro)"] = health_cost
    o.update(replacement_output)
    o.update(stock_output)
    return sub_heater, sub_insulation, o


def resirf_eoles_coupling_static(subvention, buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built,
                                 post_inputs,
                                 start_year_resirf, timestep_resirf, config_eoles,
                                 year_eoles, anticipated_year_eoles, scc, hourly_gas_exogeneous,
                                 existing_capacity, existing_charging_capacity, existing_energy_capacity,
                                 maximum_capacity, method_hourly_profile,
                                 scenario_cost, existing_annualized_costs_elec, existing_annualized_costs_CH4,
                                 existing_annualized_costs_H2, lifetime_renov=40, discount_rate_renov=0.045):
    """

    :param subvention: 2 dimensional numpy array
        Attention to the shape of the array necessary for optimization !!
    :return:
    """

    # example one year simulation
    endyear_resirf = start_year_resirf + timestep_resirf

    sub_heater, sub_insulation = float(subvention[0, 0]), float(subvention[0, 1])
    print(f'Subvention: {sub_heater}, {sub_insulation}')

    buildings_copy = deepcopy(buildings)

    # simulation between start and end - flow output represents annual values for year "end"
    output = simu_res_irf(buildings_copy, sub_heater, sub_insulation, start_year_resirf, endyear_resirf, energy_prices,
                          taxes, cost_heater, cost_insulation, flow_built, post_inputs, output_resirf=False,
                          climate=2006, efficiency_hour=True)  # we use hourly hp cop coefficient

    heating_consumption = output[endyear_resirf - 1]['Consumption (kWh/h)']
    heating_consumption = heating_consumption.sort_index(ascending=True)

    hourly_heat_elec = heating_consumption.loc["Electricity"] * 1e-6  # GWh
    hourly_heat_gas = heating_consumption.loc["Natural gas"] * 1e-6  # GWh
    hourly_heat_elec = hourly_heat_elec.reset_index().drop(columns=["index"]).squeeze()
    hourly_heat_gas = hourly_heat_gas.reset_index().drop(columns=["index"]).squeeze()
    hourly_heat_gas = hourly_heat_gas + hourly_gas_exogeneous  # we add exogeneous gas demand

    oil_consumption = heating_consumption.sum(axis=1)["Oil fuel"]*1e-6  # GWh
    wood_consumption = heating_consumption.sum(axis=1)["Wood fuel"]*1e-6  # GWh
    # TODO: add emissions linked to oil !!

    # # OLD VERSION
    # sub_heater, sub_insulation = subvention[0, 0], subvention[0, 1]
    #
    # sub_heater_return, sub_insulation_return, o = run_resirf(sub_heater, sub_insulation, start_year_resirf, end_year_resirf)

    investment_cost = sum(
        [output[y]["Investment (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €
    subsidies_cost = sum(
        [output[y]["Subsidies (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €
    health_cost = sum(
        [output[y]["Health cost (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €

    # TODO: attention, certains coûts sont peut-être déjà la somme de coûts annuels !!

    annuity_health_cost = calculate_annuities_resirf(health_cost, lifetime_renov, discount_rate_renov)
    annuity_investment_cost = calculate_annuities_resirf(investment_cost, lifetime_renov, discount_rate_renov)
    print(annuity_health_cost, annuity_investment_cost)

    m_eoles = ModelEOLES(name="trajectory", config=config_eoles, path="eoles/outputs", logger=logger, nb_years=1,
                         hourly_heat_elec=hourly_heat_elec, hourly_heat_gas=hourly_heat_gas, heat_wood=wood_consumption,
                         heat_fuel=oil_consumption,
                         existing_capacity=existing_capacity, existing_charging_capacity=existing_charging_capacity,
                         existing_energy_capacity=existing_energy_capacity, maximum_capacity=maximum_capacity,
                         method_hourly_profile=method_hourly_profile,
                         social_cost_of_carbon=scc, year=year_eoles, anticipated_year=anticipated_year_eoles,
                         scenario_cost=scenario_cost, existing_annualized_costs_elec=existing_annualized_costs_elec,
                         existing_annualized_costs_CH4=existing_annualized_costs_CH4,
                         existing_annualized_costs_H2=existing_annualized_costs_H2)
    m_eoles.build_model()
    solver_results, status, termination_condition = m_eoles.solve(solver_name="gurobi")

    objective = m_eoles.results["objective"]
    print(objective)

    # TODO: attention, à changer si on considère plusieurs années météo dans EOLES
    objective += annuity_health_cost
    objective += annuity_investment_cost
    return np.array([objective])  # return an array


def optimize_blackbox_resirf_eoles_coupling(buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built,
                                            post_inputs, start_year_resirf, timestep_resirf, config_eoles,
                                            year_eoles, anticipated_year_eoles, scc, hourly_gas_exogeneous,
                                            existing_capacity, existing_charging_capacity, existing_energy_capacity,
                                            maximum_capacity, method_hourly_profile,
                                            scenario_cost, existing_annualized_costs_elec, existing_annualized_costs_CH4,
                                            existing_annualized_costs_H2, lifetime_renov=30, discount_rate_renov=0.045,
                                            max_iter=20, initial_design_numdata=3, plot=False):
    bounds2d = [{'name': 'sub_heater', 'type': 'continuous', 'domain': (0, 1)},
                {'name': 'sub_insulation', 'type': 'continuous', 'domain': (0, 1)}]

    optimizer = BayesianOptimization(f=lambda x: resirf_eoles_coupling_static(x, buildings=buildings, energy_prices=energy_prices,
                                                                              taxes=taxes, cost_heater=cost_heater,
                                                                              cost_insulation=cost_insulation, flow_built=flow_built,
                                                                              post_inputs=post_inputs, start_year_resirf=start_year_resirf,
                                                                              timestep_resirf=timestep_resirf,
                                                                              config_eoles=config_eoles, year_eoles=year_eoles,
                                                                              anticipated_year_eoles=anticipated_year_eoles,
                                                                              scc=scc, hourly_gas_exogeneous=hourly_gas_exogeneous,
                                                                              existing_capacity=existing_capacity,
                                                                              existing_charging_capacity=existing_charging_capacity,
                                                                              existing_energy_capacity=existing_energy_capacity,
                                                                              maximum_capacity=maximum_capacity,
                                                                              method_hourly_profile=method_hourly_profile,
                                                                              scenario_cost=scenario_cost,
                                                                              existing_annualized_costs_elec=existing_annualized_costs_elec,
                                                                              existing_annualized_costs_CH4=existing_annualized_costs_CH4,
                                                                              existing_annualized_costs_H2=existing_annualized_costs_H2,
                                                                              lifetime_renov=lifetime_renov,
                                                                              discount_rate_renov=discount_rate_renov),
                                     domain=bounds2d,
                                     model_type='GP',  # gaussian process
                                     # kernel=kernel,
                                     acquisition_type='EI',  # expected improvement algorithm
                                     acquisition_jitter=0.01,
                                     noise_var=0,  # no noise in surrogate function evaluation
                                     exact_feval=True,  # no noise in evaluations
                                     normalize_Y=False,
                                     maximize=False,
                                     verbosity=True,
                                     initial_design_numdata=initial_design_numdata)  # number of initial points before starting optimization
    optimizer.run_optimization(max_iter=max_iter)
    if plot:
        optimizer.plot_acquisition()
    return optimizer, optimizer.x_opt


def resirf_eoles_coupling_dynamic(buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built, post_inputs,
                                  list_year, list_trajectory_scc, scenario_cost, config_eoles, max_iter, add_CH4_demand=False,
                                  return_optimizer=False):
    """Performs multistep optimization of capacities and subsidies."""
    # INITIALIZATION OF EOLES PARAMETERS

    # importing evolution of historical capacity and expected evolution of demand
    existing_capacity_historical, existing_charging_capacity_historical, existing_energy_capacity_historical, \
    maximum_capacity_evolution, heating_gas_demand_RTE_timesteps, ECS_gas_demand_RTE_timesteps, annuity_fOM_historical, storage_annuity_historical = eoles.utils.load_evolution_data()

    existing_capacity_historical = existing_capacity_historical.drop(["heat_pump", "resistive", "gas_boiler", "fuel_boiler", "wood_boiler"], axis=0)

    # TODO: il faudra ajouter les capacités pour charging et discharging

    # Initialize results dataframes
    new_capacity_tot = pd.Series(0, index=existing_capacity_historical.index, dtype=float)
    new_charging_capacity_tot = pd.Series(0, index=existing_charging_capacity_historical.index, dtype=float)
    new_energy_capacity_tot = pd.Series(0, index=existing_energy_capacity_historical.index, dtype=float)

    reindex_primary_prod = ["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "nuclear",
                            "methanization", "pyrogazification", "natural_gas"]

    capacity_df = pd.DataFrame(index=existing_capacity_historical.index, dtype=float)
    new_capacity_df = pd.DataFrame(index=existing_capacity_historical.index, dtype=float)
    generation_df = pd.DataFrame(index=existing_capacity_historical.index, dtype=float)
    primary_generation_df = pd.DataFrame(index=reindex_primary_prod, dtype=float)
    charging_capacity_df = pd.DataFrame(index=existing_charging_capacity_historical.index, dtype=float)
    energy_capacity_df = pd.DataFrame(index=existing_energy_capacity_historical.index, dtype=float)

    weighted_average_elec_price = pd.DataFrame(dtype=float)
    weighted_average_CH4_price = pd.DataFrame(dtype=float)
    weighted_average_H2_price = pd.DataFrame(dtype=float)
    lcoe_elec = pd.DataFrame(dtype=float)
    lcoe_elec_volume = pd.DataFrame(dtype=float)
    lcoe_elec_value = pd.DataFrame(dtype=float)

    annualized_costs_new_capacity = pd.DataFrame(0, index=existing_capacity_historical.index, columns=["annualized_costs"], dtype=float)
    annualized_costs_new_energy_capacity = pd.DataFrame(0, index=existing_energy_capacity_historical.index, columns=["annualized_costs"], dtype=float)

    list_sub_heater, list_sub_insulation  = [], []
    list_investment_heater, list_investment_insulation, list_subsidies_heater_cost, list_subsidies_insulation_cost, list_health_cost = [], [], [], [], []
    list_electricity_consumption, list_gas_consumption, list_wood_consumption, list_oil_consumption = [], [], [], []
    replacement_heater_df = pd.DataFrame(index=LIST_REPLACEMENT_HEATER, dtype=float)
    stock_heater_df = pd.DataFrame(index=LIST_STOCK_HEATER, dtype=float)

    for y, scc in zip(list_year, list_trajectory_scc):
        print(f"Year {y}, SCC {scc}")
        # TODO: a modifier selon ce qu'on souhaite faire comme anticipation
        if y < 2050:
            year_eoles, anticipated_year_eoles, start_year_resirf, timestep_resirf = y, y+5, y, 5
        else:  # specific case for 2050
            year_eoles, anticipated_year_eoles, start_year_resirf, timestep_resirf = y, y, y, 5
        # if y < 2045:
        #     anticipated_year = y + 10
        # else:
        #     anticipated_year = 2050

        #### Get existing and maximum capacities
        existing_capa_historical_y = existing_capacity_historical[
            [str(anticipated_year_eoles)]].squeeze()  # get historical capacity still installed for year of interest
        existing_charging_capacity_historical_y = existing_charging_capacity_historical[
            [str(anticipated_year_eoles)]].squeeze()
        existing_energy_capacity_historical_y = existing_energy_capacity_historical[[str(anticipated_year_eoles)]].squeeze()

        new_maximum_capacity_y = maximum_capacity_evolution[[str(anticipated_year_eoles)]].squeeze()  # get maximum new capacity to be built

        # Existing capacities at year y
        existing_capacity = existing_capa_historical_y + new_capacity_tot  # existing capacity are equal to newly built
        # capacities over the whole time horizon before t + existing capacity (from before 2020)
        existing_charging_capacity = existing_charging_capacity_historical_y + new_charging_capacity_tot
        existing_energy_capacity = existing_energy_capacity_historical_y + new_energy_capacity_tot

        # We reinitialize the heating vectors so that we can adapt at each time step the way to satisfy heating demand (ambitious hypothesis)
        # existing_capacity[["heat_pump", "resistive", "gas_boiler", "wood_boiler", "fuel_boiler"]] = 0

        maximum_capacity = (
                    existing_capacity + new_maximum_capacity_y).dropna()  # we drop nan values, which correspond to technologies without any upper bound

        #### Historical LCOE based on historical costs
        annualized_costs_capacity_historical, annualized_costs_energy_capacity_historical = eoles.utils.annualized_costs_investment_historical(
            existing_capa_historical_y, annuity_fOM_historical,
            existing_energy_capacity_historical_y, storage_annuity_historical)

        ### Compile total annualized investment costs from existing capacities (both historical capacities + newly built capacities before t)
        # Necessary for calculus of LCOE accounting for evolution of capacities
        annualized_costs_capacity = pd.concat(
            [annualized_costs_capacity_historical.rename(columns={'annualized_costs': 'historical_annualized_costs'}),
             annualized_costs_new_capacity], axis=1)
        annualized_costs_capacity['annualized_costs'] = annualized_costs_capacity['historical_annualized_costs'] + \
                                                        annualized_costs_capacity['annualized_costs']
        annualized_costs_energy_capacity = pd.concat([annualized_costs_energy_capacity_historical.rename(
            columns={'annualized_costs': 'historical_annualized_costs'}), annualized_costs_new_energy_capacity], axis=1)
        annualized_costs_energy_capacity['annualized_costs'] = annualized_costs_energy_capacity[
                                                                   'historical_annualized_costs'] + \
                                                               annualized_costs_energy_capacity['annualized_costs']

        existing_annualized_costs_elec, existing_annualized_costs_CH4, existing_annualized_costs_H2 = eoles.utils.process_annualized_costs_per_vector(
            annualized_costs_capacity[["annualized_costs"]].squeeze(), annualized_costs_energy_capacity[["annualized_costs"]].squeeze())
        # print(existing_annualized_costs_elec, existing_annualized_costs_CH4)

        if add_CH4_demand:
            # Create additional gas profile (tertiary heating + ECS)
            heating_gas_demand = heating_gas_demand_RTE_timesteps[anticipated_year_eoles] * 1e3  # in TWh
            ECS_gas_demand = ECS_gas_demand_RTE_timesteps[anticipated_year_eoles] * 1e3  # in TWh
            ECS_demand_hourly = ECS_gas_demand / 8760
            hourly_gas = eoles.utils.create_hourly_residential_demand_profile(total_consumption=heating_gas_demand,
                                                                  method=HOURLY_PROFILE_METHOD)  # value for gas heating demand in tertiary sector
            hourly_ECS = pd.Series(ECS_demand_hourly, index=hourly_gas.index)
            hourly_exogeneous_CH4 = hourly_gas + hourly_ECS
        else:
            # we do not add any demand
            hourly_exogeneous_CH4 = eoles.utils.create_hourly_residential_demand_profile(total_consumption=0,
                                                                  method=HOURLY_PROFILE_METHOD)

        # Find optimal subsidy
        optimizer, opt_sub = \
            optimize_blackbox_resirf_eoles_coupling(buildings, energy_prices, taxes, cost_heater, cost_insulation,
                                                    flow_built, post_inputs, start_year_resirf, timestep_resirf,
                                                    config_eoles, year_eoles, anticipated_year_eoles, scc,
                                                    hourly_gas_exogeneous=hourly_exogeneous_CH4,
                                                    existing_capacity=existing_capacity, existing_charging_capacity=existing_charging_capacity,
                                                    existing_energy_capacity=existing_energy_capacity,
                                                    maximum_capacity=maximum_capacity, method_hourly_profile="valentin",
                                                    scenario_cost=scenario_cost, existing_annualized_costs_elec=existing_annualized_costs_elec,
                                                    existing_annualized_costs_CH4=existing_annualized_costs_CH4,
                                                    existing_annualized_costs_H2=existing_annualized_costs_H2,
                                                    lifetime_renov=40, discount_rate_renov=0.045, plot=False,
                                                    max_iter=max_iter, initial_design_numdata=3)
        opt_sub_heater, opt_sub_insulation = opt_sub[0], opt_sub[1]
        list_sub_heater.append(opt_sub_heater)
        list_sub_insulation.append(opt_sub_insulation)

        # Rerun ResIRF with optimal subvention parameters
        endyear_resirf = start_year_resirf + timestep_resirf

        output_opt = simu_res_irf(buildings, opt_sub_heater, opt_sub_insulation, start_year_resirf, endyear_resirf,
                                  energy_prices, taxes, cost_heater, cost_insulation, flow_built, post_inputs,
                                  output_resirf=False, climate=2006, efficiency_hour=True)

        heating_consumption = output_opt[endyear_resirf - 1]['Consumption (kWh/h)']
        heating_consumption = heating_consumption.sort_index(ascending=True)

        hourly_heat_elec = heating_consumption.loc["Electricity"] * 1e-6  # GWh
        hourly_heat_gas = heating_consumption.loc["Natural gas"] * 1e-6  # GWh
        hourly_heat_elec = hourly_heat_elec.reset_index().drop(columns=["index"]).squeeze()
        hourly_heat_gas = hourly_heat_gas.reset_index().drop(columns=["index"]).squeeze()
        hourly_heat_gas = hourly_heat_gas + hourly_exogeneous_CH4  # we add exogeneous gas demand

        electricity_consumption = heating_consumption.sum(axis=1)["Electricity"] * 1e-6
        gas_consumption = heating_consumption.sum(axis=1)["Natural gas"] * 1e-6
        oil_consumption = heating_consumption.sum(axis=1)["Oil fuel"] * 1e-6  # GWh
        wood_consumption = heating_consumption.sum(axis=1)["Wood fuel"] * 1e-6  # GWh

        investment_heater_cost = sum(
            [output_opt[y]["Investment heater (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €
        investment_insulation_cost = sum(
            [output_opt[y]["Investment insulation (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €
        subsidies_heater_cost = sum(
            [output_opt[y]["Subsidies heater (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €
        subsidies_insulation_cost = sum(
            [output_opt[y]["Subsidies insulation (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €
        health_cost = sum(
            [output_opt[y]["Health cost (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €

        replacement_output = pd.Series({key: sum(output_opt[y][key] for y in range(start_year_resirf, endyear_resirf)) for key in
                              list(output_opt[endyear_resirf - 1].keys()) if "Replacement" in key})
        stock_output = pd.Series({key: output_opt[endyear_resirf - 1][key] for key in list(output_opt[endyear_resirf - 1].keys()) if
                        "Stock" in key})

        list_investment_heater.append(investment_heater_cost)
        list_investment_insulation.append(investment_insulation_cost)
        list_subsidies_heater_cost.append(subsidies_heater_cost)
        list_subsidies_insulation_cost.append(subsidies_insulation_cost)
        list_health_cost.append(health_cost)
        list_electricity_consumption.append(electricity_consumption)
        list_gas_consumption.append(gas_consumption)
        list_wood_consumption.append(wood_consumption)
        list_oil_consumption.append(oil_consumption)
        replacement_heater_df = pd.concat([replacement_heater_df, replacement_output.to_frame().rename(columns={0: y})], axis=1)
        stock_heater_df = pd.concat([stock_heater_df, stock_output.to_frame().rename(columns={0: y})], axis=1)

        # TODO: ajouter des dictionnaires avec les outputs d'intérêt

        # Rerun EOLES with optimal parameters
        m_eoles = ModelEOLES(name="trajectory", config=config_eoles, path="eoles/outputs", logger=logger, nb_years=1,
                             hourly_heat_elec=hourly_heat_elec, hourly_heat_gas=hourly_heat_gas, heat_wood=wood_consumption,
                             heat_fuel=oil_consumption,
                             existing_capacity=existing_capacity, existing_charging_capacity=existing_charging_capacity,
                             existing_energy_capacity=existing_energy_capacity, maximum_capacity=maximum_capacity,
                             method_hourly_profile="valentin",
                             social_cost_of_carbon=scc, year=year_eoles, anticipated_year=anticipated_year_eoles,
                             scenario_cost=scenario_cost, existing_annualized_costs_elec=existing_annualized_costs_elec,
                             existing_annualized_costs_CH4=existing_annualized_costs_CH4,
                             existing_annualized_costs_H2=existing_annualized_costs_H2)

        m_eoles.build_model()
        solver_results, status, termination_condition = m_eoles.solve(solver_name="gurobi")

        #### Update total capacities and new capacities
        new_capacity = m_eoles.capacities - existing_capacity  # we get the newly installed capacities at year y
        new_charging_capacity = m_eoles.charging_capacity - existing_charging_capacity
        new_energy_capacity = m_eoles.energy_capacity - existing_energy_capacity
        gene_per_tec = pd.Series(m_eoles.generation_per_technology)

        new_capacity_tot = new_capacity_tot + new_capacity  # total newly built capacity over the time horizon until t included
        new_charging_capacity_tot = new_charging_capacity_tot + new_charging_capacity
        new_energy_capacity_tot = new_energy_capacity_tot + new_energy_capacity

        # update annualized costs from new capacity built until time t included
        annualized_costs_new_capacity = pd.concat([annualized_costs_new_capacity[["annualized_costs"]],
                                                   m_eoles.new_capacity_annualized_costs[["annualized_costs"]].rename(
                                                       columns={'annualized_costs': 'new_annualized_costs'})], axis=1)
        annualized_costs_new_capacity["annualized_costs"] = annualized_costs_new_capacity["annualized_costs"] + \
                                                            annualized_costs_new_capacity['new_annualized_costs']
        annualized_costs_new_capacity = annualized_costs_new_capacity.drop(columns=['new_annualized_costs'])

        annualized_costs_new_energy_capacity = pd.concat([annualized_costs_new_energy_capacity[["annualized_costs"]],
                                                          m_eoles.new_energy_capacity_annualized_costs[
                                                              ["annualized_costs"]].rename(
                                                              columns={'annualized_costs': 'new_annualized_costs'})],
                                                         axis=1)
        annualized_costs_new_energy_capacity["annualized_costs"] = annualized_costs_new_energy_capacity[
                                                                       "annualized_costs"] + \
                                                                   annualized_costs_new_energy_capacity[
                                                                       'new_annualized_costs']
        annualized_costs_new_energy_capacity = annualized_costs_new_energy_capacity.drop(
            columns=['new_annualized_costs'])

        capacity_df = pd.concat([capacity_df, m_eoles.capacities.to_frame().rename(columns={0: y})], axis=1)
        charging_capacity_df = pd.concat(
            [charging_capacity_df, m_eoles.charging_capacity.to_frame().rename(columns={0: y})], axis=1)
        generation_df = pd.concat([generation_df, gene_per_tec.to_frame().rename(columns={0: y})], axis=1)
        energy_capacity_df = pd.concat([energy_capacity_df, m_eoles.energy_capacity.to_frame().rename(columns={0: y})],
                                       axis=1)
        new_capacity_df = pd.concat([new_capacity_df, new_capacity.to_frame().rename(columns={0: y})], axis=1)
        primary_generation_df = pd.concat([primary_generation_df,
                                           m_eoles.primary_generation.reindex(reindex_primary_prod).to_frame().rename(
                                               columns={0: y})], axis=1)

        weighted_average_elec_price = pd.concat([weighted_average_elec_price, pd.Series(
            m_eoles.summary["weighted_elec_price_demand"]).to_frame().rename(columns={0: y})], axis=1)
        weighted_average_CH4_price = pd.concat([weighted_average_CH4_price,
                                                pd.Series(m_eoles.summary["weighted_CH4_price_demand"]).to_frame().rename(
                                                    columns={0: y})], axis=1)
        lcoe_elec = pd.concat([lcoe_elec, pd.Series(m_eoles.summary["lcoe_elec"]).to_frame().rename(columns={0: y})],
                              axis=1)
        lcoe_elec_volume = pd.concat(
            [lcoe_elec_volume, pd.Series(m_eoles.summary["lcoe_elec_volume"]).to_frame().rename(columns={0: y})], axis=1)
        lcoe_elec_value = pd.concat(
            [lcoe_elec_value, pd.Series(m_eoles.summary["lcoe_elec_value"]).to_frame().rename(columns={0: y})], axis=1)
    output = {
        "capacity": capacity_df,
        "new_capacity": new_capacity_df,
        "generation": generation_df,
        "primary_generation": primary_generation_df,
        "charging_capacity": charging_capacity_df,
        "energy_capacity": energy_capacity_df,
        "weighted_average_elec_price": weighted_average_elec_price,
        "weighted_average_CH4_price": weighted_average_CH4_price,
        "weighted_average_H2_price": weighted_average_H2_price,
        "lcoe_elec": lcoe_elec,
        "lcoe_elec_volume": lcoe_elec_volume,
        "lcoe_elec_value": lcoe_elec_value,
        "sub_heater": list_sub_heater,
        "sub_insulation": list_sub_insulation,
        "investment_heater_cost": list_investment_heater,
        "investment_insulation_cost": list_investment_insulation,
        "investment_subsidies_heater_cost": list_subsidies_heater_cost,
        "investment_subsidies_insulation_cost": list_subsidies_insulation_cost,
        "health_cost": list_health_cost,
        "electricity_consumption": list_electricity_consumption,
        "gas_consumption": list_gas_consumption,
        "wood_consumption": list_wood_consumption,
        "oil_consumption": list_oil_consumption,
        "replacement_heater": replacement_heater_df,
        "stock_heater": stock_heater_df
    }
    if return_optimizer:
        return output, optimizer
    return output


def resirf_eoles_coupling_dynamic_no_opti(list_sub_heater, list_sub_insulation, buildings, energy_prices, taxes,
                                          cost_heater, cost_insulation, flow_built, post_inputs,
                                          list_year, list_trajectory_scc, scenario_cost, config_eoles,
                                          add_CH4_demand=False):
    """Computes multistep optimization of capacities. (Optimal) subsidies are taken as input."""
    # INITIALIZATION OF EOLES PARAMETERS

    # importing evolution of historical capacity and expected evolution of demand
    existing_capacity_historical, existing_charging_capacity_historical, existing_energy_capacity_historical, \
    maximum_capacity_evolution, heating_gas_demand_RTE_timesteps, ECS_gas_demand_RTE_timesteps, annuity_fOM_historical, storage_annuity_historical = eoles.utils.load_evolution_data()

    existing_capacity_historical = existing_capacity_historical.drop(["heat_pump", "resistive", "gas_boiler", "fuel_boiler", "wood_boiler"], axis=0)

    # TODO: il faudra ajouter les capacités pour charging et discharging

    # Initialize results dataframes
    new_capacity_tot = pd.Series(0, index=existing_capacity_historical.index, dtype=float)
    new_charging_capacity_tot = pd.Series(0, index=existing_charging_capacity_historical.index, dtype=float)
    new_energy_capacity_tot = pd.Series(0, index=existing_energy_capacity_historical.index, dtype=float)

    reindex_primary_prod = ["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "nuclear",
                            "methanization", "pyrogazification", "natural_gas"]

    capacity_df = pd.DataFrame(index=existing_capacity_historical.index, dtype=float)
    new_capacity_df = pd.DataFrame(index=existing_capacity_historical.index, dtype=float)
    generation_df = pd.DataFrame(index=existing_capacity_historical.index, dtype=float)
    primary_generation_df = pd.DataFrame(index=reindex_primary_prod, dtype=float)
    charging_capacity_df = pd.DataFrame(index=existing_charging_capacity_historical.index, dtype=float)
    energy_capacity_df = pd.DataFrame(index=existing_energy_capacity_historical.index, dtype=float)

    weighted_average_elec_price = pd.DataFrame(dtype=float)
    weighted_average_CH4_price = pd.DataFrame(dtype=float)
    weighted_average_H2_price = pd.DataFrame(dtype=float)
    lcoe_elec = pd.DataFrame(dtype=float)
    lcoe_elec_volume = pd.DataFrame(dtype=float)
    lcoe_elec_value = pd.DataFrame(dtype=float)

    annualized_costs_new_capacity = pd.DataFrame(0, index=existing_capacity_historical.index, columns=["annualized_costs"], dtype=float)
    annualized_costs_new_energy_capacity = pd.DataFrame(0, index=existing_energy_capacity_historical.index, columns=["annualized_costs"], dtype=float)

    list_investment_heater, list_investment_insulation, list_subsidies_heater_cost, list_subsidies_insulation_cost, list_health_cost = [], [], [], [], []
    list_electricity_consumption, list_gas_consumption, list_wood_consumption, list_oil_consumption = [], [], [], []
    replacement_heater_df = pd.DataFrame(index=LIST_REPLACEMENT_HEATER, dtype=float)
    stock_heater_df = pd.DataFrame(index=LIST_STOCK_HEATER, dtype=float)

    t = 0
    for y, scc in zip(list_year, list_trajectory_scc):
        print(f"Year {y}, SCC {scc}")
        # TODO: a modifier selon ce qu'on souhaite faire comme anticipation
        if y < 2050:
            year_eoles, anticipated_year_eoles, start_year_resirf, timestep_resirf = y, y+5, y, 5
        else:  # specific case for 2050
            year_eoles, anticipated_year_eoles, start_year_resirf, timestep_resirf = y, y, y, 5
        # if y < 2045:
        #     anticipated_year = y + 10
        # else:
        #     anticipated_year = 2050

        #### Get existing and maximum capacities
        existing_capa_historical_y = existing_capacity_historical[
            [str(anticipated_year_eoles)]].squeeze()  # get historical capacity still installed for year of interest
        existing_charging_capacity_historical_y = existing_charging_capacity_historical[
            [str(anticipated_year_eoles)]].squeeze()
        existing_energy_capacity_historical_y = existing_energy_capacity_historical[[str(anticipated_year_eoles)]].squeeze()

        new_maximum_capacity_y = maximum_capacity_evolution[[str(anticipated_year_eoles)]].squeeze()  # get maximum new capacity to be built

        # Existing capacities at year y
        existing_capacity = existing_capa_historical_y + new_capacity_tot  # existing capacity are equal to newly built
        # capacities over the whole time horizon before t + existing capacity (from before 2020)
        existing_charging_capacity = existing_charging_capacity_historical_y + new_charging_capacity_tot
        existing_energy_capacity = existing_energy_capacity_historical_y + new_energy_capacity_tot

        # We reinitialize the heating vectors so that we can adapt at each time step the way to satisfy heating demand (ambitious hypothesis)
        # existing_capacity[["heat_pump", "resistive", "gas_boiler", "wood_boiler", "fuel_boiler"]] = 0

        maximum_capacity = (
                    existing_capacity + new_maximum_capacity_y).dropna()  # we drop nan values, which correspond to technologies without any upper bound

        #### Historical LCOE based on historical costs
        annualized_costs_capacity_historical, annualized_costs_energy_capacity_historical = eoles.utils.annualized_costs_investment_historical(
            existing_capa_historical_y, annuity_fOM_historical,
            existing_energy_capacity_historical_y, storage_annuity_historical)

        ### Compile total annualized investment costs from existing capacities (both historical capacities + newly built capacities before t)
        # Necessary for calculus of LCOE accounting for evolution of capacities
        annualized_costs_capacity = pd.concat(
            [annualized_costs_capacity_historical.rename(columns={'annualized_costs': 'historical_annualized_costs'}),
             annualized_costs_new_capacity], axis=1)
        annualized_costs_capacity['annualized_costs'] = annualized_costs_capacity['historical_annualized_costs'] + \
                                                        annualized_costs_capacity['annualized_costs']
        annualized_costs_energy_capacity = pd.concat([annualized_costs_energy_capacity_historical.rename(
            columns={'annualized_costs': 'historical_annualized_costs'}), annualized_costs_new_energy_capacity], axis=1)
        annualized_costs_energy_capacity['annualized_costs'] = annualized_costs_energy_capacity[
                                                                   'historical_annualized_costs'] + \
                                                               annualized_costs_energy_capacity['annualized_costs']

        existing_annualized_costs_elec, existing_annualized_costs_CH4, existing_annualized_costs_H2 = eoles.utils.process_annualized_costs_per_vector(
            annualized_costs_capacity[["annualized_costs"]].squeeze(), annualized_costs_energy_capacity[["annualized_costs"]].squeeze())
        # print(existing_annualized_costs_elec, existing_annualized_costs_CH4)

        if add_CH4_demand:
            # Create additional gas profile (tertiary heating + ECS)
            heating_gas_demand = heating_gas_demand_RTE_timesteps[anticipated_year_eoles] * 1e3  # in TWh
            ECS_gas_demand = ECS_gas_demand_RTE_timesteps[anticipated_year_eoles] * 1e3  # in TWh
            ECS_demand_hourly = ECS_gas_demand / 8760
            hourly_gas = eoles.utils.create_hourly_residential_demand_profile(total_consumption=heating_gas_demand,
                                                                  method=HOURLY_PROFILE_METHOD)  # value for gas heating demand in tertiary sector
            hourly_ECS = pd.Series(ECS_demand_hourly, index=hourly_gas.index)
            hourly_exogeneous_CH4 = hourly_gas + hourly_ECS
        else:
            # we do not add any demand
            hourly_exogeneous_CH4 = eoles.utils.create_hourly_residential_demand_profile(total_consumption=0,
                                                                  method=HOURLY_PROFILE_METHOD)

        opt_sub_heater, opt_sub_insulation = list_sub_heater[t], list_sub_insulation[t]

        # Rerun ResIRF with optimal subvention parameters
        endyear_resirf = start_year_resirf + timestep_resirf

        output_opt = simu_res_irf(buildings, opt_sub_heater, opt_sub_insulation, start_year_resirf, endyear_resirf,
                              energy_prices, taxes, cost_heater, cost_insulation, flow_built, post_inputs,
                                  output_resirf=False, climate=2006, efficiency_hour=True)

        heating_consumption = output_opt[endyear_resirf - 1]['Consumption (kWh/h)']
        heating_consumption = heating_consumption.sort_index(ascending=True)

        hourly_heat_elec = heating_consumption.loc["Electricity"] * 1e-6  # GWh
        hourly_heat_gas = heating_consumption.loc["Natural gas"] * 1e-6  # GWh
        hourly_heat_elec = hourly_heat_elec.reset_index().drop(columns=["index"]).squeeze()
        hourly_heat_gas = hourly_heat_gas.reset_index().drop(columns=["index"]).squeeze()
        hourly_heat_gas = hourly_heat_gas + hourly_exogeneous_CH4  # we add exogeneous gas demand

        electricity_consumption = heating_consumption.sum(axis=1)["Electricity"] * 1e-6
        gas_consumption = heating_consumption.sum(axis=1)["Natural gas"] * 1e-6
        oil_consumption = heating_consumption.sum(axis=1)["Oil fuel"] * 1e-6  # GWh
        wood_consumption = heating_consumption.sum(axis=1)["Wood fuel"] * 1e-6  # GWh

        investment_heater_cost = sum(
            [output_opt[y]["Investment heater (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €
        investment_insulation_cost = sum(
            [output_opt[y]["Investment insulation (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €
        subsidies_heater_cost = sum(
            [output_opt[y]["Subsidies heater (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €
        subsidies_insulation_cost = sum(
            [output_opt[y]["Subsidies insulation (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €
        health_cost = sum(
            [output_opt[y]["Health cost (Billion euro)"] for y in range(start_year_resirf, endyear_resirf)])  # 1e9 €

        replacement_output = pd.Series({key: sum(output_opt[y][key] for y in range(start_year_resirf, endyear_resirf)) for key in
                              list(output_opt[endyear_resirf - 1].keys()) if "Replacement" in key})
        stock_output = pd.Series({key: output_opt[endyear_resirf - 1][key] for key in list(output_opt[endyear_resirf - 1].keys()) if
                        "Stock" in key})

        list_investment_heater.append(investment_heater_cost)
        list_investment_insulation.append(investment_insulation_cost)
        list_subsidies_heater_cost.append(subsidies_heater_cost)
        list_subsidies_insulation_cost.append(subsidies_insulation_cost)
        list_health_cost.append(health_cost)
        list_electricity_consumption.append(electricity_consumption)
        list_gas_consumption.append(gas_consumption)
        list_wood_consumption.append(wood_consumption)
        list_oil_consumption.append(oil_consumption)
        replacement_heater_df = pd.concat([replacement_heater_df, replacement_output.to_frame().rename(columns={0: y})], axis=1)
        stock_heater_df = pd.concat([stock_heater_df, stock_output.to_frame().rename(columns={0: y})], axis=1)

        # Rerun EOLES with optimal parameters
        m_eoles = ModelEOLES(name="trajectory", config=config_eoles, path="eoles/outputs", logger=logger, nb_years=1,
                             hourly_heat_elec=hourly_heat_elec, hourly_heat_gas=hourly_heat_gas, heat_wood=wood_consumption,
                             heat_fuel=oil_consumption,
                             existing_capacity=existing_capacity, existing_charging_capacity=existing_charging_capacity,
                             existing_energy_capacity=existing_energy_capacity, maximum_capacity=maximum_capacity,
                             method_hourly_profile="valentin",
                             social_cost_of_carbon=scc, year=year_eoles, anticipated_year=anticipated_year_eoles,
                             scenario_cost=scenario_cost, existing_annualized_costs_elec=existing_annualized_costs_elec,
                             existing_annualized_costs_CH4=existing_annualized_costs_CH4,
                             existing_annualized_costs_H2=existing_annualized_costs_H2)

        m_eoles.build_model()
        solver_results, status, termination_condition = m_eoles.solve(solver_name="gurobi")

        #### Update total capacities and new capacities
        new_capacity = m_eoles.capacities - existing_capacity  # we get the newly installed capacities at year y
        new_charging_capacity = m_eoles.charging_capacity - existing_charging_capacity
        new_energy_capacity = m_eoles.energy_capacity - existing_energy_capacity
        gene_per_tec = pd.Series(m_eoles.generation_per_technology)

        new_capacity_tot = new_capacity_tot + new_capacity  # total newly built capacity over the time horizon until t included
        new_charging_capacity_tot = new_charging_capacity_tot + new_charging_capacity
        new_energy_capacity_tot = new_energy_capacity_tot + new_energy_capacity

        # update annualized costs from new capacity built until time t included
        annualized_costs_new_capacity = pd.concat([annualized_costs_new_capacity[["annualized_costs"]],
                                                   m_eoles.new_capacity_annualized_costs[["annualized_costs"]].rename(
                                                       columns={'annualized_costs': 'new_annualized_costs'})], axis=1)
        annualized_costs_new_capacity["annualized_costs"] = annualized_costs_new_capacity["annualized_costs"] + \
                                                            annualized_costs_new_capacity['new_annualized_costs']
        annualized_costs_new_capacity = annualized_costs_new_capacity.drop(columns=['new_annualized_costs'])

        annualized_costs_new_energy_capacity = pd.concat([annualized_costs_new_energy_capacity[["annualized_costs"]],
                                                          m_eoles.new_energy_capacity_annualized_costs[
                                                              ["annualized_costs"]].rename(
                                                              columns={'annualized_costs': 'new_annualized_costs'})],
                                                         axis=1)
        annualized_costs_new_energy_capacity["annualized_costs"] = annualized_costs_new_energy_capacity[
                                                                       "annualized_costs"] + \
                                                                   annualized_costs_new_energy_capacity[
                                                                       'new_annualized_costs']
        annualized_costs_new_energy_capacity = annualized_costs_new_energy_capacity.drop(
            columns=['new_annualized_costs'])

        capacity_df = pd.concat([capacity_df, m_eoles.capacities.to_frame().rename(columns={0: y})], axis=1)
        charging_capacity_df = pd.concat(
            [charging_capacity_df, m_eoles.charging_capacity.to_frame().rename(columns={0: y})], axis=1)
        generation_df = pd.concat([generation_df, gene_per_tec.to_frame().rename(columns={0: y})], axis=1)
        energy_capacity_df = pd.concat([energy_capacity_df, m_eoles.energy_capacity.to_frame().rename(columns={0: y})],
                                       axis=1)
        new_capacity_df = pd.concat([new_capacity_df, new_capacity.to_frame().rename(columns={0: y})], axis=1)
        primary_generation_df = pd.concat([primary_generation_df,
                                           m_eoles.primary_generation.reindex(reindex_primary_prod).to_frame().rename(
                                               columns={0: y})], axis=1)

        weighted_average_elec_price = pd.concat([weighted_average_elec_price, pd.Series(
            m_eoles.summary["weighted_elec_price_demand"]).to_frame().rename(columns={0: y})], axis=1)
        weighted_average_CH4_price = pd.concat([weighted_average_CH4_price,
                                                pd.Series(m_eoles.summary["weighted_CH4_price_demand"]).to_frame().rename(
                                                    columns={0: y})], axis=1)
        lcoe_elec = pd.concat([lcoe_elec, pd.Series(m_eoles.summary["lcoe_elec"]).to_frame().rename(columns={0: y})],
                              axis=1)
        lcoe_elec_volume = pd.concat(
            [lcoe_elec_volume, pd.Series(m_eoles.summary["lcoe_elec_volume"]).to_frame().rename(columns={0: y})], axis=1)
        lcoe_elec_value = pd.concat(
            [lcoe_elec_value, pd.Series(m_eoles.summary["lcoe_elec_value"]).to_frame().rename(columns={0: y})], axis=1)

        t += 1  # increase time step
    output = {
        "capacity": capacity_df,
        "new_capacity": new_capacity_df,
        "generation": generation_df,
        "primary_generation": primary_generation_df,
        "charging_capacity": charging_capacity_df,
        "energy_capacity": energy_capacity_df,
        "weighted_average_elec_price": weighted_average_elec_price,
        "weighted_average_CH4_price": weighted_average_CH4_price,
        "weighted_average_H2_price": weighted_average_H2_price,
        "lcoe_elec": lcoe_elec,
        "lcoe_elec_volume": lcoe_elec_volume,
        "lcoe_elec_value": lcoe_elec_value,
        "sub_heater": list_sub_heater,
        "sub_insulation": list_sub_insulation,
        "investment_heater_cost": list_investment_heater,
        "investment_insulation_cost": list_investment_insulation,
        "investment_subsidies_heater_cost": list_subsidies_heater_cost,
        "investment_subsidies_insulation_cost": list_subsidies_insulation_cost,
        "health_cost": list_health_cost,
        "electricity_consumption": list_electricity_consumption,
        "gas_consumption": list_gas_consumption,
        "wood_consumption": list_wood_consumption,
        "oil_consumption": list_oil_consumption,
        "replacement_heater": replacement_heater_df,
        "stock_heater": stock_heater_df
    }

    return output
