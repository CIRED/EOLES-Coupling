from multiprocessing import Pool
import os
import numpy as np
import pandas as pd
from itertools import product
from eoles.coupling_resirf_eoles import run_resirf, ini_res_irf
from datetime import datetime
import argparse


def define_experience_plan_grid(n):
    list_sub_heater = np.linspace(0.0, 1.0, n)
    list_sub_insulation = np.linspace(0.0, 1.0, n)
    list_sub = list(product(list_sub_heater, list_sub_insulation))
    return list_sub


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Simulate resirf.')
    # parser.add_argument("-s", "--startyear", type=int, help="start year")
    # parser.add_argument("-e", "--timestep", type=int, help="end year")
    # parser.add_argument("-n", "--npoints", type=int, help="number of points for experience plan")
    #
    # args = parser.parse_args()
    #
    # start, timestep, n_points = args.startyear, args.timestep, args.npoints
    start, timestep, n_points = 2020, 2, 3  # to remove if argparse command !!

    buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built, post_inputs = ini_res_irf(
        path=os.path.join("eoles", "outputs"),
        logger=None,
        config=None)

    # run_resirf(0.0, 0.0, buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built,
    #            post_inputs, 2020, 5)

    # list_sub = define_experience_plan_hypercube(d=2, n=100, M=100)  # with latin hypercube
    list_sub = define_experience_plan_grid(n=n_points)  # with grid
    list_argument = [(e[0], e[1],  buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built,
               post_inputs, start, timestep) for e in list_sub]
    print(len(list_argument))

    with Pool(4) as pool:
        results = pool.starmap(run_resirf, list_argument)

    sub_heater = [i[0] for i in results]
    sub_insulation = [i[1] for i in results]
    electricity = [i[2]["Electricity (TWh)"] for i in results]
    gas = [i[2]["Natural gas (TWh)"] for i in results]
    wood = [i[2]["Wood fuel (TWh)"] for i in results]
    oil = [i[2]["Oil fuel (TWh)"] for i in results]
    investment_heater = [i[2]["Investment heater (Billion euro)"] for i in results]
    investment_insulation = [i[2]["Investment insulation (Billion euro)"] for i in results]
    subsidies_heater = [i[2]["Subsidies heater (Billion euro)"] for i in results]
    subsidies_insulation = [i[2]["Subsidies insulation (Billion euro)"] for i in results]
    health_cost = [i[2]["Health cost (Billion euro)"] for i in results]
    replacement_heat_pump_water = [i[2]['Replacement Electricity-Heat pump water (Thousand)'] for i in results]
    replacement_heat_pump_air = [i[2]['Replacement Electricity-Heat pump air (Thousand)'] for i in results]
    replacement_resistive = [i[2]['Replacement Electricity-Performance boiler (Thousand)'] for i in results]
    replacement_gas_boiler = [i[2]['Replacement Natural gas-Performance boiler (Thousand)'] for i in results]
    replacement_oil_boiler = [i[2]['Replacement Oil fuel-Performance boiler (Thousand)'] for i in results]
    replacement_wood_boiler = [i[2]['Replacement Wood fuel-Performance boiler (Thousand)'] for i in results]
    stock_hp_air = [i[2]['Stock Electricity-Heat pump air (Thousand)'] for i in results]
    stock_hp_water = [i[2]['Stock Electricity-Heat pump water (Thousand)'] for i in results]
    stock_resistive = [i[2]['Stock Electricity-Performance boiler (Thousand)'] for i in results]
    stock_gas_boiler_perf = [i[2]['Stock Natural gas-Performance boiler (Thousand)'] for i in results]
    stock_gas_boiler_standard = [i[2]['Stock Natural gas-Standard boiler (Thousand)'] for i in results]
    stock_oil_boiler_perf = [i[2]['Stock Oil fuel-Performance boiler (Thousand)'] for i in results]
    stock_oil_boiler_standard = [i[2]['Stock Oil fuel-Standard boiler (Thousand)'] for i in results]
    stock_wood_boiler_perf = [i[2]['Stock Wood fuel-Performance boiler (Thousand)'] for i in results]
    stock_wood_boiler_standard = [i[2]['Stock Wood fuel-Standard boiler (Thousand)'] for i in results]


    output = pd.DataFrame({'sub_heater': sub_heater, 'sub_insulation': sub_insulation, 'Electricity (TWh)': electricity,
                           'Natural gas (TWh)': gas, 'Wood fuel (TWh)': wood, 'Oil fuel (TWh)': oil, 'Investment heater (Billion euro)': investment_heater,
                           'Investment insulation (Billion euro)': investment_insulation, 'Subsidies heater (Billion euro)': subsidies_heater,
                           'Subsidies insulation (Billion euro)': subsidies_insulation, 'Health cost (Billion euro)': health_cost,
                           'Replacement Electricity-Heat pump water (Thousand)': replacement_heat_pump_water,
                           'Replacement Electricity-Heat pump air (Thousand)': replacement_heat_pump_air,
                           'Replacement Electricity-Performance boiler (Thousand)': replacement_resistive,
                           'Replacement Natural gas-Performance boiler (Thousand)': replacement_gas_boiler,
                           'Replacement Oil fuel-Performance boiler (Thousand)': replacement_oil_boiler,
                           'Replacement Wood fuel-Performance boiler (Thousand)': replacement_wood_boiler,
                           'Stock Electricity-Heat pump air (Thousand)': stock_hp_air,
                           'Stock Electricity-Heat pump water (Thousand)': stock_hp_water,
                           'Stock Electricity-Performance boiler (Thousand)': stock_resistive,
                           'Stock Natural gas-Performance boiler (Thousand)': stock_gas_boiler_perf,
                           'Stock Natural gas-Standard boiler (Thousand)': stock_gas_boiler_standard,
                           'Stock Oil fuel-Performance boiler (Thousand)': stock_oil_boiler_perf,
                           'Stock Oil fuel-Standard boiler (Thousand)': stock_oil_boiler_standard,
                           'Stock Wood fuel-Performance boiler (Thousand)': stock_wood_boiler_perf,
                           'Stock Wood fuel-Standard boiler (Thousand)': stock_wood_boiler_standard})

    day = datetime.now().strftime("%m%d")
    name_file = day + f"_results_start{start}_timestep{timestep}_n{n_points}.csv"
    path_results = os.path.join('eoles', 'outputs', 'sensitivity_resirf', name_file)
    output.to_csv(path_results)
