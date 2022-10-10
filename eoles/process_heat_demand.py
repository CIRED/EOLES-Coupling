import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # First method: using Behrang hourly profiles
    heat_demand = pd.read_csv("inputs/heat_demand_2050.csv", index_col=0, header=None,
                              names=["heat_type", "demand"])  # final use heat demand, ADEME, GWh-th
    # remark: this corresponds to residential + tertiary, and heating + hot water + cooking --> more stuff than simply residential heating
    hp_cop = pd.read_csv("inputs/HP_COP.csv", index_col=0,
                         header=None)  # conversion factor heat pump, depending on temperature
    hp_cop = hp_cop.squeeze()
    low_heat_demand = (heat_demand.loc[heat_demand.heat_type == "lowT"]).drop(columns=["heat_type"]).squeeze()
    percent_hp_RTE = 0.4
    percent_resistive_RTE = 0.6
    eta_resistive = 0.9

    electricity_heat_demand = percent_hp_RTE * (low_heat_demand / hp_cop) + percent_resistive_RTE * (
            low_heat_demand / eta_resistive)  # GWh

    # electricity_heat_demand = electricity_heat_demand + (33.7*1e3 - electricity_heat_demand.sum())/8760  # rescale to fit RTE total demand simply on residential heating

    demand_elec_RTE = pd.read_csv("inputs/demand2050_RTE.csv", index_col=0, header=None)
    demand_elec_RTE = demand_elec_RTE.squeeze()
    # demand_elec_RTE_no_residential_heating = demand_elec_RTE - electricity_heat_demand
    # print(electricity_heat_demand.sum())

    # Second method: directly using profiles from Doudard (2018)
    daily_profile = [1 / 24 for i in range(24)]
    monthly_profile = [0.24, 0.18, 0.15, 0.05, 0.01, 0, 0, 0, 0, 0.03, 0.12, 0.22]
    days_by_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    percentage_hourly_residential_profile = []
    for i in range(len(days_by_month)):
        month_profile = []
        for j in range(days_by_month[i]):
            month_profile = month_profile + daily_profile
        rescale_month_profile = [p / (days_by_month[i]) * monthly_profile[i] for p in month_profile]
        percentage_hourly_residential_profile = percentage_hourly_residential_profile + rescale_month_profile  # we rescale to the number of hours in the month, and to the percentage of the month

    percentage_hourly_residential_profile = pd.Series(percentage_hourly_residential_profile)
    total_residential_heating = 33 * 1e3
    hourly_residential_heating = total_residential_heating * percentage_hourly_residential_profile

    demand_elec_RTE_no_residential_heating = demand_elec_RTE - hourly_residential_heating

    plt.plot(np.arange(0, 8760, 1), demand_elec_RTE_no_residential_heating)
    plt.show()

    demand_elec_RTE_no_residential_heating.to_csv("inputs/demand2050_RTE_no_residential_heating.csv", header=False)
    hourly_residential_heating.to_csv("inputs/hourly_residential_heating_RTE.csv", header=False)
    percentage_hourly_residential_profile.to_csv("inputs/percentage_hourly_residential_heating_profile.csv", header=False)
