import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime

from eoles.utils import get_pandas

#### FILE USED TO CREATE PROFILES FOR HEATING DEMAND ####

if __name__ == '__main__':
    # ### First method: using Behrang hourly profiles
    # heat_demand = pd.read_csv("inputs/heat_demand_2050.csv", index_col=0, header=None,
    #                           names=["heat_type", "demand"])  # final use heat demand, ADEME, GWh-th
    # # remark: this corresponds to residential + tertiary, and heating + hot water + cooking --> more stuff than simply residential heating
    # hp_cop = pd.read_csv("inputs/hp_cop.csv", index_col=0,
    #                      header=None)  # conversion factor heat pump, depending on temperature
    # hp_cop = hp_cop.squeeze()
    # low_heat_demand = (heat_demand.loc[heat_demand.heat_type == "lowT"]).drop(columns=["heat_type"]).squeeze()
    # percent_hp_RTE = 0.4
    # percent_resistive_RTE = 0.6
    # eta_resistive = 0.9
    #
    # electricity_heat_demand = percent_hp_RTE * (low_heat_demand / hp_cop) + percent_resistive_RTE * (
    #         low_heat_demand / eta_resistive)  # GWh
    #
    # # electricity_heat_demand = electricity_heat_demand + (33.7*1e3 - electricity_heat_demand.sum())/8760  # rescale to fit RTE total demand simply on residential heating
    #
    # demand_elec_RTE = pd.read_csv("inputs/demand2050_RTE.csv", index_col=0, header=None)
    # demand_elec_RTE = demand_elec_RTE.squeeze()
    # # demand_elec_RTE_no_residential_heating = demand_elec_RTE - electricity_heat_demand
    # # print(electricity_heat_demand.sum())
    #

    ########### Second method: directly using profiles from Doudard (2018)
    demand_elec_RTE = pd.read_csv("inputs/demand2050_RTE.csv", index_col=0, header=None)
    demand_elec_RTE = demand_elec_RTE.squeeze()
    daily_profile = [1 / 24 for i in range(24)]
    monthly_profile = [0.24, 0.18, 0.15, 0.05, 0.01, 0, 0, 0, 0, 0.03, 0.12, 0.22]
    days_by_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    percentage_hourly_residential_profile_doudard = []
    for i in range(len(days_by_month)):
        month_profile = []
        for j in range(days_by_month[i]):
            month_profile = month_profile + daily_profile
        rescale_month_profile = [p / (days_by_month[i]) * monthly_profile[i] for p in month_profile]
        percentage_hourly_residential_profile_doudard = percentage_hourly_residential_profile_doudard + rescale_month_profile  # we rescale to the number of hours in the month, and to the percentage of the month

    percentage_hourly_residential_profile_doudard = pd.Series(percentage_hourly_residential_profile_doudard)
    total_residential_heating_doudard = 33 * 1e3
    hourly_residential_heating_doudard = total_residential_heating_doudard * percentage_hourly_residential_profile_doudard
    demand_elec_RTE_no_residential_heating = demand_elec_RTE - hourly_residential_heating_doudard

    #
    # plt.plot(np.arange(0, 8760, 1), demand_elec_RTE_no_residential_heating)
    # plt.show()
    #
    # demand_elec_RTE_no_residential_heating.to_csv("inputs/demand2050_RTE_no_residential_heating_doudard.csv", header=False)
    # hourly_residential_heating_doudard.to_csv("inputs/hourly_residential_heating_RTE2050_doudard.csv", header=False)
    # percentage_hourly_residential_profile_doudard.to_csv("inputs/percentage_hourly_residential_heating_profile_doudard.csv", header=False)

    ############## Third method: using hourly profile from RTE
    tot = 0.035 + 0.039 + 0.041 + 0.042 + 0.046 + 0.05 + 0.055 + 0.058 + 0.053 + 0.049 + 0.045 + 0.041 + 0.037 + 0.034 + 0.03 + 0.033 + 0.037 + 0.042 + 0.046 + 0.041 + 0.037 + 0.034 + 0.033
    daily_profile = [0.035, 0.039, 0.041, 0.042, 0.046, 0.05, 0.055, 0.058, 0.053, 0.049, 0.045, 0.041, 0.037, 0.034, 0.03, 0.033, 0.037, 0.042, 0.046, 0.041, 0.037, 0.034, 0.033, 1-tot]
    monthly_profile = [0.24, 0.18, 0.15, 0.05, 0.01, 0, 0, 0, 0, 0.03, 0.12, 0.22]  # comes from Doudard et al
    days_by_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    percentage_hourly_residential_profile_RTE = []
    for i in range(len(days_by_month)):
        month_profile = []
        for j in range(days_by_month[i]):
            month_profile = month_profile + daily_profile
        rescale_month_profile = [p / (days_by_month[i]) * monthly_profile[i] for p in month_profile]
        percentage_hourly_residential_profile_RTE = percentage_hourly_residential_profile_RTE + rescale_month_profile  # we rescale to the number of hours in the month, and to the percentage of the month

    percentage_hourly_residential_profile_RTE = pd.Series(percentage_hourly_residential_profile_RTE)

    total_residential_heating_RTE = 33 * 1e3
    hourly_residential_heating_RTE = total_residential_heating_RTE * percentage_hourly_residential_profile_RTE

    adjust_demand = (595 * 1e3 - 580 * 1e3) / 8760
    demand_elec_RTE_noP2G = demand_elec_RTE + adjust_demand  # we adjust demand profile to obtain the correct total amount of demand

    demand_elec_RTE_no_residential_heating = demand_elec_RTE_noP2G - hourly_residential_heating_RTE
    # demand_elec_RTE_no_residential_heating.to_csv("inputs/demand2050_RTE_no_residential_heating.csv", header=False)

    # hourly_residential_heating_RTE.to_csv("inputs/hourly_residential_heating_RTE2050_RTE.csv", header=False)
    #
    # percentage_hourly_residential_profile_RTE.to_csv("inputs/hourly_profiles/percentage_hourly_residential_heating_profile_RTE.csv",
    #                                              header=False)


    ############ Fourth method: hourly profile from Valentin  ############
    L = [1850, 1750, 1800, 1850, 1900, 1950, 2050, 2120, 2250, 2100, 2000, 1850, 1700, 1550, 1600, 1650, 1800, 2000,
         2100, 2150, 2200, 2150, 2100, 2000]  # profil issu de Valentin
    daily_profile_valentin = [e / sum(L) for e in L]
    monthly_profile = [0.24, 0.18, 0.15, 0.05, 0.01, 0, 0, 0, 0, 0.03, 0.12, 0.22]  # comes from Doudard et al
    days_by_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    percentage_hourly_residential_profile_valentin = []
    for i in range(len(days_by_month)):
        month_profile = []
        for j in range(days_by_month[i]):
            month_profile = month_profile + daily_profile_valentin
        rescale_month_profile = [p / (days_by_month[i]) * monthly_profile[i] for p in month_profile]
        percentage_hourly_residential_profile_valentin = percentage_hourly_residential_profile_valentin + rescale_month_profile  # we rescale to the number of hours in the month, and to the percentage of the month

    percentage_hourly_residential_profile_valentin = pd.Series(percentage_hourly_residential_profile_valentin)
    # percentage_hourly_residential_profile_valentin.to_csv("inputs/hourly_profiles/percentage_hourly_residential_heating_profile_valentin.csv",
    #                                              header=False)

    total_residential_heating_RTE = 33 * 1e3
    hourly_residential_heating_valentin = total_residential_heating_RTE * percentage_hourly_residential_profile_valentin

    #####  Comparison profile RTE et Valentin
    hourly_residential_heating_valentin = hourly_residential_heating_valentin.reset_index().rename(columns={'index': 'hour', 0: 'demand'})
    hourly_residential_heating_valentin["date"] = hourly_residential_heating_valentin.apply(
        lambda row: datetime.datetime(2006, 1, 1, 0) + datetime.timedelta(hours=row["hour"]),
        axis=1)

    hourly_residential_heating_RTE = hourly_residential_heating_RTE.reset_index().rename(columns={'index': 'hour', 0: 'demand'})
    hourly_residential_heating_RTE["date"] = hourly_residential_heating_RTE.apply(
        lambda row: datetime.datetime(2006, 1, 1, 0) + datetime.timedelta(hours=row["hour"]),
        axis=1)

    def plot_subset_residential_demand(demand1, demand2, date_start, date_end):
        """ Allows comparison of the two methodologies."""
        demand1 = demand1.set_index("date").copy()
        demand1 = demand1.loc[date_start: date_end, :]

        demand2 = demand2.set_index("date").copy()
        demand2 = demand2.loc[date_start: date_end, :]
        fig, ax = plt.subplots(1, 1)

        demand1[['demand']].squeeze().plot(ax=ax, style='-', c='red', label='1')
        demand2[['demand']].squeeze().plot(ax=ax, style='-', c='blue', label='2')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.set_title("Hourly production and demand (GW)", loc='left', color='black')
        ax.set_xlabel('')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

        plt.show()


    date_start, date_end = datetime.datetime(2006, 1, 1, 0, 0), datetime.datetime(2006, 1, 7, 0, 0)
    plot_subset_residential_demand(hourly_residential_heating_RTE, hourly_residential_heating_valentin, date_start, date_end)


    ####### Fifth method: hourly profile from BDEW, Zeyen #######
    heat_load = get_pandas("eoles/inputs/hourly_profiles/heat_load_profile.csv", lambda x: pd.read_csv(x))
    daily_profile_BDEW = heat_load["residential_space_weekday_percentage_BDEW"].tolist()
    monthly_profile = [0.24, 0.18, 0.15, 0.05, 0.01, 0, 0, 0, 0, 0.03, 0.12, 0.22]  # comes from Doudard et al
    days_by_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    percentage_hourly_residential_profile_BDEW = []
    for i in range(len(days_by_month)):
        month_profile = []
        for j in range(days_by_month[i]):
            month_profile = month_profile + daily_profile_BDEW
        rescale_month_profile = [p / (days_by_month[i]) * monthly_profile[i] for p in month_profile]
        percentage_hourly_residential_profile_BDEW = percentage_hourly_residential_profile_BDEW + rescale_month_profile  # we rescale to the number of hours in the month, and to the percentage of the month

    percentage_hourly_residential_profile_BDEW = pd.Series(percentage_hourly_residential_profile_BDEW)
    percentage_hourly_residential_profile_BDEW.to_csv("inputs/hourly_profiles/percentage_hourly_residential_heating_profile_BDEW.csv",
                                                 header=False)


    # ### Test
    # # The goal is to test a caricatural profile where all heating demand is concentrated on 4 hours, to see if with this profile, we get worse results.
    # daily_profile = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/4, 1/4, 1/4, 1/4, 0, 0, 0]
    # monthly_profile = [0.24, 0.18, 0.15, 0.05, 0.01, 0, 0, 0, 0, 0.03, 0.12, 0.22]
    # days_by_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    #
    # percentage_hourly_residential_profile_test = []
    # for i in range(len(days_by_month)):
    #     month_profile = []
    #     for j in range(days_by_month[i]):
    #         month_profile = month_profile + daily_profile
    #     rescale_month_profile = [p / (days_by_month[i]) * monthly_profile[i] for p in month_profile]
    #     percentage_hourly_residential_profile_test = percentage_hourly_residential_profile_test + rescale_month_profile  # we rescale to the number of hours in the month, and to the percentage of the month
    #
    # percentage_hourly_residential_profile_test = pd.Series(percentage_hourly_residential_profile_test)
    #
    # total_residential_heating_RTE = 33 * 1e3
    # hourly_residential_heating_RTE = total_residential_heating_RTE * percentage_hourly_residential_profile_test
    #
    # hourly_residential_heating_RTE.to_csv("inputs/hourly_residential_heating_RTE2050_test.csv", header=False)
    #
    # percentage_hourly_residential_profile_test.to_csv("inputs/percentage_hourly_residential_heating_profile_test.csv",
    #                                              header=False)