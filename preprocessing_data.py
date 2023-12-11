from sklearn.linear_model import LinearRegression
import pandas as pd
import os
import datetime
from matplotlib import pyplot as plt


####### Demand profile #########

# Process csv file to get the demand profile
# First column is the type of vehicule, second is the hour, last is the demand value in MW
demand_ev = pd.read_csv('eoles/inputs/demand_data_other/demand_transport2050.csv', index_col=0, header=None).reset_index().rename(columns={0: 'vehicule', 1: 'hour', 2: 'demand'})
demand_rte = pd.read_csv("eoles/inputs/demand/demand2050_RTE.csv", index_col=0, header=None).squeeze("columns")

adjust_demand = (530 * 1e3 - 580 * 1e3) / 8760  # 580TWh is the total of the profile we use as basis for electricity hourly demand (from RTE), c'est bien vérifié
demand_rte_rescaled_1 = demand_rte + adjust_demand
demand_rte_rescaled_2 = demand_rte * (530 / 580)

plt.plot(demand_rte_rescaled_1[0:150], c='red')
plt.plot(demand_rte_rescaled_2[0:150], c='blue')
# plt.plot(demand_rte[0:1000], c='green')
plt.show()

# plot the demand profile for vehicule = 'light'

demand_ev_light = demand_ev.loc[demand_ev.vehicule == 'light']
demand_ev_light = demand_ev_light.drop(columns=['vehicule'])
demand_ev_light = demand_ev_light.set_index('hour')

demand_ev_heavy = demand_ev.loc[demand_ev.vehicule == 'heavy']
demand_ev_heavy = demand_ev_heavy.drop(columns=['vehicule'])
demand_ev_heavy = demand_ev_heavy.set_index('hour')

demand_ev_bus = demand_ev.loc[demand_ev.vehicule == 'bus']
demand_ev_bus = demand_ev_bus.drop(columns=['vehicule'])
demand_ev_bus = demand_ev_bus.set_index('hour')
# plot only for a subset of hours
# demand_ev_light.loc[0:100].plot()
# plt.show()



# ####### Processing historical data #########
# file_eco2mix = ['eoles/inputs/hourly_profiles/eCO2mix_RTE_Annuel-Definitif_2013.csv',
#                 'eoles/inputs/hourly_profiles/eCO2mix_RTE_Annuel-Definitif_2014.csv',
#                 'eoles/inputs/hourly_profiles/eCO2mix_RTE_Annuel-Definitif_2015.csv',
#                 'eoles/inputs/hourly_profiles/eCO2mix_RTE_Annuel-Definitif_2016.csv',
#                 'eoles/inputs/hourly_profiles/eCO2mix_RTE_Annuel-Definitif_2017.csv',
#                 'eoles/inputs/hourly_profiles/eCO2mix_RTE_Annuel-Definitif_2018.csv']
# for file in file_eco2mix:
#     eco2mix = pd.read_csv(file, delimiter=';')
#     eco2mix = eco2mix.rename(columns={
#         'PÈrimËtre': 'Perimetre',
#         'NuclÈaire': 'Nucleaire'
#     })
#     eco2mix['capa_nuc'] = (63.1 - 1.6) * 1e3
#     eco2mix['capacity_factor'] = eco2mix['Nucleaire'] / eco2mix['capa_nuc']
#     # print(eco2mix.capacity_factor.describe(percentiles=[.1, .25, .5, .75, .9]))  # we display the distribution of capacity factors for nuclear
#
#     demand = eco2mix[::4]['Consommation'].reset_index(drop=True)
#     demand = demand.drop(demand.index[-1]) * 1e-3
#     print(demand.sum())
#
#     eco2mix['Emissions CO2'] = eco2mix['Taux de Co2'] * eco2mix['Consommation'] *1e-6  # in MtCO2
#
# eco2mix = pd.read_csv('eoles/inputs/hourly_profiles/eCO2mix_RTE_Annuel-Definitif_2016.csv', delimiter=';')
# demand2016 = eco2mix[::4]['Consommation'].reset_index(drop=True)
# demand2016 = demand2016.drop(demand2016.index[-1]) * 1e-3
# demand2016 = demand2016.drop(demand2016.index[1416:1416+24]).reset_index(drop=True)  # we drop the last day for 2016 as it is a bissextile year
# demand2016.to_csv('eoles/inputs/demand_data_other/demand2016_RTE.csv', header=None)
#
# eco2mix = pd.read_csv('eoles/inputs/hourly_profiles/eCO2mix_RTE_Annuel-Definitif_2017.csv', delimiter=';')
# eco2mix = eco2mix.rename(columns={
#     'PÈrimËtre': 'Perimetre',
#     'NuclÈaire': 'Nucleaire'
# })
# eco2mix = eco2mix[::4].reset_index(drop=True)
# eco2mix = eco2mix.drop(eco2mix.index[-1])
# eco2mix = eco2mix.reset_index().rename(columns={'index': 'hour'})
# eco2mix["date"] = eco2mix.apply(lambda row: datetime.datetime(2017, 1, 1, 0) + datetime.timedelta(hours=row["hour"]),
#     axis=1)
#
# prod_nuc = eco2mix.Nucleaire.sum()
# capacity_factor = prod_nuc / (61.6*8760*1000)
# eco2mix.Nucleaire.describe()
# print(capacity_factor)
#
# def plot_subset_nuclear(eco2mix, date_start, date_end):
#     eco2mix = eco2mix.set_index("date").copy()
#     eco2mix = eco2mix.loc[date_start: date_end, :]
#
#     fig, ax = plt.subplots(1, 1)
#
#     eco2mix[['Nucleaire']].squeeze().plot(ax=ax, style='-', c='red')
#
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(True)
#     ax.spines['left'].set_visible(True)
#     ax.set_title("Hourly production and demand (GW)", loc='left', color='black')
#     ax.set_xlabel('')
#
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
#
#     plt.show()
#
# date_start, date_end = datetime.datetime(2017, 1, 1, 0, 0), datetime.datetime(2017, 1, 30, 0, 0)
# plot_subset_nuclear(eco2mix, date_start, date_end)
#
# ##### Explore demand profile
# demandRTE = pd.read_csv('eoles/inputs/demand/demand2050_RTE.csv', index_col=0, header=None).reset_index().rename(columns={0: 'hour', 1: 'demand 2050'})
# demandRTE["date"] = demandRTE.apply(lambda row: datetime.datetime(2006, 1, 1, 0) + datetime.timedelta(hours=row["hour"]),
#     axis=1)
# adjust_demand = (475 * 1e3 - 580 * 1e3) / 8760
# demandRTE['demand 2050'] = demandRTE['demand 2050'] + adjust_demand
#
# demand2017 = pd.read_csv('eoles/inputs/demand_data_other/demand2017_RTE.csv', index_col=0, header=None).reset_index().rename(columns={0: 'hour', 1: 'demand'})
# demand2017["date"] = demand2017.apply(lambda row: datetime.datetime(2006, 1, 1, 0) + datetime.timedelta(hours=row["hour"]),
#     axis=1)
#
# demand2018 = pd.read_csv('eoles/inputs/demand_data_other/demand2018_RTE.csv', index_col=0, header=None).reset_index().rename(columns={0: 'hour', 1: 'demand'})
# demand2018["date"] = demand2018.apply(lambda row: datetime.datetime(2006, 1, 1, 0) + datetime.timedelta(hours=row["hour"]),
#     axis=1)
#
# demand2016 = pd.read_csv('eoles/inputs/demand_data_other/demand2016_RTE.csv', index_col=0, header=None).reset_index().rename(columns={0: 'hour', 1: 'demand'})
# demand2016["date"] = demand2016.apply(lambda row: datetime.datetime(2016, 1, 1, 0) + datetime.timedelta(hours=row["hour"]),
#     axis=1)
#
# def plot_subset_demand(demand1, demand2, date_start, date_end):
#     demand1 = demand1.set_index("date").copy()
#     demand1 = demand1.loc[date_start: date_end, :]
#
#     demand2 = demand2.set_index("date").copy()
#     demand2 = demand2.loc[date_start: date_end, :]
#     fig, ax = plt.subplots(1, 1)
#
#     demand1[['demand 2050']].squeeze().plot(ax=ax, style='-', c='red')
#     demand2[['demand']].squeeze().plot(ax=ax, style='-', c='blue')
#
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(True)
#     ax.spines['left'].set_visible(True)
#     ax.set_title("Hourly production and demand (GW)", loc='left', color='black')
#     ax.set_xlabel('')
#
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
#
#     plt.show()
#
#
#
# date_start, date_end = datetime.datetime(2006, 1, 1, 0, 0), datetime.datetime(2006, 1, 7, 0, 0)
# plot_subset_demand(demandRTE, demand2017, date_start, date_end)
#
# ####### Test to compare the values of the different technologies
#
# load_factors = pd.read_csv('eoles/inputs/hourly_profiles/vre_profiles_2018.csv', index_col=[0, 1], header=None)
# load_factors.loc['offshore_f'].describe()
#
# load_factors = pd.read_csv('eoles/inputs/hourly_profiles/vre_profiles_2006.csv', index_col=[0, 1], header=None)
# load_factors.loc['offshore_f'].describe()
#
# ############# Extract required years  #############
# ############# Extract required years for renewable data #########
vre_profiles_initial = pd.read_csv('eoles/inputs/hourly_profiles/vre_profiles_2000-2019.csv', index_col=0).reset_index()
vre_profiles_initial.columns = ["tec", "hour", "capacity_factor"]
vre_profiles_initial = vre_profiles_initial.loc[vre_profiles_initial.tec.isin(['offshore_f', 'river'])]
vre_profiles_quentin = pd.read_csv('eoles/inputs/hourly_profiles/vre_profiles_all_years.csv', index_col=0).reset_index()
vre_profiles_quentin.columns = ["tec", "hour", "capacity_factor"]
# vre_profiles_quentin = vre_profiles_quentin.replace('pv_g_EW', 'pv_g')  # we rename to have the good technology name
# vre_profiles_quentin = vre_profiles_quentin.replace('offshore_f', 'offshore_g')  # we rename to have the good technology name
vre_profiles_quentin = vre_profiles_quentin.loc[vre_profiles_quentin.tec.isin(['offshore_g', 'onshore', 'pv_c', 'pv_g'])]

lake_inflows = pd.read_csv('eoles/inputs/hourly_profiles/lake_2000-2019.csv', index_col=0, header=None).reset_index()
lake_inflows.columns = ["month", "capacity_factor"]
#
list_year = [2012]
vre_profiles_subset = pd.DataFrame()
lake_inflows_subset = pd.DataFrame()

for (i, y) in enumerate(list_year):
    n = y - 2000
    initial_hour, final_hour = 8760*n, 8760*n+8759
    initial_month, final_month = 12*n+1, 12*n+12
    # Load factors
    vre_profiles_initial_y = vre_profiles_initial.loc[(vre_profiles_initial.hour >= initial_hour) & (vre_profiles_initial.hour <= final_hour)]
    vre_profiles_quentin_y = vre_profiles_quentin.loc[(vre_profiles_quentin.hour >= initial_hour) & (vre_profiles_quentin.hour <= final_hour)]
    vre_profiles_y = pd.concat([vre_profiles_initial_y, vre_profiles_quentin_y], axis=0)
    vre_profiles_y['hour'] = vre_profiles_y['hour'].apply(lambda x: x + 8760*(i-n))
    vre_profiles_subset = pd.concat([vre_profiles_subset, vre_profiles_y], axis=0)
    # Lake inflows
    lake_inflows_y = lake_inflows.loc[(lake_inflows.month >= initial_month) & (lake_inflows.month <= final_month)]
    lake_inflows_y['month'] = lake_inflows_y['month'].apply(lambda x: x + 12*(i-n))
    lake_inflows_subset = pd.concat([lake_inflows_subset, lake_inflows_y], axis=0)

vre_profiles_subset = vre_profiles_subset.sort_values(by=["tec", "hour"])
vre_profiles_subset = vre_profiles_subset.set_index("tec")

lake_inflows_subset = lake_inflows_subset.set_index('month')
#
vre_profiles_subset.to_csv('eoles/inputs/hourly_profiles/vre_profiles_2012.csv', header=False)
lake_inflows_subset.to_csv('eoles/inputs/hourly_profiles/lake_2012.csv', header=False)
#
# ############## Estimate run of river values for different years  ######################
#
# river_historic = pd.DataFrame(columns=['river_capacity_factor'])
# capacity = 9.66  # capacity of river + ecluse
#
# for year in [2013, 2014, 2015, 2016, 2017, 2018]:
#     tmp = pd.read_csv(os.path.join(f"eoles/inputs/hourly_profiles/eCO2mix_RTE_Annuel-Definitif_{year}.csv"), sep=";", low_memory=False)
#     tmp = tmp[['Date', 'Heures', 'Hydraulique - Fil de l?eau + ÈclusÈe']]
#     tmp = tmp.dropna()
#     tmp = tmp.replace('ND', tmp.iloc[1,2])
#     tmp = tmp.loc[tmp.Date != f'29/02/{year}']  # get rid of bissextile
#     tmp = tmp.rename(columns={'Hydraulique - Fil de l?eau + ÈclusÈe': "river"})
#     tmp = tmp.astype(dtype={"river": float})
#     tmp["date"] = tmp.apply(lambda row: datetime.datetime.strptime(row['Date'] + ' ' + row['Heures'][:2], '%d/%m/%Y %H'), axis=1)
#
#     tmp_hour = tmp[::2]
#     tmp_half_hour = tmp[1::2]
#     tmp2 = tmp_hour.merge(tmp_half_hour, on="date")
#     tmp2["river"] = (tmp2["river_x"] + tmp2["river_y"])/2
#     tmp2["river_capacity_factor"] = tmp2["river"] / 1000 / capacity
#     tmp2 = tmp2[["river_capacity_factor"]]
#     river_historic = pd.concat([river_historic, tmp2], ignore_index=True)
#
# river_historic.to_csv(os.path.join("eoles/inputs/hourly_profiles/river_2013-2018.csv"))
#

# ################## Projection of transport and distribution costs #############
#
# transport_and_distrib = pd.DataFrame(
#     data={'M0': [5, 2, 2, 0, 0, 19], 'M1': [5, 2, 2, 0, 0, 19], 'M23': [5, 3, 2, 0, 0, 17],
#           'N1': [5, 2, 2, 0, 0, 17], 'N2': [5, 1, 1, 0, 0, 16], 'N03': [5, 1, 1, 0, 0, 16]},
#     index=["Transport - réseau existant et renouvellement", "Transport - raccordement éolien en mer",
#            "Transport - adaptation", "Transport - interconnexions", "Transport - autre", "Distribution"])
# transport_and_distrib = transport_and_distrib.T
# transport_and_distrib["transport"] = transport_and_distrib["Transport - réseau existant et renouvellement"] + \
#                                      transport_and_distrib["Transport - adaptation"] + transport_and_distrib[
#                                          "Transport - interconnexions"] + transport_and_distrib["Transport - autre"]
# transport_and_distrib["transport_and_distribution"] = transport_and_distrib["transport"] + transport_and_distrib[
#     "Distribution"]
#
# prod_renewable = pd.DataFrame(
#     data={'M0': [208, 74, 62], 'M1': [214, 59, 45], 'M23': [125, 72, 60], 'N1': [118, 58, 45],
#           'N2': [90, 52, 36], 'N03': [70, 43, 22]}, index=["solar", "onshore_wind", "offshore_wind"])
# prod_renewable = prod_renewable.T
#
# y = transport_and_distrib[["transport_and_distribution"]].squeeze().to_numpy()
# X = prod_renewable[["solar", "onshore_wind"]].to_numpy()
#
# reg = LinearRegression().fit(X, y)
# print(f"Coefficients for predicting transport and distribution: {reg.coef_}")
# print(f"Intercept for predicting transport and distribution: {reg.intercept_}")
#
# y_distrib = transport_and_distrib[["Distribution"]].squeeze().to_numpy()
# reg_distrib = LinearRegression().fit(X, y_distrib)
# print(f"Coefficients for predicting distribution: {reg_distrib.coef_}")
# print(f"Intercept for predicting transport and distribution: {reg.intercept_}")
#
# y_offshore = transport_and_distrib[["Transport - raccordement éolien en mer"]].squeeze().to_numpy()
# X_offshore = prod_renewable[["offshore_wind"]].to_numpy()
# reg_offshore = LinearRegression().fit(X_offshore, y_offshore)
# print(f"Coefficients for predicting offshore costs: {reg_offshore.coef_}")
# print(f"Intercept for predicting transport and distribution: {reg.intercept_}")