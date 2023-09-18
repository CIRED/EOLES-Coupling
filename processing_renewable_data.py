from sklearn.linear_model import LinearRegression
import pandas as pd
import os
from datetime import datetime


############# Extract required years  #############
############# Extract required years for renewable data #########
vre_profiles_initial = pd.read_csv('eoles/inputs/hourly_profiles/vre_profiles_2000-2019.csv', index_col=0).reset_index()
vre_profiles_initial.columns = ["tec", "hour", "capacity_factor"]
vre_profiles_initial = vre_profiles_initial.loc[vre_profiles_initial.tec.isin(['offshore_f', 'river'])]
vre_profiles_quentin = pd.read_csv('eoles/inputs/hourly_profiles/vre_profiles_all_years.csv', index_col=0).reset_index()
vre_profiles_quentin.columns = ["tec", "hour", "capacity_factor"]
# vre_profiles_quentin = vre_profiles_quentin.replace('pv_g_EW', 'pv_g')  # we rename to have the good technology name
# vre_profiles_quentin = vre_profiles_quentin.replace('offshore_f', 'offshore_g')  # we rename to have the good technology name
vre_profiles_quentin = vre_profiles_quentin.loc[vre_profiles_quentin.tec.isin(['offshore_g', 'onshore', 'pv_c', 'pv_g'])]

list_year = [2006, 2004]
vre_profiles_subset = pd.DataFrame()

for (i, y) in enumerate(list_year):
    n = y - 2000
    initial_hour = 8760*n
    final_hour = 8760*n+8759
    vre_profiles_initial_y = vre_profiles_initial.loc[(vre_profiles_initial.hour >= initial_hour) & (vre_profiles_initial.hour <= final_hour)]
    vre_profiles_quentin_y = vre_profiles_quentin.loc[(vre_profiles_quentin.hour >= initial_hour) & (vre_profiles_quentin.hour <= final_hour)]
    vre_profiles_y = pd.concat([vre_profiles_initial_y, vre_profiles_quentin_y], axis=0)
    vre_profiles_y['hour'] = vre_profiles_y['hour'].apply(lambda x: x + 8760*(i-n))
    vre_profiles_subset = pd.concat([vre_profiles_subset, vre_profiles_y], axis=0)
vre_profiles_subset = vre_profiles_subset.sort_values(by=["tec", "hour"])
vre_profiles_subset = vre_profiles_subset.set_index("tec")

vre_profiles_subset.to_csv('eoles/inputs/hourly_profiles/vre_profiles_2006_2004.csv', header=False)

############## Estimate run of river values for different years  ######################

river_historic = pd.DataFrame(columns=['river_capacity_factor'])
capacity = 9.66  # capacity of river + ecluse

for year in [2013, 2014, 2015, 2016, 2017, 2018]:
    tmp = pd.read_csv(os.path.join(f"eoles/inputs/hourly_profiles/eCO2mix_RTE_Annuel-Definitif_{year}.csv"), sep=";", low_memory=False)
    tmp = tmp[['Date', 'Heures', 'Hydraulique - Fil de l?eau + ÈclusÈe']]
    tmp = tmp.dropna()
    tmp = tmp.replace('ND', tmp.iloc[1,2])
    tmp = tmp.loc[tmp.Date != f'29/02/{year}']  # get rid of bissextile
    tmp = tmp.rename(columns={'Hydraulique - Fil de l?eau + ÈclusÈe': "river"})
    tmp = tmp.astype(dtype={"river": float})
    tmp["date"] = tmp.apply(lambda row: datetime.strptime(row['Date'] + ' ' + row['Heures'][:2], '%d/%m/%Y %H'), axis=1)

    tmp_hour = tmp[::2]
    tmp_half_hour = tmp[1::2]
    tmp2 = tmp_hour.merge(tmp_half_hour, on="date")
    tmp2["river"] = (tmp2["river_x"] + tmp2["river_y"])/2
    tmp2["river_capacity_factor"] = tmp2["river"] / 1000 / capacity
    tmp2 = tmp2[["river_capacity_factor"]]
    river_historic = pd.concat([river_historic, tmp2], ignore_index=True)

# river_historic.to_csv(os.path.join("eoles/inputs/hourly_profiles/river_2013-2018.csv"))


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