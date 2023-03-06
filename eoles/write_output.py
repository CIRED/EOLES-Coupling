import pandas as pd
import matplotlib.pyplot as plt
import os
from os import listdir
import seaborn as sns
import numpy as np
from pickle import load
import datetime
from PIL import Image

from eoles.inputs.resources import resources_data
from project.utils import save_fig

sns.set_theme(context="talk", style="white")

COLORS_SCENARIOS = {}  # a ajouter des couleurs par simulation
STYLES = ['-', '--', ':', "-.",  '*-', 's-', 'o-', '^-', 's-', 'o-', '^-', '*-']

DICT_TRANSFORM_LEGEND = {
    "Annualized electricity system costs": "electricity",
    "Annualized investment heater costs": "heater",
    "Annualized investment insulation costs": "insulation",
    "Annualized health costs": "health",
    "Annualized total costs HC excluded": "total costs",
    "Annualized total costs": "total costs",
    "Investment electricity costs": "electricity",
    "Functionment costs": "energy operational",
    "Investment heater costs": "heater",
    "Investment insulation costs": "insulation",
    "Health costs": "health",
    "Total costs HC excluded": "total costs",
    "Total costs": "total costs",
    "Consumption saving insulation (TWh/year)": "insulation",
    "Consumption saving heater (TWh/year)": "heater",
    "Consumption saving insulation (TWh)": "insulation",
    "Consumption saving heater (TWh)": "heater",
    "Investment heater (Billion euro)": "investment heater",
    "Investment insulation (Billion euro)": "investment insulation",
    "Subsidies heater (Billion euro)": "subsidies heat pump",
    "Subsidies insulation (Billion euro)": "subsidies insulation",
    "offshore_f": "offshore floating wind",
    "offshore_g": "offshore ground wind",
    "onshore": "onshore wind",
    "pv_g": "pv ground",
    "pv_c": "pv large roof",
    "nuclear": "nuclear",
    "river": "river",
    "lake": "lake",
    "methanization": "methanization",
    "pyrogazification": "pyrogazification",
    "natural_gas": "natural gas"
}


def plot_comparison(output1, output2, name1, name2):
    # capacity_df_1 = output1["Capacities (GW)"].T.rename(index={2025: name1})
    resirf_subsidies_1 = output1["Subsidies (%)"].rename(index={2025: name1})
    resirf_costs_df_1 = output1["ResIRF costs (Billion euro)"].rename(index={2025: name1})
    resirf_consumption_df_1 = output1["ResIRF consumption (TWh)"].rename(index={2025: name1})
    resirf_replacement_heater_1 = output1["ResIRF replacement heater (Thousand)"].T.rename(index={2025: name1})
    resirf_stock_heater_1 = output1["ResIRF stock heater (Thousand)"].T.rename(index={2029: name1})
    annualized_system_costs_1 = output1["Annualized system costs (Billion euro / year)"].rename(index={2025: name1})

    # capacity_df_2 = output2["Capacities (GW)"]
    resirf_subsidies_2 = output2["Subsidies (%)"].rename(index={2025: name2})
    resirf_costs_df_2 = output2["ResIRF costs (Billion euro)"].rename(index={2025: name2})
    resirf_consumption_df_2 = output2["ResIRF consumption (TWh)"].rename(index={2025: name2})
    resirf_replacement_heater_2 = output2["ResIRF replacement heater (Thousand)"].T.rename(index={2025: name2})
    resirf_stock_heater_2 = output2["ResIRF stock heater (Thousand)"].T.rename(index={2029: name2})
    annualized_system_costs_2 = output2["Annualized system costs (Billion euro / year)"].rename(index={2025: name2})

    resirf_subsidies = pd.concat([resirf_subsidies_1, resirf_subsidies_2], axis=0)
    resirf_costs_df = pd.concat([resirf_costs_df_1, resirf_costs_df_2], axis=0)
    resirf_consumption_df = pd.concat([resirf_consumption_df_1, resirf_consumption_df_2], axis=0)
    resirf_replacement_heater = pd.concat([resirf_replacement_heater_1, resirf_replacement_heater_2], axis=0)
    resirf_stock_heater = pd.concat([resirf_stock_heater_1, resirf_stock_heater_2], axis=0)
    annualized_system_costs = pd.concat([annualized_system_costs_1, annualized_system_costs_2], axis=0)

    make_line_plot(resirf_subsidies, y_label="Subsidies (%)", save=None,
                   format_y=lambda y, _: '{:.0f}'.format(y), index_int=False, str=True)

    make_line_plot(resirf_costs_df, y_label="Costs (Billion euro)", subset=["Heater", "Insulation", "Health cost"],
                   save=None, format_y=lambda y, _: '{:.0f}'.format(y), index_int=False, str=True)

    make_line_plot(resirf_consumption_df, subset=["Electricity", "Natural gas", "Oil fuel", "Wood fuel"],
                   y_label="Heating consumption (TWh)", colors=resources_data["colors_resirf"],
                   save=None, format_y=lambda y, _: '{:.0f}'.format(y), index_int=False, str=True)

    make_line_plot(resirf_consumption_df, subset=['Saving heater', "Saving insulation"],
                   y_label="Savings (TWh)",
                   save=None, format_y=lambda y, _: '{:.0f}'.format(y), index_int=False, str=True)

    make_line_plot(annualized_system_costs, y_label="Annualized system costs", subset=["Annualized electricity system costs"],
                   save=None, index_int=False, str=True)

    make_line_plot(resirf_replacement_heater, y_label="Replacement heater (Thousand households)",
                   save=None, format_y=lambda y, _: '{:.0f}'.format(y), index_int=False, str=True)

    make_line_plot(resirf_stock_heater, y_label="Stock heater (Thousand households)",
                   save=None, format_y=lambda y, _: '{:.0f}'.format(y), index_int=False, str=True)


def comparison_simulations(dict_output:dict, ref, health=False, save_path=None):
    annualized_system_costs_df = pd.DataFrame(dtype=float)
    total_system_costs_df = pd.DataFrame(dtype=float)
    complete_system_costs_2050_df = pd.DataFrame(dtype=float)
    consumption_savings_tot_df = pd.DataFrame(dtype=float)
    peak_electricity_load_dict = {}
    consumption_saving_evolution_dict = {}
    emissions_dict = {}
    annualized_system_costs_dict = {}
    subsidies_insulation_dict = {}
    subsidies_heater_dict = {}
    capacities_dict = {}

    for path, name_config in zip(dict_output.values(), [n for n in dict_output.keys()]):
        with open(os.path.join(path, 'coupling_results.pkl'), "rb") as file:
            output = load(file)
            annualized_system_costs = output["Annualized system costs (Billion euro / year)"]
            annualized_system_costs_dict[name_config] = annualized_system_costs.sum(axis=0)
            annualized_system_costs = annualized_system_costs.sum(axis=0).to_frame().rename(columns={0: name_config})
            annualized_system_costs_df = pd.concat([annualized_system_costs_df, annualized_system_costs], axis=1)

            annualized_new_investment_df = output["Annualized new investments (1e9€/yr)"]
            annualized_new_energy_capacity_df = output["Annualized costs new energy capacity (1e9€/yr)"]
            functionment_costs_df = output["System functionment (1e9€/yr)"]
            total_system_costs = process_total_costs(annualized_new_investment_df, annualized_new_energy_capacity_df, functionment_costs_df)
            total_system_costs = total_system_costs.to_frame().rename(columns={0: name_config})
            total_system_costs_df = pd.concat([total_system_costs_df, total_system_costs], axis=1)

            annualized_new_investment_df = output["Annualized new investments (1e9€/yr)"]
            annualized_new_energy_capacity_df = output["Annualized costs new energy capacity (1e9€/yr)"]
            functionment_costs_df = output["System functionment (1e9€/yr)"]
            complete_system_costs_2050 = process_complete_system_cost_2050(annualized_new_investment_df, annualized_new_energy_capacity_df, functionment_costs_df)
            complete_system_costs_2050 = complete_system_costs_2050.to_frame().rename(columns={0: name_config})
            complete_system_costs_2050_df = pd.concat([complete_system_costs_2050_df, complete_system_costs_2050], axis=1)

            try:
                consumption_savings = output["ResIRF consumption savings (TWh/year)"]
            except:
                consumption_savings = output["ResIRF consumption savings (TWh)"]
                consumption_savings = consumption_savings.rename(
                    columns={"Consumption saving heater (TWh)": "Consumption saving heater (TWh/year)",
                             "Consumption saving insulation (TWh)": "Consumption saving insulation (TWh/year)"})
            consumption_savings_tot = consumption_savings.sum(axis=0).to_frame().rename(columns={0: name_config})
            consumption_savings_tot_df = pd.concat([consumption_savings_tot_df, consumption_savings_tot], axis=1)

            consumption_savings_evolution = consumption_savings.reset_index().rename(columns={'index': 'year'})
            consumption_savings_evolution["period"] = consumption_savings_evolution.apply(lambda row: (row["year"] - 2025) // 5, axis=1)
            consumption_savings_evolution = consumption_savings_evolution.groupby("period").agg(
                {"year": np.min, "Consumption saving heater (TWh/year)": np.sum,
                 "Consumption saving insulation (TWh/year)": np.sum}).set_index("year")
            consumption_savings_evolution.index.name = None
            consumption_saving_evolution_dict[name_config] = consumption_savings_evolution
            #
            peak_electricity_load_info_df = output["Peak electricity load"]
            peak_electricity_load_info_df = peak_electricity_load_info_df[["peak_electricity_load", "year"]].groupby(["year"]).mean().squeeze()
            peak_electricity_load_dict[name_config] = peak_electricity_load_info_df

            emissions = output["Emissions (MtCO2)"]
            emissions_dict[name_config] = emissions.squeeze()

            subsidies = output["Subsidies (%)"] * 100
            dataframe_subsidy_list = [subsidies]
            for i in range(4):
                tmp = subsidies.copy()
                tmp.index += i + 1
                dataframe_subsidy_list.append(tmp)
            dataframe_subsidy = pd.concat(dataframe_subsidy_list, axis=0).sort_index(ascending=True)

            subsidies_insulation_dict[name_config] = dataframe_subsidy[["Insulation"]].squeeze()
            subsidies_heater_dict[name_config] = dataframe_subsidy[["Heater"]].squeeze()

            capacities_df = output["Capacities (GW)"].T
            selected_capacities = ["offshore_f", "offshore_g", "pv_g", "pv_c", "battery1", "battery4"]
            capacities_df = capacities_df[selected_capacities]
            capacities_df["offshore"] = capacities_df["offshore_f"] + capacities_df["offshore_g"]
            capacities_df["pv"] = capacities_df["pv_g"] + capacities_df["pv_c"]
            capacities_df["battery"] = capacities_df["battery1"] + capacities_df["battery4"]
            capacities_dict[name_config] = capacities_df[["offshore", "pv", "battery"]]

    # # Total annualized system costs
    # subset_annualized_costs = ["Annualized electricity system costs", "Annualized investment heater costs",
    #                            "Annualized investment insulation costs", "Annualized health costs"]
    # if save_path is None:
    #     save_path_plot = None
    # else:
    #     save_path_plot = os.path.join(save_path, "total_annualized_system_costs.png")
    # make_stacked_bar_plot(annualized_system_costs_df.T, subset=subset_annualized_costs, y_label="Total annualized system costs (Md€ / year)",
    #                       colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y), index_int=False,
    #                       rotation=90, dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot)
    #
    # annualized_system_costs_df = annualized_system_costs_df.T
    # annualized_system_costs_df["Annualized total costs HC excluded"] = annualized_system_costs_df["Annualized total costs"] - \
    #                                                          annualized_system_costs_df["Annualized health costs"]
    # annualized_system_costs_df = annualized_system_costs_df.T
    #
    # for col in annualized_system_costs_df.columns:
    #     if col != ref:
    #         annualized_system_costs_df[col] = annualized_system_costs_df[col] - annualized_system_costs_df[ref]
    # if health:
    #     subset_annualized_costs = ["Annualized electricity system costs", "Annualized investment heater costs",
    #                                "Annualized investment insulation costs", "Annualized health costs"]
    # else:
    #     subset_annualized_costs = ["Annualized electricity system costs", "Annualized investment heater costs",
    #                                "Annualized investment insulation costs"]
    # if len(annualized_system_costs_df.columns) >= 3:  # ie, at least two scenarios to compare to the ref
    #     if save_path is None:
    #         save_path_plot = None
    #     else:
    #         save_path_plot = os.path.join(save_path, "difference_total_annualized_system_costs.png")
    #     if health:
    #         make_stacked_investment_plot(df=annualized_system_costs_df.drop(columns=[ref]).T,
    #                                      y_label="Difference of total annualized system costs over 2025-2050 (Md€/yr)",
    #                                      subset=subset_annualized_costs,
    #                                      scatter=annualized_system_costs_df.drop(columns=[ref]).T[
    #                                          ["Annualized total costs"]].squeeze(),
    #                                      save=save_path_plot, colors=resources_data["colors_eoles"],
    #                                      format_y=lambda y, _: '{:.0f}'.format(y), rotation=90,
    #                                      dict_legend=DICT_TRANSFORM_LEGEND)
    #     else:
    #         make_stacked_investment_plot(df=annualized_system_costs_df.drop(columns=[ref]).T, y_label="Difference of total annualized system costs over 2025-2050 (Md€/yr)",
    #                                      subset=subset_annualized_costs,
    #                                      scatter=annualized_system_costs_df.drop(columns=[ref]).T[
    #                                          ["Annualized total costs HC excluded"]].squeeze(),
    #                                      save=save_path_plot, colors=resources_data["colors_eoles"],
    #                                      format_y=lambda y, _: '{:.0f}'.format(y), rotation=90,
    #                                      dict_legend=DICT_TRANSFORM_LEGEND)

    # Total system costs
    subset_annualized_costs = ["Investment electricity costs", "Investment heater costs",
                                   "Investment insulation costs", "Functionment costs", "Health costs"]
    if save_path is None:
        save_path_plot = None
    else:
        save_path_plot = os.path.join(save_path, "total_system_costs.png")
    make_stacked_bar_plot(total_system_costs_df.T, subset=subset_annualized_costs, y_label="Total system costs (Md€)",
                          colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.0f}'.format(y), index_int=False,
                          rotation=90, dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot)

    # Total system costs
    total_system_costs_df = total_system_costs_df.T
    total_system_costs_df["Total costs HC excluded"] = total_system_costs_df["Total costs"] - \
                                                             total_system_costs_df["Health costs"]
    total_system_costs_df = total_system_costs_df.T

    for col in total_system_costs_df.columns:
        if col != ref:
            total_system_costs_df[col] = total_system_costs_df[col] - total_system_costs_df[ref]
    if health:
        subset_costs = ["Investment electricity costs", "Investment heater costs",
                                   "Investment insulation costs", "Functionment costs", "Health costs"]
    else:
        subset_costs = ["Investment electricity costs", "Investment heater costs",
                        "Investment insulation costs", "Functionment costs"]
    if len(total_system_costs_df.columns) >= 3:  # ie, at least two scenarios to compare to the ref
        if save_path is None:
            save_path_plot = None
        else:
            save_path_plot = os.path.join(save_path, "difference_total_system_costs.png")
        if health:
            make_stacked_investment_plot(df=total_system_costs_df.drop(columns=[ref]).T,
                                         y_label="Difference of total system costs over 2025-2050 (Billion €)",
                                         subset=subset_costs,
                                         scatter=total_system_costs_df.drop(columns=[ref]).T[
                                             ["Total costs"]].squeeze(),
                                         save=save_path_plot, colors=resources_data["colors_eoles"],
                                         format_y=lambda y, _: '{:.0f}'.format(y), rotation=90,
                                         dict_legend=DICT_TRANSFORM_LEGEND)
        else:
            make_stacked_investment_plot(df=total_system_costs_df.drop(columns=[ref]).T, y_label="Difference of total system costs over 2025-2050 (Billion €)",
                                         subset=subset_costs,
                                         scatter=total_system_costs_df.drop(columns=[ref]).T[
                                             ["Total costs HC excluded"]].squeeze(),
                                         save=save_path_plot, colors=resources_data["colors_eoles"],
                                         format_y=lambda y, _: '{:.0f}'.format(y), rotation=90,
                                         dict_legend=DICT_TRANSFORM_LEGEND)
    else:
        make_stacked_investment_plot(df=total_system_costs_df.drop(columns=[ref]).T,
                                     y_label="Difference of total system costs over 2025-2050 (Billion €)",
                                     subset=subset_costs,
                                     scatter=total_system_costs_df.drop(columns=[ref]).T["Total costs"],
                                     save=None, colors=resources_data["colors_eoles"],
                                     format_y=lambda y, _: '{:.0f}'.format(y), rotation=90,
                                     dict_legend=DICT_TRANSFORM_LEGEND)

    # Complete system costs in 2050
    subset_complete_costs = ["Investment electricity costs", "Investment heater costs",
                                   "Investment insulation costs", "Functionment costs", "Health costs"]
    if save_path is None:
        save_path_plot = None
    else:
        save_path_plot = os.path.join(save_path, "complete_system_costs_2050.png")
    make_stacked_bar_plot(complete_system_costs_2050_df.T, subset=subset_complete_costs, y_label="Complete system costs in 2050 (Md€/year)",
                          colors=resources_data["colors_eoles"], format_y=lambda y, _: '{:.01f}'.format(y), index_int=False,
                          rotation=90, dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot)

    for col in complete_system_costs_2050_df.columns:
        if col != ref:
            complete_system_costs_2050_df[col] = complete_system_costs_2050_df[col] - complete_system_costs_2050_df[ref]

    if len(total_system_costs_df.columns) >= 3:  # ie, at least two scenarios to compare to the ref
        if save_path is None:
            save_path_plot = None
        else:
            save_path_plot = os.path.join(save_path, "difference_complete_system_costs_2050.png")
        make_stacked_investment_plot(df=complete_system_costs_2050_df.drop(columns=[ref]).T,
                                     y_label="Difference of complete system costs in 2050 (Billion € / year)",
                                     subset=subset_complete_costs,
                                     scatter=complete_system_costs_2050_df.drop(columns=[ref]).T[
                                         ["Total costs"]].squeeze(),
                                     save=save_path_plot, colors=resources_data["colors_eoles"],
                                     format_y=lambda y, _: '{:.1f}'.format(y), rotation=90,
                                     dict_legend=DICT_TRANSFORM_LEGEND)
    else:
        make_stacked_investment_plot(df=complete_system_costs_2050_df.drop(columns=[ref]).T,
                                     y_label="Difference of complete system costs in 2050 (Billion € / year)",
                                     subset=subset_complete_costs,
                                     scatter=complete_system_costs_2050_df.drop(columns=[ref]).T["Total costs"],
                                     save=None, colors=resources_data["colors_eoles"],
                                     format_y=lambda y, _: '{:.1f}'.format(y), rotation=90,
                                     dict_legend=DICT_TRANSFORM_LEGEND)

    # Total consumption savings
    if save_path is None:
        save_path_plot = None
    else:
        save_path_plot = os.path.join(save_path, "consumption_savings.png")
    make_stacked_bar_plot(consumption_savings_tot_df.T, y_label="Total consumption savings (TWh)",
                          colors=resources_data["colors_resirf"], format_y=lambda y, _: '{:.0f}'.format(y), index_int=False,
                          rotation=90, dict_legend=DICT_TRANSFORM_LEGEND, save=save_path_plot)

    # Evolution of peak load
    if save_path is None:
        save_path_plot = None
    else:
        save_path_plot = os.path.join(save_path, "electricity_peak_load.png")
    make_line_plots(peak_electricity_load_dict, y_label="Electricity peak load (GW)", format_y=lambda y, _: '{:.0f}'.format(y),
                    index_int=True, save=save_path_plot)

    # Evolution of emissions
    if save_path is None:
        save_path_plot = None
    else:
        save_path_plot = os.path.join(save_path, "CO2_emissions.png")
    make_line_plots(emissions_dict, y_label="Emissions (MtCO2)", format_y=lambda y, _: '{:.0f}'.format(y),
                    index_int=True, save=save_path_plot, y_min=0)

    # Evolution of insulation subsidies
    if save_path is None:
        save_path_plot = None
    else:
        save_path_plot = os.path.join(save_path, "subsidies_insulation.png")
    make_line_plots(subsidies_insulation_dict, y_label="Subsidies insulation (%)", format_y=lambda y, _: '{:.0f}'.format(y),
                    index_int=True, save=save_path_plot, rotation=45, x_ticks=dataframe_subsidy.index[::2])

    # Evolution of heater subsidies
    if save_path is None:
        save_path_plot = None
    else:
        save_path_plot = os.path.join(save_path, "subsidies_heater.png")
    make_line_plots(subsidies_heater_dict, y_label="Subsidies heater (%)", format_y=lambda y, _: '{:.2f}'.format(y),
                    index_int=True, save=save_path_plot, rotation=45, x_ticks=dataframe_subsidy.index[::2])

    if len(dict_output.keys()) <= 3:
        # Evolution of consumption savings
        if save_path is None:
            save_path_plot = None
        else:
            save_path_plot = os.path.join(save_path, "evolution_consumption_savings.png")
        stacked_bars(consumption_saving_evolution_dict, y_label="Consumption savings (TWh)", format_y=lambda y, _: '{:.0f}'.format(y),
                     colors=None, x_ticks=None, index_int=True, save=save_path_plot, rotation=0, n=len(consumption_saving_evolution_dict.keys()),
                     dict_legend=DICT_TRANSFORM_LEGEND)

    # Evolution of electricity capacities
    if save_path is None:
        save_path_plot = None
    else:
        save_path_plot = os.path.join(save_path, "electricity_capacities.png")
    make_line_plots(capacities_dict, y_label="Capacities (GW)", format_y=lambda y, _: '{:.0f}'.format(y),
                    index_int=True, colors=resources_data["colors_eoles"], multiple_legend=True, save=save_path_plot)

    return annualized_system_costs_df, total_system_costs_df, consumption_savings_tot_df, complete_system_costs_2050_df


def save_summary_pdf(path):
    """Saves a summary of files as pdf"""
    images_to_save = [os.path.join(path, "plots", "img", "consumption_energy.png"),
                      os.path.join(path, "plots", "img", "consumption_heater.png"),
                      os.path.join(path, "plots", "img", "investment.png"),
                      os.path.join(path, "plots", "img", "renovation_decision_maker.png"),
                      os.path.join(path, "plots", "img", "replacement_insulation.png"),
                      os.path.join(path, "plots", "img", "retrofit_measures.png"),
                      os.path.join(path, "plots", "img", "stock_heater.png"),
                      os.path.join(path, "plots", "img", "stock_performance.png"),
                      os.path.join(path, "plots", "img", "switch_heater.png"),
                      os.path.join(path, "plots", "emissions.png"),
                      os.path.join(path, "plots", "prices.png"),
                      os.path.join(path, "plots", "primary_generation.png"),
                      os.path.join(path, "plots", "resirf_subsidies.png"),
                      ]
    images = [Image.open(img) for img in images_to_save]
    new_images = []
    for png in images:
        png.load()
        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
        new_images.append(background)

    pdf_path = os.path.join(path, "summary_pdf.pdf")

    new_images[0].save(
        pdf_path, "PDF", resolution=100.0, save_all=True, append_images=new_images[1:]
    )


def save_subsidies_again(dict_output, save_path):
    for path, name_config in zip(dict_output.values(), [n for n in dict_output.keys()]):
        with open(os.path.join(path, 'coupling_results.pkl'), "rb") as file:
            output = load(file)
            subsidies = output["Subsidies (%)"]
            subsidies = output["Subsidies (%)"]
            dataframe_subsidy_list = [subsidies]
            for i in range(4):
                tmp = subsidies.copy()
                tmp.index += i + 1
                dataframe_subsidy_list.append(tmp)
            dataframe_subsidy = pd.concat(dataframe_subsidy_list, axis=0).sort_index(ascending=True)
            new_subsidies_2020_2024 = pd.concat(
                [subsidies.iloc[0].to_frame().T.rename(index={2025: 2020 + i}) for i in range(5)], axis=0)
            dataframe_subsidy = pd.concat([new_subsidies_2020_2024, dataframe_subsidy], axis=0)
            dataframe_subsidy = pd.concat([dataframe_subsidy, dataframe_subsidy.iloc[-1].to_frame().T.rename(index={2049:2050})], axis=0)
            # subsidies = pd.concat([subsidies, subsidies.iloc[0].to_frame().T.rename(index={2025: 2020})], axis=0).sort_index()
            dataframe_subsidy["Heater"].to_csv(os.path.join(save_path, f"subsidies_heater_{name_config}.csv"), header=None)
            dataframe_subsidy["Insulation"].to_csv(os.path.join(save_path, f"subsidies_insulation_{name_config}.csv"), header=None)


def process_total_costs(annualized_new_investment_df, annualized_new_energy_capacity_df, functionment_costs_df):
    """
    Calculates total system (new) costs over the considered time period. This adds investment annuity for each year when the investment was present.
    Remark: we only include new costs, not costs of historical capacities.
    :param annualized_new_investment_df: pd.DataFrame
        Annualized costs of new investment done during the considered year.
    :param annualized_new_energy_capacity_df: pd.DataFrame
        Annualized costs of new energy capacity investment done during the considered year.
    :param functionment_costs_df: pd.DataFrame
        Functionment cost of the system for one year.
    :return:
    """
    annualized_new_investment_df_copy = annualized_new_investment_df.copy()
    annualized_new_energy_capacity_df_copy = annualized_new_energy_capacity_df.copy()
    functionment_costs_df_copy = functionment_costs_df.copy()
    dict_count = {2030: 5*5, 2035: 4*5, 2040: 3*5, 2045: 2*5, 2050: 5}
    for col in annualized_new_investment_df_copy.columns:  # attention à vérifier que les colonnes sont des int
        annualized_new_investment_df_copy[col] = annualized_new_investment_df_copy[col] * dict_count[col]
        annualized_new_energy_capacity_df_copy[col] = annualized_new_energy_capacity_df_copy[col] * dict_count[col]
    functionment_costs_df_copy = functionment_costs_df_copy * 5  # we count each functionment costs 5 times

    elec_inv = annualized_new_investment_df_copy.drop(index=["investment_heater", "investment_insulation"]).sum().sum() + annualized_new_energy_capacity_df_copy.sum().sum()
    heater_inv = annualized_new_investment_df_copy.T[["investment_heater"]].sum().sum()
    insulation_inv = annualized_new_investment_df_copy.T[["investment_insulation"]].sum().sum()
    functionment_cost = functionment_costs_df_copy.drop(index=["health_costs"]).sum().sum()
    health_costs = functionment_costs_df_copy.T[["health_costs"]].sum().sum()
    total_costs = elec_inv + heater_inv + insulation_inv + functionment_cost + health_costs
    total_system_costs = pd.Series(index=["Investment electricity costs", "Investment heater costs", "Investment insulation costs", "Functionment costs", "Health costs", "Total costs"],
                                   data=[elec_inv, heater_inv, insulation_inv, functionment_cost, health_costs, total_costs])
    return total_system_costs


def process_evolution_annualized_energy_system_cost(annualized_new_investment_df, annualized_new_energy_capacity_df, functionment_costs_df,
                                             historical_capacity_df, historical_energy_capacity_df, transport_distribution_costs):
    """Process the evolution of complete energy annualized system costs, in 1e9 €/yr. This includes in particular
    historical costs, and transport and distribution costs. Be careful: cannot be compared directly to RTE outputs, since this includes
    as well the functionment of wood and oil boilers. Otherwise, we do look like RTE. DOES NOT INCLUDE HEATER AND INSULATION INVESTMENT."""
    total_cost = annualized_new_investment_df.drop(index=["investment_heater", "investment_insulation"])  # we are only interested in the energy system cost
    total_cost = total_cost.add(annualized_new_energy_capacity_df, fill_value=0)  # we add the value of investments
    for i in range(1, annualized_new_investment_df.shape[1]):  # we estimate cumulated costs from new investments which are still active in following years
        total_cost[total_cost.columns[i]] = total_cost[total_cost.columns[i-1]] + total_cost[total_cost.columns[i]]

    total_cost = total_cost.add(functionment_costs_df.drop(index=["health_costs"]), fill_value=0)  # add functionment cost for each year, and not interested in health costs
    total_cost = total_cost.add(historical_capacity_df, fill_value=0)  # add historical capacity cost present during the considered year
    total_cost = total_cost.add(historical_energy_capacity_df, fill_value=0)  # add historical energy capacity cost present during the considered year
    total_cost = total_cost.add(transport_distribution_costs, fill_value=0)  # add transport and distribution costs
    total_cost = total_cost.sum(axis=0)
    return total_cost.T.squeeze()


def process_complete_system_cost_2050(annualized_new_investment_df, annualized_new_energy_capacity_df, functionment_costs_df):
    annualized_new_investment_df_copy = annualized_new_investment_df.copy()
    annualized_new_energy_capacity_df_copy = annualized_new_energy_capacity_df.copy()
    functionment_costs_df_copy = functionment_costs_df.copy()
    investment_costs = annualized_new_investment_df_copy  # we are only interested in the energy system cost
    investment_costs = investment_costs.add(annualized_new_energy_capacity_df_copy, fill_value=0)  # we add the value of investments
    for i in range(1, annualized_new_investment_df_copy.shape[1]):  # we estimate cumulated costs from new investments which are still active in following years
        investment_costs[investment_costs.columns[i]] = investment_costs[investment_costs.columns[i-1]] + investment_costs[investment_costs.columns[i]]

    elec_inv = investment_costs.drop(index=["investment_heater", "investment_insulation"]).sum(axis=0)
    heater_inv = investment_costs.T[["investment_heater"]].squeeze()
    insulation_inv = investment_costs.T[["investment_insulation"]].squeeze()
    functionment_cost = functionment_costs_df_copy.drop(index=["health_costs"]).sum(axis=0)
    health_costs = functionment_costs_df_copy.T[["health_costs"]].squeeze()

    total_cost = investment_costs.add(functionment_costs_df_copy, fill_value=0)  # add functionment cost for each year, and not interested in health costs
    total_cost = total_cost.sum(axis=0)
    return pd.Series(data=[elec_inv.loc[2050], heater_inv.loc[2050], insulation_inv.loc[2050], functionment_cost.loc[2050], health_costs.loc[2050], total_cost.loc[2050]],
                     index=["Investment electricity costs", "Investment heater costs", "Investment insulation costs", "Functionment costs", "Health costs", "Total costs"])


def plot_simulation(output, save_path):
    capacity_df = output["Capacities (GW)"]
    generation_df = output["Generation (TWh)"]
    primary_generation_df = output["Primary generation (TWh)"]
    conversion_generation_df = output["Conversion generation (TWh)"]
    prices_df = output["Prices (€/MWh)"]
    resirf_subsidies = output["Subsidies (%)"]
    resirf_costs_df = output["ResIRF costs (Billion euro)"]
    resirf_costs_eff_df = output["ResIRF costs eff (euro/kWh)"]
    resirf_consumption_df = output["ResIRF consumption (TWh)"]
    try:
        resirf_consumption_saving_df = output["ResIRF consumption savings (TWh/year)"]
    except:  # old name
        resirf_consumption_saving_df = output["ResIRF consumption savings (TWh)"]
        resirf_consumption_saving_df = resirf_consumption_saving_df.rename(columns={"Consumption saving heater (TWh)": "Consumption saving heater (TWh/year)",
                                                     "Consumption saving insulation (TWh)": "Consumption saving insulation (TWh/year)"})
    resirf_replacement_heater = output["ResIRF replacement heater (Thousand)"]
    resirf_stock_heater = output["ResIRF stock heater (Thousand)"]
    annualized_system_costs = output["Annualized system costs (Billion euro / year)"]
    peak_electricity_load_df = output["Peak electricity load"]
    peak_heat_load_df = output["Peak heat load"]
    emissions = output["Emissions (MtCO2)"]
    annualized_new_investment_df = output["Annualized new investments (1e9€/yr)"]
    annualized_new_energy_capacity_df = output["Annualized costs new energy capacity (1e9€/yr)"]
    functionment_costs_df = output["System functionment (1e9€/yr)"]

    # Plot capacities
    # # Electricity generation capacities
    elec_generation = ["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "nuclear", "ocgt",
                       "ccgt"]
    make_line_plot(capacity_df.T, subset=elec_generation, y_label="Capacity power (GWh)",
                   colors=resources_data["colors_eoles"],
                   save=os.path.join(save_path, "capacities_electricity_generation.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    # # CH4 generation capacities
    CH4_generation = ["methanization", "pyrogazification", "methanation"]
    make_line_plot(capacity_df.T, subset=CH4_generation, y_label="Capacity CH4 (GWh)",
                   colors=resources_data["colors_eoles"],
                   save=os.path.join(save_path, "capacities_CH4_generation.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    # Plot generation
    # # Primary generation
    make_area_plot(primary_generation_df.T, y_label="Primary generation (TWh)", colors=resources_data["colors_eoles"],
                   save=os.path.join(save_path, "primary_generation.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    # # CH4 generation
    # TODO: ajouter une selection
    CH4_generation = ["methanation", "methanization", "pyrogazification", "natural_gas"]
    make_area_plot(generation_df.T, subset=CH4_generation, y_label="CH4 generation (TWh)", colors=resources_data["colors_eoles"],
                   save=os.path.join(save_path, "primary_CH4_generation.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    # # CH4 génération with demand
    df = pd.concat([generation_df.T[CH4_generation], conversion_generation_df.T[["ocgt", "ccgt"]],
                    resirf_consumption_df[["Natural gas"]].rename(columns={'Natural gas': 'Gas for heating'})], axis=1)
    df["ocgt"] = - df["ocgt"]
    df["ccgt"] = - df["ccgt"]
    df['Gas for heating'] = - df['Gas for heating']
    make_area_plot(df, y_label="Gas generation and conversion (TWh)", colors=resources_data["colors_eoles"],
                   save=os.path.join(save_path, "CH4_generation_and_conversion.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    # # Power génération with demand
    df = pd.concat([generation_df.T[elec_generation], conversion_generation_df.T[["electrolysis", "methanation"]],
                    resirf_consumption_df[["Electricity"]].rename(columns={'Electricity': 'Electricity for heating'})], axis=1)
    df["electrolysis"] = - df["electrolysis"]
    df["methanation"] = - df["methanation"]
    df['Electricity for heating'] = - df['Electricity for heating']
    make_area_plot(df, y_label="Power generation and conversion (TWh)", colors=resources_data["colors_eoles"],
                   save=os.path.join(save_path, "power_generation_and_conversion.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    # Plot LCOE and price
    make_line_plot(prices_df, y_label="Prices (€/MWh)", colors=resources_data["colors_eoles"],
                   save=os.path.join(save_path, "prices.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    # Plot subsidies ResIRF
    dataframe_subsidy_list = [resirf_subsidies]
    for i in range(4):
        tmp = resirf_subsidies.copy()
        tmp.index += i + 1
        dataframe_subsidy_list.append(tmp)
    dataframe_subsidy = pd.concat(dataframe_subsidy_list, axis=0).sort_index(ascending=True)
    dataframe_subsidy *= 100  # we write in percentage
    make_line_plot(dataframe_subsidy, y_label="Subsidies (%)", save=os.path.join(save_path, "resirf_subsidies.png"), format_y=lambda y, _: '{:.0f}'.format(y),
                   rotation=45, x_ticks=dataframe_subsidy.index[::2])

    # make_line_plot(dataframe_subsidy, y_label="Subsidies (%)", save=os.path.join(save_path, "resirf_subsidies.png"))

    # Plots no longer needed (use of ResIRF plots)
    # # Plot consumption ResIRF
    # make_area_plot(resirf_consumption_df, subset=["Electricity", "Natural gas", "Oil fuel", "Wood fuel"],
    #                y_label="Heating consumption (TWh)", colors=resources_data["colors_resirf"],
    #                save=os.path.join(save_path, "resirf_consumption.png"), format_y=lambda y, _: '{:.0f}'.format(y))
    #
    # try:  # TODO: ajout temporaire pour gérer un problème de savings négatif
    #     # Plot savings ResIRF
    #     make_area_plot(resirf_consumption_saving_df, y_label="Consumption savings (TWh)",
    #                    save=os.path.join(save_path, "resirf_savings.png"), format_y=lambda y, _: '{:.0f}'.format(y),
    #                    rotation=45, x_ticks=resirf_consumption_saving_df.index[::2])
    #
    #     # Unique plot consumption + savings
    #     resirf_consumption_saving_df = resirf_consumption_saving_df.reset_index().rename(columns={'index': 'year'})
    #     resirf_consumption_saving_df["Consumption saving heater cumulated (TWh)"] = resirf_consumption_saving_df["Consumption saving heater (TWh/year)"].cumsum()
    #     resirf_consumption_saving_df["Consumption saving insulation cumulated (TWh)"] = resirf_consumption_saving_df[
    #         "Consumption saving insulation (TWh/year)"].cumsum()
    #     resirf_consumption_saving_df = resirf_consumption_saving_df.loc[resirf_consumption_saving_df["year"] % 5 == 4]  # we only keep rows like 2029, 2034, etc...
    #     resirf_consumption_saving_df["year"] = resirf_consumption_saving_df["year"] + 1  # we modify the year
    #     resirf_consumption_saving_df = resirf_consumption_saving_df.set_index("year")
    #     # resirf_consumption_saving_df["period"] = resirf_consumption_saving_df.apply(lambda row: (row["year"] - 2025) // 5, axis=1)
    #     # resirf_consumption_saving_df = resirf_consumption_saving_df.groupby("period").agg(
    #     #     {"year": np.min, "Consumption saving heater (TWh)": np.sum, "Consumption saving insulation (TWh)": np.sum}).set_index("year")
    #
    #     resirf_consumption_total_df = pd.concat([resirf_consumption_df, resirf_consumption_saving_df[["Consumption saving heater cumulated (TWh)", "Consumption saving insulation cumulated (TWh)"]]], axis=1)
    #     make_area_plot(resirf_consumption_total_df, y_label="Consumption and savings (TWh)", colors=resources_data["colors_resirf"],
    #                    save=os.path.join(save_path, "resirf_consumption_savings.png"), format_y=lambda y, _: '{:.0f}'.format(y))
    # except:
    #     pass
    #
    # # Plot costs ResIRF
    # make_line_plot(resirf_costs_df, y_label="Costs (Billion euro)", save=os.path.join(save_path, "resirf_costs.png"),
    #                colors=resources_data["colors_eoles"],
    #                format_y=lambda y, _: '{:.0f}'.format(y), rotation=45, x_ticks=resirf_costs_df.index[::2])
    #
    # try:
    #     if "ResIRF costs eff (euro/kWh)" in output.keys():
    #         resirf_costs_eff_df = output["ResIRF costs eff (euro/kWh)"]
    #         make_line_plot(resirf_costs_eff_df, y_label="Costs per saving (euro/kWh)", save=os.path.join(save_path, "resirf_costs_eff.png"),
    #                        format_y=lambda y, _: '{:.2f}'.format(y), rotation=45, x_ticks=resirf_costs_eff_df.index[::2])
    # except:
    #     pass
    #
    # # Plot stock and replacement ResIRF
    # make_area_plot(resirf_replacement_heater, y_label="Replacement heater (Thousand households)",
    #                save=os.path.join(save_path, "resirf_replacement.png"), format_y=lambda y, _: '{:.0f}'.format(y),
    #                rotation=45, x_ticks=resirf_replacement_heater.index[::2])
    #
    # resirf_stock_heater = resirf_stock_heater.T
    # resirf_stock_heater["Heat pump"] = resirf_stock_heater["Stock Heat pump (Million)"]
    # resirf_stock_heater["Electric heating"] = resirf_stock_heater["Stock Direct electric (Million)"]
    # resirf_stock_heater["Natural gas"] = resirf_stock_heater["Stock Natural gas (Million)"]
    # resirf_stock_heater["Oil fuel"] = resirf_stock_heater[ "Stock Oil fuel (Million)"]
    # resirf_stock_heater["Wood fuel"] = resirf_stock_heater["Stock Wood fuel (Million)"]
    # resirf_stock_heater.index += 1  # we increase the index to get the correct years
    # make_area_plot(resirf_stock_heater[["Heat pump", "Electric heating", "Natural gas", "Oil fuel", "Wood fuel"]], y_label="Stock heater (Million)",
    #                save=os.path.join(save_path, "resirf_stock.png"), format_y=lambda y, _: '{:.0f}'.format(y), colors=resources_data["colors_eoles"])

    # Plot annualized system costs
    make_line_plot(annualized_system_costs, y_label="Annualized system costs (Md€ / year)",
                   save=os.path.join(save_path, "annualized_system_costs.png"), format_y=lambda y, _: '{:.0f}'.format(y))

    subset_annualized_costs = ["Annualized electricity system costs", "Annualized investment heater costs", "Annualized investment insulation costs", "Annualized health costs"]
    make_area_plot(annualized_system_costs, subset=subset_annualized_costs, y_label="Annualized system costs (Md€ / year)",
                   save=os.path.join(save_path, "annualized_system_costs_area.png"), format_y=lambda y, _: '{:.0f}'.format(y),
                   colors=resources_data["colors_eoles"])

    # Peak load
    peak_electricity_load = peak_electricity_load_df[["peak_electricity_load", "year"]].groupby(["year"]).mean()
    make_line_plot(peak_electricity_load, y_label="Electricity peak load (GW)", save=os.path.join(save_path, "peak_load_electricity.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    peak_heat_load = peak_heat_load_df[["heat_elec", "year"]].groupby(["year"]).mean()
    make_line_plot(peak_heat_load, y_label="Heat peak load (GW)", save=os.path.join(save_path, "peak_load_heat.png"),
                   format_y=lambda y, _: '{:.2f}'.format(y))

    # Emissions
    make_line_plot(emissions, y_label="Emissions (MtCO2)", save=os.path.join(save_path, "emissions.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y), y_min=0)


def plot_investment_trajectory(resirf_costs_df, save=None):
    if save is None:
        fig, ax = plt.subplots(1, 1)
        save_path = None
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
        save_path = os.path.join(save, "investment_subsidies_heater.png")
    resirf_costs_df[["Investment heater (Billion euro)"]].plot.line(ax=ax, color=resources_data["colors_eoles"])
    resirf_costs_df[["Subsidies heater (Billion euro)"]].plot.line(ax=ax, color=resources_data["colors_eoles"],
                                                                   style="--")

    ax = format_ax(ax, title="Investments heater (Billion euro)", x_ticks=resirf_costs_df.index[::2],
                   format_y=lambda y, _: '{:.0f}'.format(y), rotation=45)
    format_legend(ax, dict_legend=DICT_TRANSFORM_LEGEND)
    save_fig(fig, save=save_path)

    if save is None:
        fig, ax = plt.subplots(1, 1)
        save_path = None
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
        save_path = os.path.join(save, "investment_subsidies_insulation.png")
    resirf_costs_df[["Investment insulation (Billion euro)"]].plot.line(ax=ax, color=resources_data["colors_eoles"])
    resirf_costs_df[["Subsidies insulation (Billion euro)"]].plot.line(ax=ax, color=resources_data["colors_eoles"],
                                                                       style="--")
    ax = format_ax(ax, title="Investments insulation (Billion euro)", x_ticks=resirf_costs_df.index[::2],
                   format_y=lambda y, _: '{:.0f}'.format(y), rotation=45)
    format_legend(ax, dict_legend=DICT_TRANSFORM_LEGEND)
    save_fig(fig, save=save_path)


def plot_typical_week(hourly_generation, date_start, date_end, climate=2006, save=None,
                          colors=None, format_y=lambda y, _: '{:.0f}'.format(y), rotation=90):
    hourly_generation_subset = hourly_generation.copy()
    hourly_generation_subset["date"] = hourly_generation_subset.apply(lambda row: datetime.datetime(climate, 1, 1, 0) + datetime.timedelta(hours=row["hour"]),
                            axis=1)
    hourly_generation_subset = hourly_generation_subset.set_index("date")

    hourly_generation_subset = hourly_generation_subset.loc[date_start: date_end, :]  # select week of interest

    hourly_generation_subset["pv"] = hourly_generation_subset["pv_g"] + hourly_generation_subset["pv_c"]
    hourly_generation_subset["wind"] = hourly_generation_subset["onshore"] + hourly_generation_subset["offshore_f"] + hourly_generation_subset["offshore_g"]
    hourly_generation_subset["hydro"] = hourly_generation_subset["river"] + hourly_generation_subset["lake"]
    hourly_generation_subset["battery_in"] = - hourly_generation_subset["battery1_in"] - hourly_generation_subset["battery4_in"]
    hourly_generation_subset["battery_discharge"] = hourly_generation_subset["battery1"] + hourly_generation_subset["battery4"]
    hourly_generation_subset["phs_in"] = - hourly_generation_subset["phs_in"]
    hourly_generation_subset["electrolysis"] = - hourly_generation_subset["electrolysis"]
    hourly_generation_subset["methanation"] = - hourly_generation_subset["methanation"]
    hourly_generation_subset["peaking_plants"] = hourly_generation_subset["ocgt"] + hourly_generation_subset["ccgt"] + hourly_generation_subset["h2_ccgt"]
    prod = hourly_generation_subset[["nuclear", "wind", "pv", "hydro", "battery_in", "battery_discharge", "phs", "phs_in", "peaking_plants", "electrolysis", "methanation"]]
    elec_demand = hourly_generation_subset[["elec_demand"]].squeeze()

    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))

    prod.plot.area(color=resources_data["colors_eoles"], ax=ax, linewidth=0)
    elec_demand.plot(ax=ax, style='-', c='red')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_title("Hourly production and demand")
    # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    # plt.gcf().autofmt_xdate()
    # ax = format_ax(ax, title="Hourly demand and production (GWh)", format_y=lambda y, _: '{:.0f}'.format(y),
    #                       rotation=45, x_ticks=prod.index[::12])
    format_legend(ax)
    plt.axhline(y=0)

    save_fig(fig, save=save)


def format_legend(ax, dict_legend=None):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    if dict_legend is not None:
        current_labels = ax.get_legend_handles_labels()[1]
        new_labels = [dict_legend[e] for e in current_labels]
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), labels=new_labels, frameon=False)
    else:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)


def format_legend_multiple(ax, d, n_style, n_color):
    """Format legend when we want multiple legends for color and linestyle."""
    dummy_lines_color = []
    labels_color = []
    labels_style = []
    dummy_lines_style = []
    for i, l in enumerate(ax.get_lines()):
        if i % n_color == 1:
            dummy_lines_style.append(l)
            labels_style.append(list(d.keys())[i // n_color])
        if i // n_color == 0:
            dummy_lines_color.append(l)
            labels_color.append(l._label)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    legend_style = plt.legend(dummy_lines_style, labels_style, loc='upper left', bbox_to_anchor=(1, 0.5), frameon=False)
    legend_color = plt.legend(dummy_lines_color, labels_color, loc='lower left', bbox_to_anchor=(1, 0.5), frameon=False)
    ax.add_artist(legend_style)
    ax.add_artist(legend_color)


def format_ax(ax: plt.Axes, title=None, y_label=None, x_label=None, x_ticks=None, format_y=lambda y, _: y,
              rotation=None, y_min=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
    if x_ticks is not None:
        ax.set_xticks(ticks=x_ticks, labels=x_ticks)
    if rotation is not None:
        ax.set_xticklabels(ax.get_xticks(), rotation=rotation)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_min is not None:
        ax.set_ylim(ymin=y_min)

    if title is not None:
        ax.set_title(title)

    return ax


def format_ax_string(ax: plt.Axes, title=None, y_label=None, x_label=None, x_ticks_labels=None, format_y=lambda y, _: y,
              rotation=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
    ax.set_xticks(ticks=range(len(x_ticks_labels)), labels=x_ticks_labels, rotation=rotation)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if title is not None:
        ax.set_title(title)

    return ax


def make_area_plot(df, subset=None, y_label=None, colors=None, format_y=lambda y, _: y, save=None, rotation=None, x_ticks=None, dict_legend=None):
    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    df.index = df.index.astype(int)
    if subset is None:
        if colors is None:
            df.plot.area(ax=ax)
        else:
            df.plot.area(ax=ax, color=colors)
    else:
        if colors is None:
            df[subset].plot.area(ax=ax)
        else:
            df[subset].plot.area(ax=ax, color=colors)

    if x_ticks is None:
        ax = format_ax(ax, title=y_label, x_ticks=df.index, format_y=format_y, rotation=rotation)
    else:
        ax = format_ax(ax, title=y_label, x_ticks=x_ticks, format_y=format_y, rotation=rotation)

    format_legend(ax, dict_legend=dict_legend)

    save_fig(fig, save=save)


def make_stacked_investment_plot(df, y_label, subset, scatter, save, colors, format_y, rotation, dict_legend):
    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    df[subset].plot(kind='bar', stacked=True, color=colors, ax=ax)
    scatter.plot(ax=ax, style='.', c='red')
    ax = format_ax_string(ax, title=y_label, x_ticks_labels=df.index, format_y=format_y, rotation=rotation)
    format_legend(ax, dict_legend=dict_legend)
    plt.axhline(y=0)
    save_fig(fig, save=save)


def make_stacked_bar_plot(df, y_label=None, subset=None, colors=None, format_y=lambda y, _: y, save=None, rotation=None,
                          index_int=True, dict_legend=None):
    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    if index_int:
        df.index = df.index.astype(int)

    if subset is not None:
        if colors is None:
            df[subset].plot(kind='bar', stacked=True, ax=ax)
        else:
            df[subset].plot(kind='bar', stacked=True, color=colors, ax=ax)
    else:
        if colors is None:
            df.plot(kind='bar', stacked=True, ax=ax)
        else:
            df.plot(kind='bar', stacked=True, color=colors, ax=ax)

    if index_int:
        ax = format_ax(ax, title=y_label, x_ticks=df.index, format_y=format_y, rotation=rotation)
    else:  # index is not an int
        ax = format_ax_string(ax, title=y_label, x_ticks_labels=df.index, format_y=format_y, rotation=rotation)

    format_legend(ax, dict_legend=dict_legend)

    save_fig(fig, save=save)


def stacked_bars(dict_df,  y_label, format_y=lambda y, _: y, colors=None, x_ticks=None, index_int=True, save=None, rotation=None,
                 dict_legend=None, n=2):
    """Plots stacked bars for different simulations. Allowed number of simulations to be compared: 2 or 3"""
    if n == 2:
        list_position = [1.1, 0]
        list_width = [0.3, 0.3]
        list_hatch = [None, ".."]
    else:  # n=3
        list_position = [1.6, 0.5, -0.6]
        list_width = [0.2, 0.2, 0.2]
        list_hatch = [None, "//", ".."]

    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    for i, (key, df) in enumerate(dict_df.items()):
        # df = df.rename(key)
        if index_int:
            df.index = df.index.astype(int)
        if colors is None:
            if i == 0:
                df.plot(kind='bar', stacked=True, ax=ax, position=list_position[i], width=list_width[i], hatch=list_hatch[i])
            else:
                df.plot(kind='bar', stacked=True, ax=ax, position=list_position[i], width=list_width[i],
                        hatch=list_hatch[i], legend=False)
        else:
            df.plot(kind='bar', stacked=True, ax=ax, color=colors, position=list_position[i], width=list_width[i], hatch=list_hatch[i])

        # if i == 0:
    ax = format_ax(ax, title=y_label, format_y=format_y)
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=rotation)
    format_legend(ax, dict_legend=dict_legend)

    save_fig(fig, save=save)


def make_line_plot(df, subset=None, y_label=None, colors=None, format_y=lambda y, _: y, save=None, rotation=None,
                   x_ticks=None, index_int=True, str=False, dict_legend=None, y_min=None):
    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    if index_int:
        df.index = df.index.astype(int)
    if subset is None:
        if colors is None:
            df.plot.line(ax=ax)
        else:
            df.plot.line(ax=ax, color=colors)
    else:
        if colors is None:
            df[subset].plot.line(ax=ax)
        else:
            df[subset].plot.line(ax=ax, color=colors)

    if not str:  # index is an int, not a string
        if x_ticks is None:
            ax = format_ax(ax, title=y_label, x_ticks=df.index, format_y=format_y, rotation=rotation, y_min=y_min)
        else:
            ax = format_ax(ax, title=y_label, x_ticks=x_ticks, format_y=format_y, rotation=rotation, y_min=y_min)
    else:
        ax = format_ax_string(ax, title=y_label, x_ticks_labels=df.index, format_y=format_y, rotation=rotation)
    format_legend(ax, dict_legend=dict_legend)

    save_fig(fig, save=save)


def make_line_plots(dict_df, y_label, format_y=lambda y, _: y, colors=None, x_ticks=None, index_int=True, save=None, rotation=None,
                    multiple_legend=False, y_min=None):
    """Make line plot by combining different scenarios."""
    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    for i, (key, df) in enumerate(dict_df.items()):
        if isinstance(df, pd.Series):
            df = df.rename(key)
        if index_int:
            df.index = df.index.astype(int)
        if colors is None:
            df.plot.line(ax=ax, style=STYLES[i])
        else:
            df.plot.line(ax=ax, color=colors, style=STYLES[i])

    if x_ticks is None:
        ax = format_ax(ax, title=y_label, x_ticks=df.index, format_y=format_y, rotation=rotation, y_min=y_min)
    else:
        ax = format_ax(ax, title=y_label, x_ticks=x_ticks, format_y=format_y, rotation=rotation, y_min=y_min)

    if not multiple_legend:
        format_legend(ax)
    else:
        format_legend_multiple(ax, dict_df, n_style=len(list(dict_df.keys())), n_color=df.shape[1])

    save_fig(fig, save=save)


def make_area_plot_multiple(df1, df2, y_label=None, colors=None, format_y=lambda y, _: y, save=None, rotation=None):
    """In progress: pour ajouter des """
    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    df1.index = df1.index.astype(int)
    df2.index = df2.index.astype(int)
    if colors is None:
        df1.plot.area(ax=ax)
    else:
        df1.plot.area(ax=ax, color=colors)

    ax = format_ax(ax, title=y_label, x_ticks=df1.index, format_y=format_y, rotation=rotation)

    format_legend(ax)

    save_fig(fig, save=save)


def plot_blackbox_optimization(dict_optimizer, save_path):
    for key in dict_optimizer.keys():
        optimizer = dict_optimizer[key]
        optimizer.plot_convergence(filename=os.path.join(save_path, "plots", f"optimizer_{key}_convergence.png"))
        optimizer.plot_acquisition(filename=os.path.join(save_path, "plots", f"optimizer_{key}_acquisition.png"))

        optimizer.save_evaluations(os.path.join(save_path, f'evaluations_optimizer_{key}.csv'))
        optimizer.save_report(os.path.join(save_path, f'report_optimizer_{key}.txt'))




