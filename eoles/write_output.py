import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

from eoles.inputs.resources import resources_data
from project.utils import save_fig

sns.set_theme(context="talk", style="whitegrid")


def plot_simulation(output, save_path):
    capacity_df = output["Capacities (GW)"]
    generation_df = output["Generation (TWh)"]
    primary_generation_df = output["Primary generation (TWh)"]
    conversion_generation_df = output["Conversion generation (TWh)"]
    prices_df = output["Prices (€/MWh)"]
    resirf_subsidies = output["Subsidies (%)"]
    resirf_costs_df = output["ResIRF costs (Billion euro)"]
    resirf_costs_eff_df = output["ResIRF costs eff (euro / kWh.year)"]
    resirf_consumption_df = output["ResIRF consumption (TWh)"]
    resirf_replacement_heater = output["ResIRF replacement heater (Thousand)"]
    resirf_stock_heater = output["ResIRF stock heater (Thousand)"]
    annualized_system_costs = output["Annualized system costs (Billion euro / year)"]

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
    CH4_generation = ["methanation", "natural_gas", "methanization", "pyrogazification"]
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
    make_line_plot(prices_df, y_label="Prices (€ / MWh)", colors=resources_data["colors_eoles"],
                   save=os.path.join(save_path, "prices.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    # Plot subsidies ResIRF
    make_line_plot(resirf_subsidies, y_label="Subsidies (%)", save=os.path.join(save_path, "resirf_subsidies.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    # Plot consumption ResIRF
    make_area_plot(resirf_consumption_df, subset=["Electricity", "Natural gas", "Oil fuel", "Wood fuel"],
                   y_label="Heating consumption (TWh)", colors=resources_data["colors_resirf"],
                   save=os.path.join(save_path, "resirf_consumption.png"), format_y=lambda y, _: '{:.0f}'.format(y))

    # Plot savings ResIRF
    make_area_plot(resirf_consumption_df, subset=['Saving heater', "Saving insulation"],
                   y_label="Savings (TWh)",
                   save=os.path.join(save_path, "resirf_savings.png"), format_y=lambda y, _: '{:.0f}'.format(y))

    # Plot costs ResIRF
    make_line_plot(resirf_costs_df, y_label="Costs (Billion euro)", save=os.path.join(save_path, "resirf_costs.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y))

    make_line_plot(resirf_costs_eff_df, y_label="Costs per saving (euro / kWh.year)", save=os.path.join(save_path, "resirf_costs_eff.png"),
                   format_y=lambda y, _: '{:.0f}'.format(y), rotation=45, x_ticks=resirf_costs_eff_df.index[::2])

    # Plot stock and replacement ResIRF
    make_area_plot(resirf_replacement_heater.T, y_label="Replacement heater (Thousand households)",
                   save=os.path.join(save_path, "resirf_replacement.png"), format_y=lambda y, _: '{:.0f}'.format(y))

    make_area_plot(resirf_stock_heater.T, y_label="Stock heater (Thousand households)",
                   save=os.path.join(save_path, "resirf_stock.png"), format_y=lambda y, _: '{:.0f}'.format(y))

    # Plot annualized system costs
    make_line_plot(annualized_system_costs, y_label="Annualized system costs",
                   save=os.path.join(save_path, "annualized_system_costs.png"), format_y=lambda y, _: '{:.0f}'.format(y))


def format_legend(ax):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def format_ax(ax: plt.Axes, title=None, y_label=None, x_label=None, x_ticks=None, format_y=lambda y, _: y,
              rotation=None):
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
    if x_ticks is not None:
        ax.set_xticks(ticks=x_ticks, labels=x_ticks)
    if rotation is not None:
        ax.set_xticklabels(ax.get_xticks(), rotation=45)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if title is not None:
        ax.set_title(title)

    return ax


def make_area_plot(df, subset=None, y_label=None, colors=None, format_y=lambda y, _: y, save=None, ymin=None, loc='upper'):
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

    ax = format_ax(ax, title=y_label, x_ticks=df.index, format_y=format_y)
    format_legend(ax)

    save_fig(fig, save=save)


def make_line_plot(df, subset=None, y_label=None, colors=None, format_y=lambda y, _: y, save=None, rotation=None,
                   x_ticks=None):
    if save is None:
        fig, ax = plt.subplots(1, 1)
    else:  # we change figure size when saving figure
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
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

    if x_ticks is None:
        ax = format_ax(ax, title=y_label, x_ticks=df.index, format_y=format_y, rotation=rotation)
    else:
        ax = format_ax(ax, title=y_label, x_ticks=x_ticks, format_y=format_y, rotation=rotation)
    format_legend(ax)

    save_fig(fig, save=save)
