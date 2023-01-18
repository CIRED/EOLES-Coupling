"""
Power system components.
"""

import pandas as pd
import numpy as np
import logging
import json
import os
import math
from eoles.utils import get_pandas, process_RTE_demand, calculate_annuities_capex, calculate_annuities_storage_capex, \
    update_ngas_cost, define_month_hours, calculate_annuities_renovation, get_technical_cost, extract_hourly_generation, \
    extract_spot_price, extract_capacities, extract_energy_capacity, extract_supply_elec, extract_primary_gene, \
    extract_use_elec, extract_renovation_rates, extract_heat_gene, calculate_LCOE_gene_tec, calculate_LCOE_conv_tec, \
    extract_charging_capacity, extract_annualized_costs_investment_new_capa
from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Set,
    NonNegativeReals,  # a verifier, mais je ne pense pas que ce soit une erreur
    Constraint,
    SolverFactory,
    Suffix,
    Var,
    Objective,
    value
)


# file_handler = logging.FileHandler('root_log.log')
# file_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
# logger.addHandler(file_handler)


class ModelEOLES():
    def __init__(self, name, config, path, logger, nb_years, hourly_heat_elec, hourly_heat_gas, heat_wood=0, heat_fuel=0,
                 existing_capacity=None,
                 existing_charging_capacity=None, existing_energy_capacity=None, maximum_capacity=None,
                 method_hourly_profile="valentin", social_cost_of_carbon=0, year=2050, anticipated_year=2050,
                 scenario_cost=None, existing_annualized_costs_elec=0,
                 existing_annualized_costs_CH4=0, existing_annualized_costs_H2=0):
        """

        :param name: str
        :param config: dict
        :param nb_years: int
        :param existing_capa: pd.Series
        :param social_cost_of_carbon: int
            Social cost of carbon used to calculate emissions
        """
        self.name = name
        self.config = config
        self.logger = logger
        self.path = path
        self.model = ConcreteModel()
        # Dual Variable, used to get the marginal value of an equation.
        self.model.dual = Suffix(direction=Suffix.IMPORT)
        self.nb_years = nb_years
        self.scc = social_cost_of_carbon
        self.year = year
        self.anticipated_year = anticipated_year
        self.existing_annualized_costs_elec = existing_annualized_costs_elec
        self.existing_annualized_costs_CH4 = existing_annualized_costs_CH4
        self.existing_annualized_costs_H2 = existing_annualized_costs_H2

        assert hourly_heat_elec is not None, "Hourly electricity heat profile should be provided to the model"
        assert hourly_heat_gas is not None, "Hourly gas heat profile should be provided to the model"

        # loading exogeneous variable data
        data_variable = read_input_variable(config, self.anticipated_year, method=method_hourly_profile)
        self.load_factors = data_variable["load_factors"]
        self.elec_demand1y = data_variable["demand"]
        self.lake_inflows = data_variable["lake_inflows"]

        self.hourly_heat_elec_1y = hourly_heat_elec

        self.elec_demand1y = self.elec_demand1y + self.hourly_heat_elec_1y  # we add electricity demand from heating, may require changing if we include multiple weather years
        self.hourly_heat_gas = hourly_heat_gas  # TODO: attention, bien vérifier que ce profil inclut toute la demande en gaz qu'on veut, notamment le tertiaire et ECS également !
        self.heat_wood = heat_wood
        self.heat_fuel = heat_fuel

        self.H2_demand = {}
        self.CH4_demand = {}

        # concatenate electricity demand data
        self.elec_demand = self.elec_demand1y
        for i in range(self.nb_years - 1):  # TODO: a changer si on ajoute plusieurs années météo, ca ne sera plus simplement une concaténation
            self.elec_demand = pd.concat([self.elec_demand, self.elec_demand1y], ignore_index=True)

        if self.hourly_heat_gas is not None:  # we provide hourly gas data
            self.gas_demand = self.hourly_heat_gas
            for i in range(self.nb_years - 1):
                self.gas_demand = pd.concat([self.gas_demand, self.hourly_heat_gas], ignore_index=True)

        # loading exogeneous static data
        # data_static = read_input_static(self.config, self.year)
        data_technology = read_technology_data(self.config, self.year)  # get current technology data
        data_annual = read_annual_demand_data(self.config, self.anticipated_year)  # get anticipated demand and energy prices
        data_technology.update(data_annual)
        data_static = data_technology
        if scenario_cost is not None:  # we update costs based on data given in scenario
            for df in scenario_cost.keys():
                for tec in scenario_cost[df].keys():
                    data_static[df][tec] = scenario_cost[df][tec]

        self.epsilon = data_static["epsilon"]
        if existing_capacity is not None:
            self.existing_capacity = existing_capacity
        else:  # default value
            self.existing_capacity = data_static["existing_capacity"]
        if existing_charging_capacity is not None:
            self.existing_charging_capacity = existing_charging_capacity
        else:  # default value
            self.existing_charging_capacity = data_static["existing_charging_capacity"]
        if existing_energy_capacity is not None:
            self.existing_energy_capacity = existing_energy_capacity
        else:  # default value
            self.existing_energy_capacity = data_static["existing_energy_capacity"]
        if maximum_capacity is not None:
            self.maximum_capacity = maximum_capacity
        else:
            self.maximum_capacity = data_static["maximum_capacity"]
        self.maximum_charging_capacity = data_static["maximum_charging_capacity"]
        self.maximum_energy_capacity = data_static["maximum_energy_capacity"]
        self.fix_capa = data_static["fix_capa"]
        self.lifetime = data_static["lifetime"]
        self.construction_time = data_static["construction_time"]
        self.capex = data_static["capex"]
        self.storage_capex = data_static["storage_capex"]
        self.fOM = data_static["fOM"]
        self.vOM = data_static["vOM"]
        self.eta_in = data_static["eta_in"]
        self.eta_out = data_static["eta_out"]
        self.conversion_efficiency = data_static["conversion_efficiency"]
        self.miscellaneous = data_static["miscellaneous"]
        self.biomass_potential = data_static["biomass_potential"]
        self.total_H2_demand = data_static["demand_H2_RTE"]
        self.energy_prices = data_static["energy_prices"]
        self.vOM["wood_boiler"], self.vOM["fuel_boiler"] = self.energy_prices["wood"] * 1e-3, self.energy_prices[
            "fuel"] * 1e-3  # €/kWh

        # calculate annuities
        self.annuities = calculate_annuities_capex(self.miscellaneous, self.capex, self.construction_time,
                                                   self.lifetime)
        self.storage_annuities = calculate_annuities_storage_capex(self.miscellaneous, self.storage_capex,
                                                                   self.construction_time, self.lifetime)

        # Update natural gaz vOM based on social cost of carbon
        self.vOM.loc["natural_gas"] = update_ngas_cost(self.vOM.loc["natural_gas"], scc=self.scc)  # €/kWh
        self.vOM["fuel_boiler"] = update_ngas_cost(self.vOM["fuel_boiler"], scc=self.scc, emission_rate=0.271)  # to check !!
        self.vOM["wood_boiler"] = update_ngas_cost(self.vOM["wood_boiler"], scc=self.scc,
                                                   emission_rate=0)  # to check !!

        # defining needed time steps
        self.first_hour = 0
        self.last_hour = len(self.elec_demand)
        self.first_month = self.miscellaneous['first_month']

        self.hours_by_months = {1: 744, 2: 672, 3: 744, 4: 720, 5: 744, 6: 720, 7: 744, 8: 744, 9: 720, 10: 744,
                                11: 720, 12: 744}  # question: pas de problème avec les années bissextiles ?

        self.months_hours = {1: range(0, self.hours_by_months[self.first_month])}
        self.month_hours = define_month_hours(self.first_month, self.nb_years, self.months_hours, self.hours_by_months)

    def define_sets(self):
        # Range of hour
        self.model.h = \
            RangeSet(self.first_hour, self.last_hour - 1)
        # Months
        self.model.months = \
            RangeSet(1, 12 * self.nb_years)
        # Technologies
        self.model.tec = \
            Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "methanization",
                            "ocgt", "ccgt", "nuclear", "h2_ccgt", "phs", "battery1", "battery4",
                            "methanation", "pyrogazification", "electrolysis", "natural_gas", "hydrogen", "methane"])
        # Variables Technologies
        self.model.vre = \
            Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river"])
        # Electricity generating technologies
        self.model.elec_balance = \
            Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "nuclear", "phs",
                            "battery1", "battery4", "ocgt", "ccgt", "h2_ccgt"])

        # Technologies for upward FRR
        self.model.frr = Set(initialize=["lake", "phs", "ocgt", "ccgt", "nuclear", "h2_ccgt"])

        # Technologies producing electricity (not including storage technologies)
        # self.model.elec_gene = Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake",
        #                                        "nuclear", "ocgt", "ccgt", "h2_ccgt"])
        self.model.elec_gene = Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake",
                                               "nuclear"])
        # Primary energy production
        self.model.primary_gene = Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river",
                                                  "lake", "nuclear", "methanization", "pyrogazification",
                                                  "natural_gas"])
        # Technologies using electricity
        self.model.use_elec = Set(initialize=["phs", "battery1", "battery4", "electrolysis"])
        # Technologies producing gas
        self.model.gas_gene = Set(initialize=["methanization", "pyrogazification"])

        # Gas technologies used for balance (both CH4 and H2)
        self.model.CH4_balance = Set(
            initialize=["methanization", "pyrogazification", "natural_gas", "methanation", "methane"])
        self.model.H2_balance = Set(initialize=["electrolysis", "hydrogen"])

        # Conversion technologies
        self.model.from_elec_to_CH4 = Set(initialize=["methanation"])
        self.model.from_elec_to_H2 = Set(initialize=["electrolysis"])
        self.model.from_CH4_to_elec = Set(initialize=["ocgt", "ccgt"])
        self.model.from_H2_to_elec = Set(initialize=["h2_ccgt"])

        # Storage technologies
        self.model.str = \
            Set(initialize=["phs", "battery1", "battery4", "hydrogen", "methane"])
        # Electricity storage Technologies
        self.model.str_elec = Set(initialize=["phs", "battery1", "battery4"])
        # Battery Storage
        self.model.battery = Set(initialize=["battery1", "battery4"])
        # CH4 storage
        self.model.str_CH4 = Set(initialize=["methane"])
        # H2 storage
        self.model.str_H2 = Set(initialize=["hydrogen"])

    def define_other_demand(self):
        # Set the hydrogen demand for each hour
        for hour in self.model.h:
            # self.H2_demand[hour] = self.miscellaneous['H2_demand']
            self.H2_demand[hour] = self.total_H2_demand / 8760  # We make the assumption that H2 demand profile is flat

        # Set the methane demand for each hour
        if self.hourly_heat_gas is None:
            for hour in self.model.h:
                self.CH4_demand[hour] = self.miscellaneous['CH4_demand']
        else:
            for hour in self.model.h:
                self.CH4_demand[hour] = self.gas_demand[hour]  # a bit redundant, could be removed

    def define_variables(self):

        def capacity_bounds(model, i):
            if i in self.maximum_capacity.keys():  # there exists a max capacity
                return self.existing_capacity[i], self.maximum_capacity[
                    i]  # existing capacity is always the lower bound
            else:
                return self.existing_capacity[i], None  # in this case, only lower bound exists

        def charging_capacity_bounds(model, i):
            # TODO: j'ai enlevé cette contrainte, car je suppose ici que la seule contrainte provient de la discharging capacity
            # if i in self.maximum_charging_capacity.keys():
            #     return self.existing_charging_capacity[i], self.maximum_capacity[i]
            # else:
            return self.existing_charging_capacity[i], None

        def energy_capacity_bounds(model, i):
            if i in self.maximum_energy_capacity.keys():
                return self.existing_energy_capacity[i], self.maximum_energy_capacity[i]
            else:
                return self.existing_energy_capacity[i], None

            # Hourly energy generation in GWh/h

        self.model.gene = \
            Var(((tec, h) for tec in self.model.tec for h in self.model.h), within=NonNegativeReals, initialize=0)

        # Overall yearly installed capacity in GW
        self.model.capacity = \
            Var(self.model.tec, within=NonNegativeReals, bounds=capacity_bounds)

        # Charging power capacity of each storage technology in GW  # TODO: check the unit
        self.model.charging_capacity = \
            Var(self.model.str, within=NonNegativeReals, bounds=charging_capacity_bounds)

        # Energy volume of storage technology in GWh
        self.model.energy_capacity = \
            Var(self.model.str, within=NonNegativeReals, bounds=energy_capacity_bounds)

        # Hourly electricity input of battery storage GWh  # TODO: check that unit is right
        self.model.storage = \
            Var(((storage, h) for storage in self.model.str for h in self.model.h), within=NonNegativeReals,
                initialize=0)

        # Energy stored in each storage technology in GWh = Stage of charge
        self.model.stored = \
            Var(((storage, h) for storage in self.model.str for h in self.model.h), within=NonNegativeReals,
                initialize=0)

        # Required upward frequency restoration reserve in GW
        self.model.reserve = \
            Var(((reserve, h) for reserve in self.model.frr for h in self.model.h), within=NonNegativeReals,
                initialize=0)

    def fix_values(self):
        for tec in self.model.tec:
            if tec in self.fix_capa.keys():
                self.model.capacity[tec].fix(self.fix_capa[tec])

    def define_constraints(self):
        def generation_vre_constraint_rule(model, h, vre):
            """Cnstraint on variables renewable profiles generation."""
            return model.gene[vre, h] == model.capacity[vre] * self.load_factors[vre, h]

        def generation_capacity_constraint_rule(model, h, tec):
            """Constraint on maximum power for non-VRE technologies."""
            return model.capacity[tec] >= model.gene[tec, h]

        def battery1_capacity_constraint_rule(model):
            """Constraint on capacity of battery 1h."""
            # TODO: check that the constraint is ok
            return model.capacity['battery1'] == model.energy_capacity['battery1']

        def battery4_capacity_constraint_rule(model):
            """Constraint on capacity of battery 4h."""
            # TODO: check that the constraint is ok
            return model.capacity['battery4'] == model.energy_capacity['battery4'] / 4

        def frr_capacity_constraint_rule(model, h, frr):
            """Constraint on maximum generation including reserves"""
            return model.capacity[frr] >= model.gene[frr, h] + model.reserve[frr, h]

        def storing_constraint_rule(model, h, storage_tecs):
            """Constraint on energy storage consistency."""
            hPOne = h + 1 if h < (self.last_hour - 1) else 0
            charge = model.storage[storage_tecs, h] * self.eta_in[storage_tecs]
            discharge = model.gene[storage_tecs, h] / self.eta_out[storage_tecs]
            flux = charge - discharge
            return model.stored[storage_tecs, hPOne] == model.stored[storage_tecs, h] + flux

        def storage_constraint_rule(model, storage_tecs):
            """Constraint on stored energy to be equal at the end and at the start."""
            first = model.stored[storage_tecs, self.first_hour]
            last = model.stored[storage_tecs, self.last_hour - 1]
            charge = model.storage[storage_tecs, self.last_hour - 1] * self.eta_in[storage_tecs]
            discharge = model.gene[storage_tecs, self.last_hour - 1] / self.eta_out[storage_tecs]
            flux = charge - discharge
            return first == last + flux

        def lake_reserve_constraint_rule(model, month):
            """Constraint on maximum monthly lake generation. Total generation from lake over a month cannot exceed
            a certain given value."""
            return sum(model.gene['lake', hour] for hour in self.months_hours[month]) <= self.lake_inflows[month] * 1000

        def stored_capacity_constraint(model, h, storage_tecs):
            """Constraint on maximum energy that is stored in storage units"""
            return model.stored[storage_tecs, h] <= model.energy_capacity[storage_tecs]

        def storage_charging_capacity_constraint_rule(model, h, storage_tecs):
            """Constraint on the capacity with hourly charging relationship of storage. Energy entering the battery
            during one hour cannot exceed the charging capacity."""
            return model.storage[storage_tecs, h] <= model.charging_capacity[storage_tecs]

        def battery_capacity_constraint_rule(model, battery):
            """Constraint on battery's capacity: battery charging capacity equals battery discharging capacity."""
            # TODO: check that the constraint is ok: charging capacity = capacity ?
            return model.charging_capacity[battery] == model.capacity[battery]

        def storage_charging_discharging_constraint_rule(model, storage_tec):
            """Constraint to limit charging capacity to be lower than discharging capacity"""
            return model.charging_capacity[storage_tec] <= model.capacity[storage_tec]

        def methanization_constraint_rule(model):
            """Constraint on methanization. The annual power production from methanization is limited to a certain amount."""
            gene_biogas = sum(model.gene['methanization', hour] for hour in model.h)
            return gene_biogas <= self.biomass_potential["methanization"] * 1000  # max biogas yearly energy expressed in TWh

        def pyrogazification_constraint_rule(model):
            """Constraint on pyrogazification. The annual power production from pyro is limited to a certain amount."""
            gene_pyro = sum(model.gene['pyrogazification', hour] for hour in model.h)
            return gene_pyro <= self.biomass_potential["pyrogazification"] * 1000  # max pyro yearly energy expressed in TWh

        def reserves_constraint_rule(model, h):
            """Constraint on frr reserves"""
            res_req = sum(self.epsilon[vre] * model.capacity[vre] for vre in model.vre)
            load_req = self.elec_demand[h] * self.miscellaneous['load_uncertainty'] * (1 + self.miscellaneous['delta'])
            return sum(model.reserve[frr, h] for frr in model.frr) == res_req + load_req

        def hydrogen_balance_constraint_rule(model, h):
            """Constraint on hydrogen's balance. Hydrogen production must satisfy CCGT-H2 plants, H2 demand and
            methanation demand."""
            gene_e_h = model.gene['electrolysis', h] + model.gene['hydrogen', h]
            dem_sto = model.gene['h2_ccgt', h] / self.conversion_efficiency['h2_ccgt'] + self.H2_demand[h] + \
                      model.storage[
                          'hydrogen', h]
            return gene_e_h == dem_sto

        def methane_balance_constraint_rule(model, h):
            """Constraint on methane's balance. Methane production must satisfy CCGT and OCGT plants and CH4 demand"""
            gene_methane = model.gene['methanation', h] + model.gene['methanization', h] + \
                           model.gene['pyrogazification', h] + model.gene['methane', h] + model.gene["natural_gas", h]
            dem_sto = model.gene['ocgt', h] / self.conversion_efficiency['ocgt'] + model.gene['ccgt', h] / \
                      self.conversion_efficiency['ccgt'] + self.CH4_demand[h] + model.storage['methane', h]
            return gene_methane == dem_sto

        def electricity_adequacy_constraint_rule(model, h):
            """Constraint for supply/demand electricity relation'"""
            storage = sum(model.storage[str, h] for str in model.str_elec)  # need in electricity storage
            gene_from_elec = model.gene['electrolysis', h] / self.conversion_efficiency['electrolysis'] + model.gene[
                'methanation', h] / self.conversion_efficiency[
                                 'methanation']  # technologies using electricity for conversion
            prod_elec = sum(model.gene[balance, h] for balance in model.elec_balance)
            return prod_elec >= (
                    self.elec_demand[h] + storage + gene_from_elec)

        def ramping_nuclear_up_constraint_rule(model, h):
            """Constraint setting an upper ramping limit for nuclear flexibility"""
            previous_h = model.h.last() if h == 0 else h - 1
            return model.gene['nuclear', h] - model.gene['nuclear', previous_h] + model.reserve['nuclear', h] - \
                   model.reserve[
                       'nuclear', previous_h] <= \
                   self.miscellaneous['hourly_ramping_nuclear'] * model.capacity['nuclear']

        def ramping_nuclear_down_constraint_rule(model, h):
            """Constraint setting a lower ramping limit for nuclear flexibility"""
            previous_h = model.h.last() if h == 0 else h - 1
            return model.gene['nuclear', previous_h] - model.gene['nuclear', h] + model.reserve['nuclear', previous_h] - \
                   model.reserve[
                       'nuclear', h] <= \
                   self.miscellaneous['hourly_ramping_nuclear'] * model.capacity['nuclear']

        def methanation_constraint_rule(model, h):
            """Constraint on CO2 balance from methanization. OLD VERSION."""
            # TODO: cette contrainte actuellement est peu réaliste car c'est une contrainte horaire ! normalement
            #  (cf Behrang, 2021) cela devrait être une contrainte sur la somme sur toutes les heures, cf contrainte suivante
            return model.gene['methanation', h] / self.conversion_efficiency['methanation'] <= (
                    model.gene['methanization', h]) * self.miscellaneous[
                       'percentage_co2_from_methanization']

        def methanation_CO2_constraint_rule(model):
            """Constraint on CO2 balance from methanization, summing over all hours of the year"""
            return sum(model.gene['methanation', h] for h in model.h) / self.conversion_efficiency['methanation'] <= (
                    sum(model.gene['methanization', h] for h in model.h) * self.miscellaneous[
                'percentage_co2_from_methanization']
            )

        self.model.generation_vre_constraint = \
            Constraint(self.model.h, self.model.vre, rule=generation_vre_constraint_rule)

        self.model.generation_capacity_constraint = \
            Constraint(self.model.h, self.model.tec, rule=generation_capacity_constraint_rule)

        self.model.battery_1_capacity_constraint = Constraint(rule=battery1_capacity_constraint_rule)

        self.model.battery_4_capacity_constraint = Constraint(rule=battery4_capacity_constraint_rule)

        self.model.frr_capacity_constraint = Constraint(self.model.h, self.model.frr, rule=frr_capacity_constraint_rule)

        self.model.storing_constraint = Constraint(self.model.h, self.model.str, rule=storing_constraint_rule)

        self.model.storage_constraint = Constraint(self.model.str, rule=storage_constraint_rule)

        self.model.lake_reserve_constraint = Constraint(self.model.months, rule=lake_reserve_constraint_rule)

        self.model.stored_capacity_constraint = Constraint(self.model.h, self.model.str,
                                                           rule=stored_capacity_constraint)

        self.model.storage_capacity_1_constraint = \
            Constraint(self.model.h, self.model.str, rule=storage_charging_capacity_constraint_rule)

        self.model.battery_capacity_constraint = Constraint(self.model.battery, rule=battery_capacity_constraint_rule)

        self.model.storage_charging_discharging_constraint = \
            Constraint(self.model.str, rule=storage_charging_discharging_constraint_rule)

        self.model.biogas_constraint = Constraint(rule=methanization_constraint_rule)

        self.model.pyrogazification_constraint = Constraint(rule=pyrogazification_constraint_rule)

        self.model.ramping_nuclear_up_constraint = Constraint(self.model.h, rule=ramping_nuclear_up_constraint_rule)

        self.model.ramping_nuclear_down_constraint = Constraint(self.model.h, rule=ramping_nuclear_down_constraint_rule)

        self.model.methanation_constraint = Constraint(rule=methanation_CO2_constraint_rule)

        self.model.reserves_constraint = Constraint(self.model.h, rule=reserves_constraint_rule)

        self.model.hydrogen_balance_constraint = Constraint(self.model.h, rule=hydrogen_balance_constraint_rule)

        self.model.methane_balance_constraint = Constraint(self.model.h, rule=methane_balance_constraint_rule)

        self.model.electricity_adequacy_constraint = Constraint(self.model.h, rule=electricity_adequacy_constraint_rule)

    def define_objective(self):
        def objective_rule(model):
            """Objective value in 10**3 M€, or 1e9€"""

            return (sum(
                (model.capacity[tec] - self.existing_capacity[tec]) * self.annuities[tec] * self.nb_years for tec in
                model.tec)
                    + sum(
                        (model.energy_capacity[storage_tecs] - self.existing_energy_capacity[storage_tecs]) *
                        self.storage_annuities[
                            storage_tecs] * self.nb_years for storage_tecs in model.str)
                    # + sum(
                    #     (model.charging_capacity[storage_tecs] - self.existing_charging_capacity[storage_tecs]) *
                    #     self.charging_capex[
                    #         storage_tecs] * self.nb_years for storage_tecs in model.str)
                    + sum(model.capacity[tec] * self.fOM[tec] * self.nb_years for tec in model.tec)
                    # + sum(
                    #     model.charging_capacity[storage_tecs] * self.charging_opex[storage_tecs] * self.nb_years
                    #     for storage_tecs in model.str)
                    + sum(sum(model.gene[tec, h] * self.vOM[tec] for h in model.h) for tec in model.tec)
                    + self.heat_fuel * self.vOM["fuel_boiler"] + self.heat_wood * self.vOM["wood_boiler"]  # we add variable costs from wood and fuel
                    ) / 1000

        # Creation of the objective -> Cost
        self.model.objective = Objective(rule=objective_rule)

    def build_model(self):
        self.define_sets()
        self.define_other_demand()
        self.define_variables()
        self.fix_values()
        self.define_constraints()
        self.define_objective()

    def solve(self, solver_name, quick_save=False):
        self.opt = SolverFactory(solver_name)
        self.logger.info("Solving EOLES model using %s", self.opt.name)
        self.solver_results = self.opt.solve(self.model,
                                             options={'Presolve': 2, 'LogFile': self.path + "/logfile_" + self.name})
        # TODO: à modifier pour utiliser un objet Path, ce sera plus propre

        status = self.solver_results["Solver"][0]["Status"]
        termination_condition = self.solver_results["Solver"][0]["Termination condition"]

        if status == "ok" and termination_condition == "optimal":
            self.logger.info("Optimization successful")
            self.extract_optimisation_results()
        elif status == "warning" and termination_condition == "other":
            self.logger.warning(
                "WARNING! Optimization might be sub-optimal. Writing output anyway"
            )
            self.extract_optimisation_results()
        else:
            self.logger.error(
                "Optimisation failed with status %s and terminal condition %s"
                % (status, termination_condition)
            )
            self.objective = np.nan
        return self.solver_results, status, termination_condition

    def extract_optimisation_results(self):
        """

        :param m: ModelEOLES
        :return:
        """
        # get value of objective function
        self.objective = self.solver_results["Problem"][0]["Upper bound"]
        self.technical_cost, self.emissions = get_technical_cost(self.model, self.objective, self.scc)
        self.hourly_generation = extract_hourly_generation(self.model, self.elec_demand,  list(self.CH4_demand.values()),
                                                           list(self.H2_demand.values()))
        self.spot_price = extract_spot_price(self.model, self.last_hour)
        self.capacities = extract_capacities(self.model)
        self.energy_capacity = extract_energy_capacity(self.model)
        self.charging_capacity = extract_charging_capacity(self.model)
        self.electricity_generation = extract_supply_elec(self.model, self.nb_years)
        self.primary_generation = extract_primary_gene(self.model, self.nb_years)
        # self.heat_generation = extract_heat_gene(self.model, self.nb_years)

        self.new_capacity_annualized_costs, self.new_energy_capacity_annualized_costs = \
            extract_annualized_costs_investment_new_capa(self.capacities, self.energy_capacity,
                                                         self.existing_capacity, self.existing_energy_capacity, self.annuities,
                                                         self.storage_annuities, self.fOM)
        # self.use_elec = extract_use_elec(self.model, self.nb_years, self.miscellaneous)
        self.summary, self.generation_per_technology, \
        self.lcoe_per_tec = extract_summary(self.model, self.elec_demand, self.H2_demand, self.CH4_demand,
                                            self.existing_capacity, self.existing_energy_capacity, self.annuities,
                                            self.storage_annuities, self.fOM, self.vOM, self.conversion_efficiency,
                                            self.existing_annualized_costs_elec, self.existing_annualized_costs_CH4,
                                            self.existing_annualized_costs_H2, self.nb_years)
        self.results = {'objective': self.objective, 'summary': self.summary,
                        'hourly_generation': self.hourly_generation,
                        'capacities': self.capacities, 'energy_capacity': self.energy_capacity,
                        'supply_elec': self.electricity_generation, 'primary_generation': self.primary_generation}


def read_input_variable(config, year, method="valentin"):
    """Reads data defined at the hourly scale"""
    load_factors = get_pandas(config["load_factors"],
                              lambda x: pd.read_csv(x, index_col=[0, 1], header=None).squeeze("columns"))
    demand = get_pandas(config["demand"], lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    demand_no_residential = process_RTE_demand(config, year, demand, method=method)

    lake_inflows = get_pandas(config["lake_inflows"],
                              lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GWh
    o = dict()
    o["load_factors"] = load_factors
    o["demand"] = demand_no_residential
    o["lake_inflows"] = lake_inflows
    return o


def read_technology_data(config, year):
    """Read technology data (capex, opex, capacity potential, etc...)
        config: json file
            Includes paths to files
        year: int
            Year to get capex."""
    epsilon = get_pandas(config["epsilon"], lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    existing_capacity = get_pandas(config["existing_capacity"],
                                   lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    existing_charging_capacity = get_pandas(config["existing_charging_capacity"],
                                            lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    existing_energy_capacity = get_pandas(config["existing_energy_capacity"],
                                          lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    maximum_capacity = get_pandas(config["maximum_capacity"],
                                  lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    maximum_charging_capacity = get_pandas(config["maximum_charging_capacity"],
                                           lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    maximum_energy_capacity = get_pandas(config["maximum_energy_capacity"],
                                         lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    fix_capa = get_pandas(config["fix_capa"],
                          lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    lifetime = get_pandas(config["lifetime"],
                          lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # years
    construction_time = get_pandas(config["construction_time"],
                                   lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # years
    capex = get_pandas(config["capex"],
                       lambda x: pd.read_csv(x, index_col=0))  # 1e6€/GW
    capex = capex[[str(year)]].squeeze()  # get capex for year of interest
    storage_capex = get_pandas(config["storage_capex"],
                               lambda x: pd.read_csv(x, index_col=0))  # 1e6€/GW
    storage_capex = storage_capex[[str(year)]].squeeze()  # get storage capex for year of interest
    fOM = get_pandas(config["fOM"], lambda x: pd.read_csv(x, index_col=0))  # 1e6€/GW/year
    fOM = fOM[[str(year)]].squeeze()  # get fOM for year of interest
    # TODO: il y a des erreurs d'unités dans le choix des vOM je crois !!
    vOM = get_pandas(config["vOM"],
                     lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # 1e6€/GWh
    eta_in = get_pandas(config["eta_in"],
                        lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    eta_out = get_pandas(config["eta_out"],
                         lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    conversion_efficiency = get_pandas(config["conversion_efficiency"],
                                       lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    miscellaneous = get_pandas(config["miscellaneous"],
                               lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    biomass_potential = get_pandas(config["biomass_potential"],
                                   lambda x: pd.read_csv(x, index_col=0))  # 1e6€/GW
    biomass_potential = biomass_potential[[str(year)]].squeeze()  # get storage capex for year of interest

    o = dict()
    o["epsilon"] = epsilon
    o["existing_capacity"] = existing_capacity
    o["existing_charging_capacity"] = existing_charging_capacity
    o["existing_energy_capacity"] = existing_energy_capacity
    o["maximum_capacity"] = maximum_capacity
    o["maximum_charging_capacity"] = maximum_charging_capacity
    o["maximum_energy_capacity"] = maximum_energy_capacity
    o["fix_capa"] = fix_capa
    o["lifetime"] = lifetime
    o["construction_time"] = construction_time
    o["capex"] = capex
    o["storage_capex"] = storage_capex
    o["fOM"] = fOM
    o["vOM"] = vOM
    o["eta_in"] = eta_in
    o["eta_out"] = eta_out
    o["conversion_efficiency"] = conversion_efficiency
    o["miscellaneous"] = miscellaneous
    o["biomass_potential"] = biomass_potential
    return o


def read_annual_demand_data(config, year):
    """Read annual demand data (H2, energy prices)
        config: json file
            Includes paths to files
        year: int
            Year to get capex."""
    demand_H2_timesteps = get_pandas(config["demand_H2_timesteps"],
                                     lambda x: pd.read_csv(x, index_col=0).squeeze())
    demand_H2_RTE = demand_H2_timesteps[year]  # TWh

    energy_prices = get_pandas(config["energy_prices"],
                               lambda x: pd.read_csv(x, index_col=0))  # €/MWh
    energy_prices = energy_prices[[str(year)]].squeeze()  # get storage capex for year of interest

    o = dict()
    o["demand_H2_RTE"] = demand_H2_RTE * 1e3  # GWh
    o["energy_prices"] = energy_prices  # € / MWh
    return o


def read_input_static(config, year):
    """Read static data"""
    epsilon = get_pandas(config["epsilon"], lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    existing_capacity = get_pandas(config["existing_capacity"],
                                   lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    existing_charging_capacity = get_pandas(config["existing_charging_capacity"],
                                            lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    existing_energy_capacity = get_pandas(config["existing_energy_capacity"],
                                          lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    maximum_capacity = get_pandas(config["maximum_capacity"],
                                  lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    maximum_charging_capacity = get_pandas(config["maximum_charging_capacity"],
                                           lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    maximum_energy_capacity = get_pandas(config["maximum_energy_capacity"],
                                         lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    fix_capa = get_pandas(config["fix_capa"],
                          lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    lifetime = get_pandas(config["lifetime"],
                          lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # years
    construction_time = get_pandas(config["construction_time"],
                                   lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # years
    capex = get_pandas(config["capex"],
                       lambda x: pd.read_csv(x, index_col=0))  # 1e6€/GW
    capex = capex[[str(year)]].squeeze()  # get capex for year of interest
    storage_capex = get_pandas(config["storage_capex"],
                               lambda x: pd.read_csv(x, index_col=0))  # 1e6€/GW
    storage_capex = storage_capex[[str(year)]].squeeze()  # get storage capex for year of interest
    fOM = get_pandas(config["fOM"], lambda x: pd.read_csv(x, index_col=0))  # 1e6€/GW/year
    fOM = fOM[[str(year)]].squeeze()  # get fOM for year of interest
    # TODO: il y a des erreurs d'unités dans le choix des vOM je crois !!
    vOM = get_pandas(config["vOM"],
                     lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # 1e6€/GWh
    eta_in = get_pandas(config["eta_in"],
                        lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    eta_out = get_pandas(config["eta_out"],
                         lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    conversion_efficiency = get_pandas(config["conversion_efficiency"],
                                       lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    miscellaneous = get_pandas(config["miscellaneous"],
                               lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    biomass_potential = get_pandas(config["biomass_potential"],
                               lambda x: pd.read_csv(x, index_col=0))  # 1e6€/GW
    biomass_potential = biomass_potential[[str(year)]].squeeze()  # get storage capex for year of interest
    demand_H2_timesteps = get_pandas(config["demand_H2_timesteps"],
                                     lambda x: pd.read_csv(x, index_col=0).squeeze())
    demand_H2_RTE = demand_H2_timesteps[year]  # TWh

    o = dict()
    o["epsilon"] = epsilon
    o["existing_capacity"] = existing_capacity
    o["existing_charging_capacity"] = existing_charging_capacity
    o["existing_energy_capacity"] = existing_energy_capacity
    o["maximum_capacity"] = maximum_capacity
    o["maximum_charging_capacity"] = maximum_charging_capacity
    o["maximum_energy_capacity"] = maximum_energy_capacity
    o["fix_capa"] = fix_capa
    o["lifetime"] = lifetime
    o["construction_time"] = construction_time
    o["capex"] = capex
    o["storage_capex"] = storage_capex
    o["fOM"] = fOM
    o["vOM"] = vOM
    o["eta_in"] = eta_in
    o["eta_out"] = eta_out
    o["conversion_efficiency"] = conversion_efficiency
    o["miscellaneous"] = miscellaneous
    o["biomass_potential"] = biomass_potential
    o["demand_H2_RTE"] = demand_H2_RTE * 1e3  # GWh
    return o


def extract_summary(model, demand, H2_demand, CH4_demand, existing_capacity, existing_energy_capacity, annuities,
                    storage_annuities, fOM, vOM, conversion_efficiency, existing_annualized_costs_elec,
                    existing_annualized_costs_CH4, existing_annualized_costs_H2, nb_years):
    # TODO: A CHANGER !!!
    summary = {}  # final dictionary for output
    elec_demand_tot = sum(demand[hour] for hour in model.h) / 1000  # electricity demand in TWh
    H2_demand_tot = sum(H2_demand[hour] for hour in model.h) / 1000  # H2 demand in TWh
    CH4_demand_tot = sum(CH4_demand[hour] for hour in model.h) / 1000  # CH4 demand in TWh

    elec_spot_price = [-1e6 * model.dual[model.electricity_adequacy_constraint[h]] for h in
                       model.h]  # 1e3€/GWh = €/MWh
    CH4_spot_price = [1e6 * model.dual[model.methane_balance_constraint[h]] for h in model.h]  # 1e3€ / GWh = €/MWh
    H2_spot_price = [1e6 * model.dual[model.hydrogen_balance_constraint[h]] for h in model.h]  # 1e3€ / GWh = €/MWh
    gene_elec = [sum(value(model.gene[tec, hour]) for tec in model.elec_gene) for hour in model.h]
    storage_elec = [sum(value(model.gene[tec, hour]) for tec in model.str_elec) for hour in model.h]

    weighted_elec_price_demand = sum([elec_spot_price[h] * demand[h] for h in model.h]) / (
            elec_demand_tot * 1e3)  # €/MWh
    summary["weighted_elec_price_demand"] = weighted_elec_price_demand
    weighted_CH4_price_demand = sum([CH4_spot_price[h] * CH4_demand[h] for h in model.h]) / (
            CH4_demand_tot * 1e3)  # €/MWh
    summary["weighted_CH4_price_demand"] = weighted_CH4_price_demand
    weighted_H2_price_demand = sum([H2_spot_price[h] * H2_demand[h] for h in model.h]) / (
            H2_demand_tot * 1e3)  # €/MWh
    summary["weighted_H2_price_demand"] = weighted_H2_price_demand

    weighted_price_generation = sum([elec_spot_price[h] * gene_elec[h] for h in model.h]) / sum(gene_elec)  # €/MWh
    summary["weighted_price_generation"] = weighted_price_generation

    summary["elec_demand_tot"] = elec_demand_tot
    summary["hydrogen_demand_tot"] = H2_demand_tot
    summary["methane_demand_tot"] = CH4_demand_tot

    # Overall yearly energy generated by the technology in TWh
    # TODO: à mettre dans un autre dataframe
    gene_per_tec = {}
    for tec in model.tec:
        gene_per_tec[tec] = sum(value(model.gene[tec, hour]) for hour in model.h) / 1000  # TWh

    # summary.update(gene_per_tec)

    sumgene_elec = sum(gene_per_tec[tec] for tec in model.elec_gene) + gene_per_tec["ocgt"] + gene_per_tec["ccgt"] + \
                   gene_per_tec["h2_ccgt"]  # production in TWh
    summary["sumgene_elec"] = sumgene_elec

    # LCOE per technology
    lcoe_per_tec = {}
    lcoe_elec_gene = calculate_LCOE_gene_tec(model.elec_gene, model, annuities, fOM, vOM, nb_years,
                                             gene_per_tec)  # € / MWh-e
    lcoe_elec_conv_CH4 = calculate_LCOE_conv_tec(["ocgt", "ccgt"], model, annuities, fOM, conversion_efficiency,
                                                 CH4_spot_price, nb_years, gene_per_tec)  # € / MWh-e
    lcoe_elec_conv_H2 = calculate_LCOE_conv_tec(["h2_ccgt"], model, annuities, fOM, conversion_efficiency,
                                                H2_spot_price, nb_years, gene_per_tec)  # € / MWh-e
    lcoe_gas_gene = calculate_LCOE_gene_tec(model.gas_gene, model, annuities, fOM, vOM, nb_years,
                                            gene_per_tec)  # € / MWh-th
    lcoe_per_tec.update(lcoe_elec_gene)
    lcoe_per_tec.update(lcoe_elec_conv_CH4)
    lcoe_per_tec.update(lcoe_elec_conv_H2)
    lcoe_per_tec.update(lcoe_gas_gene)

    G2P_bought = sum(CH4_spot_price[hour] * (
            value(model.gene["ocgt", hour]) / conversion_efficiency['ocgt'] + value(model.gene["ccgt", hour]) /
            conversion_efficiency['ccgt'])
                     for hour in model.h) / 1e3 + sum(H2_spot_price[hour] * (
            value(model.gene["h2_ccgt", hour]) / conversion_efficiency['h2_ccgt']) for hour in
                                                      model.h) / 1e3  # 1e6€ car l'objectif du modèle est en 1e9 €
    # LCOE: initial calculus from electricity costs
    # TODO: attention, j'ai supprimé les charging opex et capex pour les batteries
    # lcoe_elec = (sum(
    #     (value(model.capacity[tec])) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[tec] * vOM[tec] * 1000
    #     for tec in
    #     model.elec_balance) +
    #              sum(
    #                  (value(model.energy_capacity[storage_tecs])) * storage_annuities[
    #                      storage_tecs] * nb_years for storage_tecs in model.str_elec) + G2P_bought) / sumgene_elec  # €/MWh
    lcoe_elec = (existing_annualized_costs_elec + sum(
        (value(model.capacity[tec]) - existing_capacity[tec]) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[tec] * vOM[tec] * 1000
        for tec in model.elec_balance) +
                 sum(
                     (value(model.energy_capacity[str]) - existing_energy_capacity[str]) * storage_annuities[
                         str] * nb_years for str in model.str_elec) + G2P_bought) / sumgene_elec  # €/MWh

    summary["lcoe_elec"] = lcoe_elec

    # We calculate the costs associated to functioning of each system (elec, CH4, gas)
    costs_elec = existing_annualized_costs_elec + sum(
        (value(model.capacity[tec]) - existing_capacity[tec]) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[tec] * vOM[tec] * 1000
        for tec in model.elec_balance) + \
                 sum(
                     (value(model.energy_capacity[str]) - existing_energy_capacity[str]) * storage_annuities[
                         str] * nb_years for str in model.str_elec) # 1e6 €

    costs_CH4 = existing_annualized_costs_CH4 + sum(
        (value(model.capacity[tec]) - existing_capacity[tec]) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[tec] * vOM[tec] * 1000
        for tec in
        model.CH4_balance) + \
                sum(
                    (value(model.energy_capacity[str]) - existing_energy_capacity[str]) * storage_annuities[
                        str] * nb_years for str in model.str_CH4) # 1e6 €

    costs_H2 = existing_annualized_costs_H2 + sum(
        (value(model.capacity[tec]) - existing_capacity[tec]) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[tec] * vOM[tec] * 1000
        for tec in model.H2_balance) + \
               sum(
                   (value(model.energy_capacity[str]) - existing_energy_capacity[str]) * storage_annuities[
                       str] * nb_years for str in model.str_H2) # 1e6 €

    # print(costs_elec, costs_CH4, costs_H2)

    # We now calculate ratios to assign the costs depending on the part of those costs used to meet final demand,
    # or to meet other vectors demand.
    # Option 1: based on volumetric assumptions
    gene_from_CH4_to_elec = sum(gene_per_tec[tec] for tec in model.from_CH4_to_elec)  # TWh
    gene_from_H2_to_elec = sum(gene_per_tec[tec] for tec in model.from_H2_to_elec)  # TWh
    gene_from_elec_to_CH4 = sum(gene_per_tec[tec] for tec in model.from_elec_to_CH4)  # TWh
    gene_from_elec_to_H2 = sum(gene_per_tec[tec] for tec in model.from_elec_to_H2)  # TWh

    costs_CH4_to_demand = costs_CH4 * CH4_demand_tot / (CH4_demand_tot + gene_from_CH4_to_elec)  # 1e6 €
    costs_CH4_to_elec = costs_CH4 * gene_from_CH4_to_elec / (CH4_demand_tot + gene_from_CH4_to_elec)
    costs_H2_to_demand = costs_H2 * H2_demand_tot / (H2_demand_tot + gene_from_H2_to_elec)
    costs_H2_to_elec = costs_H2 * gene_from_H2_to_elec / (H2_demand_tot + gene_from_H2_to_elec)
    costs_elec_to_demand = costs_elec * elec_demand_tot / (
            elec_demand_tot + gene_from_elec_to_H2 + gene_from_elec_to_CH4)
    costs_elec_to_CH4 = costs_elec * gene_from_elec_to_CH4 / (
            elec_demand_tot + gene_from_elec_to_H2 + gene_from_elec_to_CH4)
    costs_elec_to_H2 = costs_elec * gene_from_elec_to_H2 / (
            elec_demand_tot + gene_from_elec_to_H2 + gene_from_elec_to_CH4)

    lcoe_elec_volume = (costs_CH4_to_elec + costs_H2_to_elec + costs_elec_to_demand) / elec_demand_tot  # € / MWh
    lcoe_CH4_volume = (costs_elec_to_CH4 + costs_CH4_to_demand) / CH4_demand_tot  # € / MWh
    lcoe_H2_volume = (costs_elec_to_H2 + costs_H2_to_demand) / H2_demand_tot  # € / MWh
    summary["lcoe_elec_volume"], summary["lcoe_CH4_volume"], summary["lcoe_H2_volume"] = \
        lcoe_elec_volume, lcoe_CH4_volume, lcoe_H2_volume

    # Option 2: based on value assumptions (we weight each volume by the hourly price
    total_elec_spot_price = sum(elec_spot_price)
    total_CH4_spot_price = sum(CH4_spot_price)
    total_H2_spot_price = sum(H2_spot_price)
    gene_from_CH4_to_elec_value = sum(
        sum(value(model.gene[tec, hour]) * CH4_spot_price[hour] for hour in model.h) / (1000 * total_CH4_spot_price)
        for tec in model.from_CH4_to_elec)  # TWh
    gene_from_H2_to_elec_value = sum(
        sum(value(model.gene[tec, hour]) * H2_spot_price[hour] for hour in model.h) / (1000 * total_H2_spot_price)
        for tec in model.from_H2_to_elec)  # TWh
    gene_from_elec_to_CH4_value = sum(
        sum(value(model.gene[tec, hour]) * elec_spot_price[hour] for hour in model.h) / (
                1000 * total_elec_spot_price) for tec in model.from_elec_to_CH4)  # TWh
    gene_from_elec_to_H2_value = sum(
        sum(value(model.gene[tec, hour]) * elec_spot_price[hour] for hour in model.h) / (
                1000 * total_elec_spot_price) for tec in model.from_elec_to_H2)  # TWh
    elec_demand_tot_value = sum(demand[hour] * elec_spot_price[hour] for hour in model.h) / (
            1000 * total_elec_spot_price)
    CH4_demand_tot_value = sum(CH4_demand[hour] * CH4_spot_price[hour] for hour in model.h) / (
            1000 * total_CH4_spot_price)
    H2_demand_tot_value = sum(H2_demand[hour] * H2_spot_price[hour] for hour in model.h) / (
            1000 * total_H2_spot_price)

    costs_CH4_to_demand_value = costs_CH4 * CH4_demand_tot_value / (
            CH4_demand_tot_value + gene_from_CH4_to_elec_value)  # 1e6 €
    costs_CH4_to_elec_value = costs_CH4 * gene_from_CH4_to_elec_value / (
            CH4_demand_tot_value + gene_from_CH4_to_elec_value)
    costs_H2_to_demand_value = costs_H2 * H2_demand_tot_value / (H2_demand_tot_value + gene_from_H2_to_elec_value)
    costs_H2_to_elec_value = costs_H2 * gene_from_H2_to_elec_value / (
            H2_demand_tot_value + gene_from_H2_to_elec_value)
    costs_elec_to_demand_value = costs_elec * elec_demand_tot_value / (
            elec_demand_tot_value + gene_from_elec_to_H2_value + gene_from_elec_to_CH4_value)
    costs_elec_to_CH4_value = costs_elec * gene_from_elec_to_CH4_value / (
            elec_demand_tot_value + gene_from_elec_to_H2_value + gene_from_elec_to_CH4_value)
    costs_elec_to_H2_value = costs_elec * gene_from_elec_to_H2_value / (
            elec_demand_tot_value + gene_from_elec_to_H2_value + gene_from_elec_to_CH4_value)

    lcoe_elec_value = (
                              costs_CH4_to_elec_value + costs_H2_to_elec_value + costs_elec_to_demand_value) / elec_demand_tot  # € / MWh
    lcoe_CH4_value = (costs_elec_to_CH4_value + costs_CH4_to_demand_value) / CH4_demand_tot  # € / MWh
    lcoe_H2_value = (costs_elec_to_H2_value + costs_H2_to_demand_value) / H2_demand_tot  # € / MWh
    summary["lcoe_elec_value"], summary["lcoe_CH4_value"], summary["lcoe_H2_value"] = \
        lcoe_elec_value, lcoe_CH4_value, lcoe_H2_value


    # Pas à jour, ancienne version du code
    # # The whole electricity input for storage in TWh
    # nSTORAGE = {}
    # for storage in model.str:
    #     for hour in model.h:
    #         nSTORAGE[(storage, hour)] = value(model.storage[storage, hour])
    #
    # # Electricity cost per MWh produced (euros/MWh)
    # lcoe_sys = value(model.objective) * 1000 / gene_tot
    # summary["lcoe_sys"] = lcoe_sys

    # # Yearly storage related loss in % of power production and in TWh
    # str_loss_percent = 100 * (
    #         sum(value(model.storage[storage, hour]) for storage in model.str for hour in model.h) -
    #         sum(gene_per_tec[storage] * 1000 for storage in model.str)) / (gene_tot * 1000)
    # str_loss_TWh = gene_per_tec['electrolysis'] / miscellaneous['eta_electrolysis'] - hydrogen_tot / miscellaneous[
    #     'eta_electrolysis'] - gene_per_tec[
    #                    'h2_ccgt']  # TODO: je ne comprends pas pourquoi on a besoin de cette ligne alors qu'on n'en avait pas besoin avant quand on calculait en pourcentage
    # # TODO: aussi, pourquoi on ne prend pas en compte la production des centrales CCGT ? et donc la méthanation et ses pertes ?
    # for storage in model.str:
    #     if storage != 'hydrogen' and storage != 'methane':
    #         str_loss_TWh += sum(nSTORAGE[storage, hour] for hour in model.h) / 1000 - gene_per_tec[storage]
    #
    # summary["str_loss_percent"] = str_loss_percent
    # summary["str_loss_TWh"] = str_loss_TWh
    #
    # # Load curtailment in % of power production and in TWh
    # lc_percent = (100 * (
    #         gene_tot - demand_tot - hydrogen_tot / miscellaneous['eta_electrolysis']) / gene_tot) - str_loss_percent
    # lc_TWh = (gene_tot - demand_tot - hydrogen_tot / miscellaneous['eta_electrolysis']) - str_loss_TWh
    #
    # summary["load_curtail_percent"] = lc_percent
    # summary["load_curtail_TWh"] = lc_TWh
    summary_df = pd.Series(summary)
    return summary_df, gene_per_tec, lcoe_per_tec


