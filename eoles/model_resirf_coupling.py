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
    extract_use_elec

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
    def __init__(self, name, config, path, logger, nb_years, existing_capacity=None, existing_charging_capacity=None,
                 existing_energy_capacity=None, total_demand_RTE=595*1e3, residential_heating_demand_RTE=33*1e3,
                 H2_demand_RTE=50*1e3, social_cost_of_carbon=0, hourly_heat_elec=None, hourly_heat_gas=None,
                 year=2050):
        """

        :param name: str
        :param config: dict
        :param nb_years: int
        :param existing_capa: pd.Series
        :param total_demand_RTE: float
            Final electricity demand excluding P2G given in RTE scenarios
        :param residential_heating_demand_RTE: float
            Final residential heating demand given in RTE scenarios
        :param residential: bool
            Indicates whether we want to change initial RTE residential heating demand values in the profile demand
            When residential is False, we remove hourly residential profile summing up to total demand (RTE hypothesis)
            and we add a new hourly residential profile summing up to specified total demand value.
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
        self.total_demand_RTE = total_demand_RTE
        self.residential_demand_RTE = residential_heating_demand_RTE

        assert hourly_heat_elec is not None, "Hourly heat profile should be provided to the model when variable " \
                                                 "residential is set to False"

        # loading exogeneous variable data
        data_variable = read_input_variable(config, self.year)
        self.load_factors = data_variable["load_factors"]
        self.demand1y = data_variable["demand"]
        self.hourly_heat_elec = hourly_heat_elec
        self.demand1y = self.demand1y + self.hourly_heat_elec  # we add electricity demand from heating
        self.hourly_heat_gas = hourly_heat_gas
        self.total_H2_demand = H2_demand_RTE

        self.lake_inflows = data_variable["lake_inflows"]
        self.H2_demand = {}
        self.CH4_demand = {}

        # concatenate electricity demand data
        self.demand = self.demand1y
        for i in range(self.nb_years - 1):
            self.demand = pd.concat([self.demand, self.demand1y], ignore_index=True)

        if self.hourly_heat_gas is not None:  # we provide gas data
            self.demand_gas = self.hourly_heat_gas
            for i in range(self.nb_years - 1):
                self.demand_gas = pd.concat([self.demand_gas, self.hourly_heat_gas], ignore_index=True)

        # loading exogeneous static data
        data_static = read_input_static(self.config)
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
        self.charging_capex = data_static["charging_capex"]
        self.charging_opex = data_static["charging_opex"]
        self.eta_in = data_static["eta_in"]
        self.eta_out = data_static["eta_out"]
        self.conversion_efficiency = data_static["conversion_efficiency"]
        self.capacity_ex = data_static["capacity_ex"]
        self.miscellaneous = data_static["miscellaneous"]

        # calculate annuities
        self.annuities = calculate_annuities_capex(self.miscellaneous, self.capex, self.construction_time,
                                                   self.lifetime)
        self.storage_annuities = calculate_annuities_storage_capex(self.miscellaneous, self.storage_capex,
                                                                   self.construction_time, self.lifetime)

        # Update natural gaz vOM based on social cost of carbon
        self.vOM.loc["natural_gas"] = update_ngas_cost(self.vOM.loc["natural_gas"], scc=self.scc)

        # defining needed time steps
        self.first_hour = 0
        self.last_hour = len(self.demand)
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
            Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "biogas1",
                            "biogas2", "ocgt", "ccgt", "nuc", "h2_ccgt", "phs", "battery1", "battery4",
                            "methanation", "pyrogazification", "electrolysis", "natural_gas", "hydrogen", "methane"])
        # Variables Technologies
        self.model.vre = \
            Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river"])
        # Electricity generating technologies
        self.model.elec_balance = \
            Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "nuc", "phs",
                            "battery1", "battery4", "ocgt", "ccgt", "h2_ccgt"])

        # Technologies for upward FRR
        self.model.frr = Set(initialize=["lake", "phs", "ocgt", "ccgt", "nuc", "h2_ccgt"])

        # Technologies producing electricity (not including storage technologies)
        self.model.elec_gene = Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake",
                                               "nuc", "ocgt", "ccgt", "h2_ccgt"])

        # Primary energy production
        self.model.primary_gene = Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river",
                                                  "lake", "nuc", "biogas1", "biogas2", "pyrogazification",
                                                  "natural_gas"])
        # Technologies using electricity
        self.model.use_elec = Set(initialize=["phs", "battery1", "battery4", "electrolysis"])

        # Gas technologies used for balance (both CH4 and H2)
        self.model.CH4_balance = Set(initialize=["biogas1", "biogas2", "pyrogazification", "natural_gas", "methanation", "methane"])
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
            self.H2_demand[hour] = self.total_H2_demand/8760  # We make the assumption that H2 demand profile is flat

        # Set the methane demand for each hour
        if self.hourly_heat_gas is None:
            for hour in self.model.h:
                self.CH4_demand[hour] = self.miscellaneous['CH4_demand']
        else:
            for hour in self.model.h:
                self.CH4_demand[hour] = self.demand_gas[hour]  # a bit redundant, could be removed

    def define_variables(self):

        def capacity_bounds(model, i):
            if i in self.maximum_capacity.keys():
                return self.existing_capacity[i], self.maximum_capacity[i]
            else:
                return self.existing_capacity[i], None

        def charging_capacity_bounds(model, i):
            if i in self.maximum_charging_capacity.keys():
                return self.existing_charging_capacity[i], self.maximum_capacity[i]
            else:
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

        # Hourly electricity input of battery storage GWh  # TODO: check that unit is right
        self.model.storage = \
            Var(((storage, h) for storage in self.model.str for h in self.model.h), within=NonNegativeReals,
                initialize=0)

        # Energy stored in each storage technology in GWh = Stage of charge
        self.model.stored = \
            Var(((storage, h) for storage in self.model.str for h in self.model.h), within=NonNegativeReals,
                initialize=0)

        # Charging power capacity of each storage technology in GW  # TODO: check the unit
        self.model.charging_capacity = \
            Var(self.model.str, within=NonNegativeReals, bounds=charging_capacity_bounds)

        # Energy volume of storage technology in GWh
        self.model.energy_capacity = \
            Var(self.model.str, within=NonNegativeReals, bounds=energy_capacity_bounds)

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
            """Constraint on stored energy to be equal at the end than at the start."""
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

        def biogas_constraint_rule(model):
            """Constraint on biogas. The annual power production from biogas is limited to a certain amount."""
            gene_biogas = sum(model.gene['biogas1', hour] + model.gene['biogas2', hour] for hour in model.h)
            return gene_biogas <= self.miscellaneous['max_biogas'] * 1000  # max biogas yearly energy expressed in TWh

        def hydrogen_balance_constraint_rule(model, h):
            """Constraint on hydrogen's balance. Hydrogen production must satisfy CCGT-H2 plants, H2 demand and
            methanation demand."""
            gene_e_h = model.gene['electrolysis', h] + model.gene['hydrogen', h]
            dem_sto = model.gene['h2_ccgt', h] / self.conversion_efficiency['h2_ccgt'] + self.H2_demand[h] + model.storage[
                'hydrogen', h]
            return gene_e_h == dem_sto

        def methane_balance_constraint_rule(model, h):
            """Constraint on methane's balance. Methane production must satisfy CCGT and OCGT plants and CH4 demand"""
            gene_methane = model.gene['methanation', h] + model.gene['biogas1', h] + model.gene['biogas2', h] + \
                           model.gene['pyrogazification', h] + model.gene['methane', h] + model.gene["natural_gas", h]
            dem_sto = model.gene['ocgt', h] / self.conversion_efficiency['ocgt'] + model.gene['ccgt', h] / \
                      self.conversion_efficiency['ccgt'] + self.CH4_demand[h] + model.storage['methane', h]
            return gene_methane == dem_sto

        def reserves_constraint_rule(model, h):
            """Constraint on frr reserves"""
            res_req = sum(self.epsilon[vre] * model.capacity[vre] for vre in model.vre)
            load_req = self.demand[h] * self.miscellaneous['load_uncertainty'] * (1 + self.miscellaneous['delta'])
            return sum(model.reserve[frr, h] for frr in model.frr) == res_req + load_req

        def electricity_adequacy_constraint_rule(model, h):
            """Constraint for supply/demand electricity relation'"""
            storage = sum(model.storage[str, h] for str in model.str_elec)  # need in electricity storage
            gene_from_elec = model.gene['electrolysis', h] / self.conversion_efficiency['electrolysis'] + model.gene[
                'methanation', h] / self.conversion_efficiency[
                                 'methanation']  # technologies using electricity for conversion
            return sum(model.gene[balance, h] for balance in model.elec_balance) >= (
                    self.demand[h] + storage + gene_from_elec)

        def ramping_nuc_up_constraint_rule(model, h):
            """Constraint setting an upper ramping limit for nuclear flexibility"""
            previous_h = model.h.last() if h == 0 else h - 1
            return model.gene['nuc', h] - model.gene['nuc', previous_h] + model.reserve['nuc', h] - model.reserve[
                'nuc', previous_h] <= \
                   self.miscellaneous['hourly_ramping_nuc'] * model.capacity['nuc']

        def ramping_nuc_down_constraint_rule(model, h):
            """Constraint setting a lower ramping limit for nuclear flexibility"""
            previous_h = model.h.last() if h == 0 else h - 1
            return model.gene['nuc', previous_h] - model.gene['nuc', h] + model.reserve['nuc', previous_h] - \
                   model.reserve[
                       'nuc', h] <= \
                   self.miscellaneous['hourly_ramping_nuc'] * model.capacity['nuc']

        def methanation_constraint_rule(model, h):
            """Constraint on CO2 balance from methanization"""
            # TODO: cette contrainte actuellement est peu réaliste car c'est une contrainte horaire ! normalement
            #  (cf Behrang, 2021) cela devrait être une contrainte sur la somme sur toutes les heures, cf contrainte suivante
            return model.gene['methanation', h] / self.miscellaneous['eta_methanation'] <= (
                    model.gene['biogas1', h] + model.gene['biogas2', h]) * self.miscellaneous[
                       'percentage_co2_from_methanization']

        def methanation_CO2_constraint_rule(model):
            """Constraint on CO2 balance from methanization, summing over all hours of the year"""
            return sum(model.gene['methanation', h] for h in model.h) / self.conversion_efficiency['methanation'] <= (
                    sum(model.gene['biogas1', h] + model.gene['biogas2', h] for h in model.h) * self.miscellaneous[
                'percentage_co2_from_methanization']
            )

        self.model.generation_vre_constraint = \
            Constraint(self.model.h, self.model.vre, rule=generation_vre_constraint_rule)

        self.model.generation_capacity_constraint = \
            Constraint(self.model.h, self.model.tec, rule=generation_capacity_constraint_rule)

        self.model.battery_1_capacity_constraint = \
            Constraint(rule=battery1_capacity_constraint_rule)

        self.model.battery_4_capacity_constraint = \
            Constraint(rule=battery4_capacity_constraint_rule)

        self.model.frr_capacity_constraint = \
            Constraint(self.model.h, self.model.frr, rule=frr_capacity_constraint_rule)

        self.model.storing_constraint = \
            Constraint(self.model.h, self.model.str, rule=storing_constraint_rule)

        self.model.storage_constraint = \
            Constraint(self.model.str, rule=storage_constraint_rule)

        self.model.lake_reserve_constraint = \
            Constraint(self.model.months, rule=lake_reserve_constraint_rule)

        self.model.stored_capacity_constraint = \
            Constraint(self.model.h, self.model.str, rule=stored_capacity_constraint)

        self.model.storage_capacity_1_constraint = \
            Constraint(self.model.h, self.model.str, rule=storage_charging_capacity_constraint_rule)

        self.model.battery_capacity_constraint = \
            Constraint(self.model.battery, rule=battery_capacity_constraint_rule)

        self.model.biogas_constraint = \
            Constraint(rule=biogas_constraint_rule)

        self.model.hydrogen_balance_constraint = \
            Constraint(self.model.h, rule=hydrogen_balance_constraint_rule)

        self.model.methane_balance_constraint = \
            Constraint(self.model.h, rule=methane_balance_constraint_rule)

        self.model.reserves_constraint = \
            Constraint(self.model.h, rule=reserves_constraint_rule)

        self.model.electricity_adequacy_constraint = \
            Constraint(self.model.h, rule=electricity_adequacy_constraint_rule)

        self.model.ramping_nuc_up_constraint = \
            Constraint(self.model.h, rule=ramping_nuc_up_constraint_rule)

        self.model.ramping_nuc_down_constraint = \
            Constraint(self.model.h, rule=ramping_nuc_down_constraint_rule)

        self.model.methanation_constraint = \
            Constraint(rule=methanation_CO2_constraint_rule)

    def define_objective(self):
        def objective_rule(model):
            """Objective value in 10**3 M€, or Billion €"""

            return (sum(
                (model.capacity[tec] - self.existing_capacity[tec]) * self.annuities[tec] * self.nb_years for tec in
                model.tec)
                    + sum(
                        (model.energy_capacity[storage_tecs] - self.existing_energy_capacity[storage_tecs]) *
                        self.storage_annuities[
                            storage_tecs] * self.nb_years for storage_tecs in model.str)
                    + sum(
                        (model.charging_capacity[storage_tecs] - self.existing_charging_capacity[storage_tecs]) *
                        self.charging_capex[
                            storage_tecs] * self.nb_years for storage_tecs in model.str)
                    + sum(model.capacity[tec] * self.fOM[tec] * self.nb_years for tec in model.tec)
                    + sum(
                        model.charging_capacity[storage_tecs] * self.charging_opex[storage_tecs] * self.nb_years
                        for storage_tecs in model.str)
                    + sum(sum(model.gene[tec, h] * self.vOM[tec] for h in model.h) for tec in model.tec)
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
        self.hourly_generation = extract_hourly_generation(self.model, self.demand)
        self.spot_price = extract_spot_price(self.model, self.last_hour)
        self.capacities = extract_capacities(self.model)
        self.energy_capacity = extract_energy_capacity(self.model)
        self.electricity_generation = extract_supply_elec(self.model, self.nb_years)
        self.primary_generation = extract_primary_gene(self.model, self.nb_years)
        self.use_elec = extract_use_elec(self.model, self.nb_years, self.miscellaneous)
        self.summary = extract_summary(self.model, self.demand, self.H2_demand, self.CH4_demand,
                                       self.annuities, self.storage_annuities, self.fOM, self.vOM, self.charging_opex,
                                       self.charging_capex, self.conversion_efficiency, self.nb_years)
        self.results = {'objective': self.objective, 'summary': self.summary,
                        'hourly_generation': self.hourly_generation,
                        'capacities': self.capacities, 'energy_capacity': self.energy_capacity,
                        'supply_elec': self.electricity_generation, 'primary_generation': self.primary_generation,
                        'use_elec': self.use_elec}


def read_input_variable(config, year):
    """Reads data defined at the hourly scale"""
    load_factors = get_pandas(config["load_factors"],
                              lambda x: pd.read_csv(x, index_col=[0, 1], header=None).squeeze("columns"))
    demand = get_pandas(config["demand"], lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GW
    demand_no_residential = process_RTE_demand(config, year, demand)

    lake_inflows = get_pandas(config["lake_inflows"],
                              lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GWh
    o = dict()
    o["load_factors"] = load_factors
    o["demand"] = demand_no_residential
    o["lake_inflows"] = lake_inflows
    return o


def read_input_static(config):
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
                       lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # 1e6€/GW
    storage_capex = get_pandas(config["storage_capex"],
                               lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # 1e6€/GW
    fOM = get_pandas(config["fOM"],
                     lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # 1e6€/GW/year
    # TODO: il y a des erreurs d'unités dans le choix des vOM je crois !!
    vOM = get_pandas(config["vOM"],
                     lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # 1e6€/GWh
    charging_capex = get_pandas(config["charging_capex"],
                                lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # M€/GW/year
    charging_opex = get_pandas(config["charging_opex"],
                               lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # M€/GWh
    eta_in = get_pandas(config["eta_in"],
                        lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    eta_out = get_pandas(config["eta_out"],
                         lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    conversion_efficiency = get_pandas(config["conversion_efficiency"],
                         lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    capacity_ex = get_pandas(config["capacity_ex"],
                             lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))  # GWh
    miscellaneous = get_pandas(config["miscellaneous"],
                               lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
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
    o["charging_capex"] = charging_capex
    o["charging_opex"] = charging_opex
    o["eta_in"] = eta_in
    o["eta_out"] = eta_out
    o["conversion_efficiency"] = conversion_efficiency
    o["capacity_ex"] = capacity_ex
    o["miscellaneous"] = miscellaneous
    return o


def extract_summary(model, demand, H2_demand, CH4_demand, annuities, storage_annuities, fOM, vOM,
                    charging_opex, charging_capex, conversion_efficiency, nb_years):
    # TODO: A CHANGER !!!
    summary = {}  # final dictionary for output
    elec_demand_tot = sum(demand[hour] for hour in model.h) / 1000  # electricity demand in TWh
    H2_demand_tot = sum(H2_demand[hour] for hour in model.h) / 1000  # H2 demand in TWh
    CH4_demand_tot = sum(CH4_demand[hour] for hour in model.h) / 1000  # CH4 demand in TWh

    elec_spot_price = [-1e6 * model.dual[model.electricity_adequacy_constraint[h]] for h in
                       model.h]  # 1e3€/GWh = €/MWh
    CH4_spot_price = [1e6 * model.dual[model.methane_balance_constraint[h]] for h in model.h]  # 1e3€ / GWh
    H2_spot_price = [1e6 * model.dual[model.hydrogen_balance_constraint[h]] for h in model.h]  # 1e3€ / GWh
    gene_elec = [sum(value(model.gene[tec, hour]) for tec in model.elec_gene) for hour in model.h]
    storage_elec = [sum(value(model.gene[tec, hour]) for tec in model.str_elec) for hour in model.h]

    weighted_price_demand = sum([elec_spot_price[h] * demand[h] for h in model.h]) / (
            elec_demand_tot * 1e3)  # €/MWh
    summary["weighted_price_demand"] = weighted_price_demand

    weighted_price_generation = sum([elec_spot_price[h] * gene_elec[h] for h in model.h]) / sum(gene_elec)  # €/MWh
    summary["weighted_price_generation"] = weighted_price_generation

    summary["demand_tot"] = elec_demand_tot
    summary["hydrogen_tot"] = H2_demand_tot
    summary["methane_tot"] = CH4_demand_tot

    # Overall yearly energy generated by the technology in TWh
    # TODO: à mettre dans un autre dataframe
    gene_per_tec = {}
    for tec in model.tec:
        gene_per_tec[tec] = sum(value(model.gene[tec, hour]) for hour in model.h) / 1000  # TWh

    summary.update(gene_per_tec)

    sumgene_elec = sum(gene_per_tec[tec] for tec in model.elec_gene)  # production in TWh
    summary["sumgene_elec"] = sumgene_elec

    G2P_bought = sum(CH4_spot_price[hour] * (
            value(model.gene["ocgt", hour]) / conversion_efficiency['ocgt'] + value(model.gene["ccgt", hour]) /
            conversion_efficiency['ccgt'])
                     for hour in model.h) / 1e3 + sum(H2_spot_price[hour] * (
            value(model.gene["h2_ccgt", hour]) / conversion_efficiency['h2_ccgt']) for hour in
                                                      model.h) / 1e3  # 1e6€ car l'objectif du modèle est en 1e9 €

    # LCOE: initial calculus from electricity costs
    lcoe_elec = (sum(
        (value(model.capacity[tec])) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[tec] * vOM[tec] * 1000
        for tec in
        model.elec_balance) +
                 sum(
                     (value(model.energy_capacity[storage_tecs])) * storage_annuities[
                         storage_tecs] * nb_years for storage_tecs in model.str_elec) +
                 sum(
                     value(model.charging_capacity[storage_tecs]) * (charging_opex[storage_tecs] + charging_capex[
                         storage_tecs]) * nb_years
                     for storage_tecs in model.str_elec) + G2P_bought) / sumgene_elec  # € / MWh

    summary["lcoe_elec"] = lcoe_elec

    # We calculate the costs associated to functioning of each system (elec, CH4, gas)
    costs_elec = sum(
        (value(model.capacity[tec])) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[tec] * vOM[tec] * 1000
        for tec in
        model.elec_balance) + \
                 sum(
                     (value(model.energy_capacity[storage_tecs])) * storage_annuities[
                         storage_tecs] * nb_years for storage_tecs in model.str_elec) + \
                 sum(
                     value(model.charging_capacity[storage_tecs]) * (charging_opex[storage_tecs] + charging_capex[
                         storage_tecs]) * nb_years
                     for storage_tecs in model.str_elec)  # 1e6 €

    costs_CH4 = sum(
        (value(model.capacity[tec])) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[tec] * vOM[tec] * 1000
        for tec in
        model.CH4_balance) + \
                sum(
                    (value(model.energy_capacity[storage_tecs])) * storage_annuities[
                        storage_tecs] * nb_years for storage_tecs in model.str_CH4) + \
                sum(
                    value(model.charging_capacity[storage_tecs]) * (charging_opex[storage_tecs] + charging_capex[
                        storage_tecs]) * nb_years
                    for storage_tecs in model.str_CH4)  # 1e6 €

    costs_H2 = sum(
        (value(model.capacity[tec])) * (annuities[tec] + fOM[tec]) * nb_years + gene_per_tec[tec] * vOM[tec] * 1000
        for tec in
        model.H2_balance) + \
               sum(
                   (value(model.energy_capacity[storage_tecs])) * storage_annuities[
                       storage_tecs] * nb_years for storage_tecs in model.str_H2) + \
               sum(
                   value(model.charging_capacity[storage_tecs]) * (charging_opex[storage_tecs] + charging_capex[
                       storage_tecs]) * nb_years
                   for storage_tecs in model.str_H2)  # 1e6 €

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
    return summary_df

