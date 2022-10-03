"""
Power system components.
"""

import pandas as pd
import json
import os
from eoles.utils import get_pandas
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

import logging

LOG_FORMATTER = '%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# consoler handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
logger.addHandler(console_handler)


# file_handler = logging.FileHandler('root_log.log')
# file_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
# logger.addHandler(file_handler)


class ModelEOLES():
    def __init__(self, name, config, nb_years, existing_capa=None):
        """

        :param name: str
        :param config: dict
        :param nb_years: int
        :param existing_capa: pd.Series
        """
        self.name = name
        self.config = config
        self.model = ConcreteModel()
        # Dual Variable, used to get the marginal value of an equation.
        self.model.dual = Suffix(direction=Suffix.IMPORT)
        self.nb_years = nb_years

        # loading exogeneous variable data
        data_variable = read_input_variable(config)
        self.load_factors = data_variable["load_factors"]
        self.demand1y = data_variable["demand"]
        self.lake_inflows = data_variable["lake_inflows"]
        self.H2_demand = {}
        self.CH4_demand = {}

        # concatenate demand data
        self.demand = self.demand1y
        for i in range(nb_years - 1):
            self.demand = pd.concat([self.demand, self.demand1y], ignore_index=True)

        # loading exogeneous static data
        data_static = read_input_static(self.config)
        self.epsilon = data_static["epsilon"]
        if existing_capa is not None:
            self.existing_capa = existing_capa
        else:
            self.existing_capa = data_static["existing_capa"]
        self.capa_max = data_static["capa_max"]
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
        self.capacity_ex = data_static["capacity_ex"]
        self.miscellaneous = data_static["miscellaneous"]

        # calculate annuities
        self.annuities = calculate_annuities_capex(self.miscellaneous, self.capex, self.construction_time,
                                                   self.lifetime)
        self.storage_annuities = calculate_annuities_storage_capex(self.miscellaneous, self.storage_capex,
                                                                   self.construction_time, self.lifetime)

        # defining needed time steps
        self.first_hour = 0
        self.last_hour = len(self.demand)
        self.first_month = self.miscellaneous['first_month']

        self.hours_by_months = {1: 744, 2: 672, 3: 744, 4: 720, 5: 744, 6: 720, 7: 744, 8: 744, 9: 720, 10: 744,
                                11: 720,
                                12: 744}

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
                            "methanation", "pyrogazification", "electrolysis", "hydrogen", "methane"])
        # Power plants. Only used to calculate sum of generation.
        self.model.gen = \
            Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "ocgt", "ccgt",
                            "nuc"])
        # Technologies producing electricity
        self.model.gen_elec = Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake",
                                              "ocgt", "ccgt", "nuc", "h2_ccgt", "phs", "battery1", "battery4"])
        # Technologies using electricity
        self.model.use_elec = Set(initialize=["phs", "battery1", "battery4", "electrolysis"])
        # Variables Technologies
        self.model.vre = \
            Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river"])
        # Electricity generating technologies
        self.model.balance = \
            Set(initialize=["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "nuc", "phs",
                            "battery1", "battery4", "h2_ccgt", "ocgt", "ccgt"])
        # Storage Technologies
        self.model.str = \
            Set(initialize=["phs", "battery1", "battery4", "hydrogen", "methane"])
        # Battery Storage
        self.model.battery = \
            Set(initialize=["battery1", "battery4"])
        # Technologies for upward FRR
        self.model.frr = \
            Set(initialize=["lake", "phs", "ocgt", "ccgt", "nuc", "h2_ccgt"])

    def define_other_demand(self):
        # Set the hydrogen demand for each hour
        for hour in self.model.h:
            self.H2_demand[hour] = self.miscellaneous['H2_demand']

        # Set the methane demand for each hour
        for hour in self.model.h:
            self.CH4_demand[hour] = self.miscellaneous['CH4_demand']

    def define_variables(self):

        def capa_bounds(model, i):
            # TODO: check that the values are OK in miscellaneous
            if i in self.capa_max.keys():
                return (None, self.capa_max[i])
            elif i == 'phs':
                return (self.miscellaneous['phs_discharging_lower'],
                        self.miscellaneous['phs_discharging_upper'])  # TODO: c'est quoi ces bornes pour PHS ??
            else:
                return (None, None)

        def s_bounds(model, i):
            if i == 'phs':
                return (self.miscellaneous['phs_charging_lower'], self.miscellaneous['phs_charging_upper'])
            else:
                return (None, None)

        def capacity_bounds(model, i):
            # TODO: check that the values are OK in miscellaneous
            if i == 'phs':
                return (self.miscellaneous['phs_energy_lower'], self.miscellaneous['phs_energy_upper'])
            elif i == 'hydrogen':
                return (self.capacity_ex['hydrogen'], None)
            else:
                return (None, None)

            # Hourly energy generation in GWh/h

        self.model.gene = \
            Var(((tec, h) for tec in self.model.tec for h in self.model.h), within=NonNegativeReals, initialize=0)

        # Overall yearly installed capacity in GW
        self.model.capacity = \
            Var(self.model.tec, within=NonNegativeReals, bounds=capa_bounds)

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
            Var(self.model.str, within=NonNegativeReals, bounds=capa_bounds)

        # Energy volume of storage technology in GWh
        self.model.energy_capacity = \
            Var(self.model.str, within=NonNegativeReals, bounds=capacity_bounds)

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
            """Get constraint on variables renewable profiles generation."""

            return model.gene[vre, h] == model.capacity[vre] * self.load_factors[vre, h]

        def generation_capacity_constraint_rule(model, h, tec):
            """Get constraint on maximum power for non-VRE technologies."""

            return model.capacity[tec] >= model.gene[tec, h]

        def battery1_capacity_constraint_rule(model):
            """Get constraint on capacity of battery1."""
            # TODO: check that the constraint is ok
            return model.capacity['battery1'] == model.energy_capacity['battery1']

        def battery4_capacity_constraint_rule(model):
            """Get constraint on capacity of battery4."""
            # TODO: check that the constraint is ok
            return model.capacity['battery4'] == model.energy_capacity['battery4'] / 4

        def frr_capacity_constraint_rule(model, h, frr):
            """Get constraint on maximum generation including reserves"""

            return model.capacity[frr] >= model.gene[frr, h] + model.reserve[frr, h]

        def storing_constraint_rule(model, h, storage_tecs):
            """Get constraint on storing."""

            hPOne = h + 1 if h < (self.last_hour - 1) else 0
            charge = model.storage[storage_tecs, h] * self.eta_in[storage_tecs]
            discharge = model.gene[storage_tecs, h] / self.eta_out[storage_tecs]
            flux = charge - discharge
            return model.stored[storage_tecs, hPOne] == model.stored[storage_tecs, h] + flux

        def storage_constraint_rule(model, storage_tecs):
            """Get constraint on stored energy to be equal at the end than at the start."""

            first = model.stored[storage_tecs, self.first_hour]
            last = model.stored[storage_tecs, self.last_hour - 1]
            charge = model.storage[storage_tecs, self.last_hour - 1] * self.eta_in[storage_tecs]
            discharge = model.gene[storage_tecs, self.last_hour - 1] / self.eta_out[storage_tecs]
            flux = charge - discharge
            return first == last + flux

        def lake_reserve_constraint_rule(model, month):
            """Get constraint on maximum monthly lake generation."""

            return sum(model.gene['lake', hour] for hour in self.months_hours[month]) <= self.lake_inflows[month] * 1000

        def stored_capacity_constraint(model, h, storage_tecs):
            """Get constraint on maximum energy that is stored in storage units"""

            return model.stored[storage_tecs, h] <= model.energy_capacity[storage_tecs]

        def storage_capacity_1_constraint_rule(model, h, storage_tecs):
            """Get constraint on the capacity with hourly charging relationship of storage"""
            # TODO: check that the constraint is ok: STORAGE plus petit que capacité de charge
            return model.storage[storage_tecs, h] <= model.charging_capacity[storage_tecs]

        def battery_capacity_constraint_rule(model, battery):
            """Get constraint on battery's capacity."""
            # TODO: check that the constraint is ok: charging capacity = capacity ?
            return model.charging_capacity[battery] == model.capacity[battery]

        def biogas_constraint_rule(model):
            """Get constraint on biogas."""

            gene_biogas = sum(model.gene['biogas1', hour] + model.gene['biogas2', hour] for hour in model.h)

            return gene_biogas <= self.miscellaneous['max_biogas'] * 1000

        def hydrogen_balance_constraint_rule(model, h):
            """Get constraint on hydrogen's balance."""

            gene_e_h = model.gene['electrolysis', h] + model.gene['hydrogen', h]
            dem_sto = model.gene['h2_ccgt', h] / self.miscellaneous['eta_h2_ccgt'] + self.H2_demand[h] + model.storage[
                'hydrogen', h] + \
                      model.gene['methanation', h] * 4 / self.miscellaneous[
                          'eta_methanation']  # 4 h2 are required to produce one CH4
            return gene_e_h == dem_sto

        def methane_balance_constraint_rule(model, h):
            """Get constraint on methane's balance."""

            gene_methane = model.gene['methanation', h] + model.gene['biogas1', h] + model.gene['biogas2', h] + \
                           model.gene[
                               'pyrogazification', h] + model.gene['methane', h]
            dem_sto = model.gene['ocgt', h] / self.miscellaneous['eta_ocgt'] + model.gene['ccgt', h] / \
                      self.miscellaneous[
                          'eta_ccgt'] + \
                      self.CH4_demand[h] + model.storage['methane', h]
            return gene_methane == dem_sto

        def reserves_constraint_rule(model, h):
            """Get constraint on frr reserves"""

            res_req = sum(self.epsilon[vre] * model.capacity[vre] for vre in model.vre)
            load_req = self.demand[h] * self.miscellaneous['load_uncertainty'] * (1 + self.miscellaneous['delta'])
            return sum(model.reserve[frr, h] for frr in model.frr) == res_req + load_req

        def adequacy_constraint_rule(model, h):
            """Get constraint for 'supply/demand relation'"""

            storage = sum(model.storage[str, h] for str in model.str if (str != "hydrogen" and str != "methane"))
            gene_electrolysis = model.gene['electrolysis', h] / self.miscellaneous['eta_electrolysis']
            return sum(model.gene[balance, h] for balance in model.balance) >= (
                    self.demand[h] + storage + gene_electrolysis)

        def ramping_nuc_up_constraint_rule(model, h):
            """Sets an upper ramping limit for nuclear flexibility"""

            old_h = model.h.last() if h == 0 else h - 1
            return model.gene['nuc', h] - model.gene['nuc', old_h] + model.reserve['nuc', h] - model.reserve[
                'nuc', old_h] <= \
                   self.miscellaneous['hourly_ramping_nuc'] * model.capacity['nuc']

        def ramping_nuc_down_constraint_rule(model, h):
            """Sets a lower ramping limit for nuclear flexibility"""

            old_h = model.h.last() if h == 0 else h - 1
            return model.gene['nuc', old_h] - model.gene['nuc', h] + model.reserve['nuc', old_h] - model.reserve[
                'nuc', h] <= \
                   self.miscellaneous['hourly_ramping_nuc'] * model.capacity['nuc']

        def methanation_constraint_rule(model, h):
            """Get constraint on CO2's balance from methanization"""

            return model.gene['methanation', h] / self.miscellaneous['eta_methanation'] <= (
                    model.gene['biogas1', h] + model.gene['biogas2', h]) * self.miscellaneous[
                       'percentage_co2_from_methanization']

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
            Constraint(self.model.h, self.model.str, rule=storage_capacity_1_constraint_rule)

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

        self.model.adequacy_constraint = \
            Constraint(self.model.h, rule=adequacy_constraint_rule)

        self.model.ramping_nuc_up_constraint = \
            Constraint(self.model.h, rule=ramping_nuc_up_constraint_rule)

        self.model.ramping_nuc_down_constraint = \
            Constraint(self.model.h, rule=ramping_nuc_down_constraint_rule)

        self.model.methanation_constraint = \
            Constraint(self.model.h, rule=methanation_constraint_rule)

    def define_objective(self):
        def objective_rule(model):
            """Get constraint for the final objective function."""

            return (sum(
                (model.capacity[tec] - self.existing_capa[tec]) * self.annuities[tec] * self.nb_years for tec in
                model.tec)
                    + sum(
                        (model.energy_capacity[storage_tecs] - self.capacity_ex[storage_tecs]) * self.storage_annuities[
                            storage_tecs] * self.nb_years for storage_tecs in model.str)
                    + sum(model.capacity[tec] * self.fOM[tec] * self.nb_years for tec in model.tec)
                    + sum(
                        model.charging_capacity[storage_tecs] * (self.charging_opex[storage_tecs] + self.charging_capex[
                            storage_tecs]) * self.nb_years
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

    def solve(self, solver_name):
        self.opt = SolverFactory(solver_name)
        logger.info("Solving model using %s", self.opt.name)
        self.solver_results = self.opt.solve(self.model,
                                             options={'Presolve': 2, 'LogFile': "eoles/outputs/logfile_" + self.name})

        status = self.solver_results["Solver"][0]["Status"]
        termination_condition = self.solver_results["Solver"][0]["Termination condition"]

        if status == "ok" and termination_condition == "optimal":
            logger.info("Optimization successful")
            self.extract_optimisation_results()
        return self.solver_results, status, termination_condition

    def extract_optimisation_results(self):
        """

        :param m: ModelEOLES
        :return:
        """
        # get value of objective function
        self.objective = self.solver_results["Problem"][0]["Upper bound"]
        self.hourly_generation = extract_hourly_generation(self.model, self.demand)
        self.spot_price = extract_spot_price(self.model)
        self.capacities = extract_capacities(self.model)
        self.energy_capacity = extract_energy_capacity(self.model)
        self.supply_elec = extract_supply_elec(self.model, self.nb_years)
        self.use_elec = extract_use_elec(self.model, self.nb_years, self.miscellaneous)
        self.summary = extract_summary(self.model, self.demand, self.H2_demand, self.CH4_demand, self.miscellaneous)
        self.results = {'objective': self.objective, 'summary': self.summary,
                        'hourly_generation': self.hourly_generation,
                        'capacities': self.capacities, 'energy_capacity': self.energy_capacity,
                        'supply_elec': self.supply_elec, 'use_elec': self.use_elec}


def read_input_variable(config):
    """Reads data defined at the hourly scale"""
    load_factors = get_pandas(config["load_factors"],
                              lambda x: pd.read_csv(x, index_col=[0, 1], header=None).squeeze("columns"))
    demand = get_pandas(config["demand"], lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    lake_inflows = get_pandas(config["lake_inflows"],
                              lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    o = dict()
    o["load_factors"] = load_factors
    o["demand"] = demand
    o["lake_inflows"] = lake_inflows
    return o


def read_input_static(config):
    """Read static data"""
    epsilon = get_pandas(config["epsilon"], lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    existing_capa = get_pandas(config["existing_capa"],
                               lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    capa_max = get_pandas(config["capa_max"],
                          lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    fix_capa = get_pandas(config["fix_capa"],
                          lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    lifetime = get_pandas(config["lifetime"],
                          lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    construction_time = get_pandas(config["construction_time"],
                                   lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    capex = get_pandas(config["capex"],
                       lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    storage_capex = get_pandas(config["storage_capex"],
                               lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    fOM = get_pandas(config["fOM"],
                     lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    vOM = get_pandas(config["vOM"],
                     lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    charging_capex = get_pandas(config["charging_capex"],
                                lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    charging_opex = get_pandas(config["charging_opex"],
                               lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    eta_in = get_pandas(config["eta_in"],
                        lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    eta_out = get_pandas(config["eta_out"],
                         lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    capacity_ex = get_pandas(config["capacity_ex"],
                             lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    miscellaneous = get_pandas(config["miscellaneous"],
                               lambda x: pd.read_csv(x, index_col=0, header=None).squeeze("columns"))
    o = dict()
    o["epsilon"] = epsilon
    o["existing_capa"] = existing_capa
    o["capa_max"] = capa_max
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
    o["capacity_ex"] = capacity_ex
    o["miscellaneous"] = miscellaneous
    return o


def calculate_annuities_capex(miscellaneous, capex, construction_time, lifetime):
    # TODO: a checker, il y a peut-être un problème ici dans le rounding
    annuities = construction_time.copy()
    for i in annuities.index:
        annuities.at[i] = miscellaneous["discount_rate"] * capex[i] * (
                miscellaneous["discount_rate"] * construction_time[i] + 1) / (
                                  1 - (1 + miscellaneous["discount_rate"]) ** (-lifetime[i]))
    return annuities


def calculate_annuities_storage_capex(miscellaneous, storage_capex, construction_time, lifetime):
    # TODO: a checker, il y a peut-être un problème ici dans le rounding
    storage_annuities = storage_capex.copy()
    for i in storage_annuities.index:
        storage_annuities.at[i] = miscellaneous["discount_rate"] * storage_capex[i] * (
                miscellaneous["discount_rate"] * construction_time[i] + 1) / (
                                          1 - (1 + miscellaneous["discount_rate"]) ** (-lifetime[i]))
    return storage_annuities


def define_month_hours(first_month, nb_years, months_hours, hours_by_months):
    j = first_month + 1
    for i in range(2, 12 * nb_years + 1):
        hour = months_hours[i - 1][-1] + 1
        months_hours[i] = range(hour, hour + hours_by_months[j])
        j += 1
        if j == 13:
            j = 1
    return months_hours


def extract_summary(model, demand, H2_demand, CH4_demand, miscellaneous):
    # TODO: A CHANGER !!!
    summary = {}  # final dictionary for output
    # The whole demand in TWh
    demand_tot = sum(demand[hour] for hour in model.h) / 1000
    # The whole electricity demand for hydrogen in TWh
    hydrogen_tot = sum(H2_demand[hour] for hour in model.h) / 1000
    # The whole electricity demand for methane in TWh
    methane_tot = sum(CH4_demand[hour] for hour in model.h) / 1000
    # The whole generation in TWh
    gene_tot = sum(value(model.gene[gen, hour]) for hour in model.h for gen in model.gen) / 1000

    summary["demand_tot"] = demand_tot
    summary["hydrogen_tot"] = hydrogen_tot
    summary["methane_tot"] = methane_tot
    summary["gene_tot"] = gene_tot

    # Overall yearly energy generated by the technology in TWh
    # TODO: à mettre dans un autre dataframe
    gene_per_tec = {}
    for tec in model.tec:
        gene_per_tec[tec] = sum(value(model.gene[tec, hour]) for hour in model.h) / 1000

    summary.update(gene_per_tec)

    # The whole electricity input for storage in TWh
    nSTORAGE = {}
    for storage in model.str:
        for hour in model.h:
            nSTORAGE[(storage, hour)] = value(model.storage[storage, hour])

    # Electricity cost per MWh produced (euros/MWh)
    lcoe_sys = value(model.objective) * 1000 / gene_tot
    summary["lcoe_sys"] = lcoe_sys

    # Yearly storage related loss in % of power production and in TWh
    str_loss_percent = 100 * (
            sum(value(model.storage[storage, hour]) for storage in model.str for hour in model.h) -
            sum(gene_per_tec[storage] * 1000 for storage in model.str)) / (gene_tot * 1000)
    str_loss_TWh = gene_per_tec['electrolysis'] / miscellaneous['eta_electrolysis'] - hydrogen_tot / miscellaneous[
        'eta_electrolysis'] - gene_per_tec[
                       'h2_ccgt']  # TODO: je ne comprends pas pourquoi on a besoin de cette ligne alors qu'on n'en avait pas besoin avant quand on calculait en pourcentage
    # TODO: aussi, pourquoi on ne prend pas en compte la production des centrales CCGT ? et donc la méthanation et ses pertes ?
    for storage in model.str:
        if storage != 'hydrogen' and storage != 'methane':
            str_loss_TWh += sum(nSTORAGE[storage, hour] for hour in model.h) / 1000 - gene_per_tec[storage]

    summary["str_loss_percent"] = str_loss_percent
    summary["str_loss_TWh"] = str_loss_TWh

    # Load curtailment in % of power production and in TWh
    lc_percent = (100 * (
            gene_tot - demand_tot - hydrogen_tot / miscellaneous['eta_electrolysis']) / gene_tot) - str_loss_percent
    lc_TWh = (gene_tot - demand_tot - hydrogen_tot / miscellaneous['eta_electrolysis']) - str_loss_TWh

    summary["load_curtail_percent"] = lc_percent
    summary["load_curtail_TWh"] = lc_TWh
    summary_df = pd.Series(summary)
    return summary_df


def extract_capacities(model):
    """Extracts capacities for all technology in GW"""
    list_tec = list(model.tec)
    capacities = pd.Series(index=list_tec, dtype=float)

    for tec in list_tec:
        capacities.loc[tec] = value(model.capacity[tec])

    return capacities


def extract_energy_capacity(model):
    """Extracts energy capacity for all storage technology, in GWh"""
    list_str = list(model.str)
    energy_capacity = pd.Series(index=list_str, dtype=float)

    for tec in list_str:
        energy_capacity.loc[tec] = value(model.energy_capacity[tec])
    return energy_capacity


def extract_hourly_generation(model, demand):
    """Extracts hourly defined data, including demand, generation and storage"""
    list_tec = list(model.tec)
    list_storage_in = [e + "_in" for e in model.str]
    list_storage_charge = [e + "_charge" for e in model.str]
    list_columns = ["hour", "demand"] + list_tec + list_storage_in + list_storage_charge
    hourly_generation = pd.DataFrame(columns=list_columns)
    hourly_generation.loc[:, "hour"] = list(model.h)
    hourly_generation.loc[:, "demand"] = demand

    for tec in list_tec:
        hourly_generation.loc[:, tec] = value(model.gene[tec, :])
    for str, str_in in zip(list(model.str), list_storage_in):
        hourly_generation.loc[:, str_in] = value(model.storage[str, :])
    for str, str_charge in zip(list(model.str), list_storage_charge):
        hourly_generation.loc[:, str_charge] = value(model.stored[str, :])
    return hourly_generation


def extract_spot_price(model):
    """Extracts spot price"""
    list_columns = ["hour", "spot_price"]
    spot_price = pd.DataFrame(columns=list_columns)
    for h in spot_price.hour:
        spot_price.loc[h, "spot_price"] = - 1000000 * model.dual[model.adequacy_constraint[h]]
    return spot_price


def extract_supply_elec(model, nb_years):
    """Extracts yearly electricity supply per technology in TWh"""
    list_tec = list(model.gen_elec)
    electricity_supply = pd.Series(index=list_tec, dtype=float)

    for tec in list_tec:
        electricity_supply.loc[tec] = sum(value(model.gene[tec, hour]) for hour in model.h) / 1000 / nb_years
    return electricity_supply


def extract_use_elec(model, nb_years, miscellaneous):
    """Extracts yearly electricity use per technology in TWh"""
    list_tec = list(model.use_elec)
    electricity_use = pd.Series(index=list_tec, dtype=float)

    for tec in list_tec:
        if tec == 'electrolysis':  # for electrolysis, we need to use the efficiency factor to obtain TWhe
            electricity_use.loc[tec] = sum(value(model.gene[tec, hour]) for hour in model.h) / 1000 / nb_years / \
                                       miscellaneous['eta_electrolysis']
        else:
            electricity_use.loc[tec] = sum(value(model.storage[tec, hour]) for hour in model.h) / 1000 / nb_years
    return electricity_use


if __name__ == '__main__':
    with open('config.json') as file:
        config = json.load(file)
    m = ModelEOLES(name="test", config=config, nb_years=1)
    m.build_model()
    results, status, termination_condition = m.solve(solver_name="gurobi", save_results=False)
