from eoles.utils import get_config, process_heating_need
from eoles.process_cost_efficiency import piecewise_linearization_cost_efficiency
from eoles.model_heat_coupling import ModelEOLES
import logging

from project.model import get_inputs, social_planner  # imports from ResIRF package
import datetime
import numpy as np


if __name__ == '__main__':
    config = get_config(spec="greenfield")

    LOG_FORMATTER = '%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # consoler handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
    logger.addHandler(console_handler)

    # load heating demand
    # output = get_inputs(variables=['buildings'])
    # buildings = output['buildings']
    # heating_need = buildings.heating_need(hourly=True, climate=2006, smooth=False)
    # heating_need = process_heating_need(heating_need, climate=2006)
    #
    # # aggregation
    # aggregation_archetype = ['Housing type', 'Heating system']  # argument for the function
    # heating_need_groupby = heating_need.groupby(aggregation_archetype, axis=1).aggregate(np.sum)
    # heating_need_groupby.columns = [" ".join(col) for col in heating_need_groupby.columns.values]
    #
    # # total
    # heating_need_tot = heating_need.sum(axis=1)
    # heating_need_tot = heating_need_tot * 1e-6  # GWh
    #
    # heating_need_tot = heating_need_tot * 0.8  # we reduce artificially the heating need by using expression
    # # from utilization rate
    #
    # # dict_demand
    # heating_demand = {
    #     "all_stock": heating_need_tot
    # }

    dict_cost, dict_heat = social_planner(climate=2006, smooth=False)
    dict_heat = process_heating_need(dict_heat, climate=2006)
    linearized_renovation_costs, threshold_linearized_renovation_costs = piecewise_linearization_cost_efficiency(
        dict_cost, number_of_segments=3, plot=False)

    dict_heat = {"all_stock": dict_heat["global"]* 1e-6*0.8}

    # TODO: modifier le code pour que mon code puisse bien prendre en compte des fichiers de coûts qui lui sont fournis.

    m_scc = ModelEOLES(name="test", config=config, path="eoles/outputs", logger=logger, nb_years=1, heating_demand=dict_heat,
                       nb_linearize=3, linearized_renovation_costs=linearized_renovation_costs,
                       threshold_linearized_renovation_costs=threshold_linearized_renovation_costs,
                       social_cost_of_carbon=400, year=2050)
    m_scc.build_model()
    solver_results, status, termination_condition = m_scc.solve(solver_name="gurobi")


