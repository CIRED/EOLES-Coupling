from eoles.utils import get_config
from eoles.model_heat_coupling import ModelEOLES
import logging

from project.model import get_inputs  # imports from ResIRF package
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
    output = get_inputs(variables=['buildings'])
    buildings = output['buildings']
    heating_need = buildings.hourly_heating_need()

    new_index_datetime = [datetime.datetime(col[0], col[2], col[3], col[1]) for col in heating_need.index.values]
    new_index_hour = [int((e - datetime.datetime(2018, 1, 1, 0)).total_seconds() / 3600) for e in new_index_datetime]  # transform into number of hours
    heating_need.index = new_index_hour
    heating_need = heating_need.sort_index(ascending=True)

    # aggregation
    aggregation_archetype = ['Housing type', 'Heating system']  # argument for the function
    heating_need_groupby = heating_need.groupby(aggregation_archetype, axis=1).aggregate(np.sum)
    heating_need_groupby.columns = [" ".join(col) for col in heating_need_groupby.columns.values]

    # total
    heating_need_tot = heating_need.sum(axis=1)
    heating_need_tot = heating_need_tot * 1e-6  # GWh

    heating_need_tot = heating_need_tot * 0.8  # we reduce artificially the heating need by using expression
    # from utilization rate

    # dict_demand
    heating_demand = {
        "all_stock": heating_need_tot
    }

    m_scc = ModelEOLES(name="test", config=config, path="eoles/outputs", logger=logger, nb_years=1, heating_demand=heating_demand,
                       nb_linearize=3, social_cost_of_carbon=400, year=2050)
    m_scc.build_model()
    solver_results, status, termination_condition = m_scc.solve(solver_name="gurobi")


