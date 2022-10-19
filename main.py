from eoles.model import ModelEOLES
import json
import time
import logging
from eoles.utils import get_config, write_output, plot_capacities  # Ecrire plutôt quelque chose avec un package parent ?

if __name__ == '__main__':
    config = get_config()

    LOG_FORMATTER = '%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # consoler handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
    logger.addHandler(console_handler)

    # m_scc1000 = ModelEOLES(name="test", config=config, path="eoles/outputs", logger=logger, nb_years=1, residential=True,
    #                social_cost_of_carbon=1000)
    # t1 = time.time()
    # m_scc1000.build_model()
    # solver_results, status, termination_condition = m_scc1000.solve(solver_name="gurobi")
    # t2 = time.time()
    # print(f'Time : {t2 - t1: .1f}')

    scc = 300
    m_scc = ModelEOLES(name="test", config=config, path="eoles/outputs", logger=logger, nb_years=1, residential=True,
                   social_cost_of_carbon=scc)
    t1 = time.time()
    m_scc.build_model()
    solver_results, status, termination_condition = m_scc.solve(solver_name="gurobi")
    t2 = time.time()
    print(f'Time : {t2 - t1: .1f}')
    # write_output(m.results, folder="outputs/test2006_3")

    # plot_capacities(m.capacities)
