import numpy as np
import pandas as pd

from eoles.model import ModelEOLES
import json
import time
import logging
from eoles.utils import get_config, write_output, plot_capacities  # Ecrire plutôt quelque chose avec un package parent ?
from matplotlib import pyplot as plt
import seaborn as sns


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

    scc = 100
    m_scc = ModelEOLES(name="test", config=config, path="eoles/outputs", logger=logger, nb_years=1, total_demand_RTE=580*1e3,
                       residential_heating_demand_RTE=33*1e3, residential=True, social_cost_of_carbon=scc)
    t1 = time.time()
    m_scc.build_model()
    solver_results, status, termination_condition = m_scc.solve(solver_name="gurobi")
    t2 = time.time()
    print(f'Time : {t2 - t1: .1f}')
    # write_output(m.results, folder="outputs/test2006_3")

    # plot_capacities(m.capacities)

    # list_scc = [0, 100, 200, 300, 400, 500]
    # list_social_cost = []
    # list_technical_cost = []
    # list_emissions = []
    # list_primary_production = []
    # reindex_primary_prod = ["offshore_f", "offshore_g", "onshore", "pv_g", "pv_c", "river", "lake", "nuc", "biogas1", "biogas2", "pyrogazification", "natural_gas"]
    # for scc in list_scc:
    #     print(f"Social cost of carbon: {scc}")
    #     m_scc = ModelEOLES(name="test", config=config, path="eoles/outputs", logger=logger, nb_years=1,
    #                        residential=True,
    #                        social_cost_of_carbon=scc)
    #     t1 = time.time()
    #     m_scc.build_model()
    #     _, _, _ = m_scc.solve(solver_name="gurobi")
    #     t2 = time.time()
    #     print(f'Time : {t2 - t1: .1f}')
    #
    #     list_social_cost.append(m_scc.objective)
    #     list_technical_cost.append(m_scc.technical_cost)
    #     list_emissions.append(m_scc.emissions)
    #
    #     list_primary_production.append(m_scc.primary_generation.reindex(reindex_primary_prod).to_list())
    #
    # list_primary_production = np.transpose(np.array(list_primary_production)).tolist()
    #
    # # set seaborn style
    # sns.set_theme()
    #
    # plt.plot(list_scc, list_social_cost)
    # plt.ylim(0, 35)
    # plt.ylabel("Annual cost in €bn/year")
    # plt.xlabel("Social cost of carbon")
    # plt.show()
    #
    # fig = plt.figure()
    # ax = plt.subplot(111)
    # ax.stackplot(list_scc, list_primary_production, labels=reindex_primary_prod)
    # ax.set_ylabel("Yearly primary production in TWh")
    # ax.set_xlabel("Social cost of carbon")
    # # Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()