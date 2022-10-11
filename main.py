from eoles.model import ModelEOLES
import json
import time
from eoles.utils import get_config, write_output, plot_capacities  # Ecrire plutôt quelque chose avec un package parent ?

if __name__ == '__main__':
    config = get_config()
    m = ModelEOLES(name="test", config=config, path="eoles/outputs", nb_years=1, residential=True, social_cost_of_carbon=100)
    t1 = time.time()
    m.build_model()
    solver_results, status, termination_condition = m.solve(solver_name="gurobi")
    t2 = time.time()
    print(f'Time : {t2 - t1: .1f}')
    # write_output(m.results, folder="outputs/test2006_3")

    # plot_capacities(m.capacities)
