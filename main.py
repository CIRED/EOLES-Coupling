from eoles.model import ModelEOLES
import json
from eoles.utils import write_output, plot_capacities  # Ecrire plutôt quelque chose avec un package parent ?

if __name__ == '__main__':
    with open('config.json') as file:
        config = json.load(file)
    m = ModelEOLES(name="test", config=config, nb_years=1)
    m.build_model()
    solver_results, status, termination_condition = m.solve(solver_name="gurobi")
    # write_output(m.results, folder="outputs/test2006_3")

    # plot_capacities(m.capacities)
