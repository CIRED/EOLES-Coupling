import pandas as pd
import scipy.optimize
from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def piecewise_linear(x, x1, x2, y0, k1, k2, k3):
    # return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
    return np.piecewise(x, [(x < x1), (x >= x1) & (x < x2), x >= x2],
                        [lambda x: k1 * x + y0, lambda x: k2 * (x - x1) + y0 + k1 * (x1), lambda x: k3 * (x - x2) + k1 * (x1) + k2 * (x2 - x1) + y0])


def piecewise_linear_zero_interpolate(x, x1, x2, k1, k2, k3):
    # return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
    return np.piecewise(x, [(x < x1), (x >= x1) & (x < x2), x >= x2],
                        [lambda x: k1 * x, lambda x: k2 * (x - x1) + k1 * x1, lambda x: k3 * (x - x2) + k1 * x1 + k2 * (x2 - x1)])


def piecewise_linear_zero_interpolate_generic(x, x_cutoffs, k_coefs):
    # return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
    # TODO: comment faire pour que cette fonction ne prenne pas en argument mais que je le transforme en array dans ma fonction ensuite ??
    n = x_cutoffs.shape[0]
    x_cutoffs.insert(0, 0)  # we add initial value
    assert k_coefs.shape[0] == n +1
    return np.piecewise(x, [(x >= x_cutoffs[i]) & (x < x_cutoffs[i + 1]) for i in range(n-1)] + [x >= x_cutoffs[-1]],
                        [lambda x, k=i: k_coefs[i] * (x - x_cutoffs[i-1]) + sum(k_coefs[j] * x_cutoffs[j] for j in range(1, i)) for i in range(1, n+1)]
                        )


def mse_piecewise_linear_zero_interpolate(params,x,y):
    x1, x2, k1, k2, k3 = params
    y_tilde = piecewise_linear_zero_interpolate(x,  x1, x2, k1, k2, k3)
    return np.mean(1/2*(y_tilde - y)**2)


def piecewise_linearization_cost_efficiency(dict_cost_efficiency_archetype, number_of_segments=3):
    number_of_archetypes = len(list(dict_cost_efficiency_archetype.keys()))
    index_archetype = [f"{archetype}_{r}" for archetype in list(dict_cost_efficiency_archetype.keys()) for r in range(number_of_segments)]

    linearized_renovation_costs = pd.Series(index=index_archetype, dtype=float)
    threshold_linearized_renovation_costs = pd.Series(index=index_archetype, dtype=float)
    for archetype in list(dict_cost_efficiency_archetype.keys()):
        cost_efficiency_stock = dict_cost_efficiency_archetype[archetype]
        cost_efficiency_stock = cost_efficiency_stock.reset_index()

        x = np.array(cost_efficiency_stock[['Consumption saved (%/initial) cumulated']]).reshape((-1,))
        y = np.array(cost_efficiency_stock[['Cost (Billion euro) cumulated']]).reshape((-1,))

        cons = ({'type': 'ineq', 'fun': lambda x: x[1] - x[0]},
                {'type': 'ineq', 'fun': lambda x: x[3] - x[2]},
                {'type': 'ineq', 'fun': lambda x: x[4] - x[3]})

        # Attention, this only includes the option of 3 segments for the time being
        res = scipy.optimize.minimize(lambda params: mse_piecewise_linear_zero_interpolate(params, x, y),
                                      x0=(0.3, 0.6, 700, 800, 900),
                                      bounds=((0, 1), (0, 1), (0, None), (0, None), (0, None)),
                                      constraints=cons)  # optimize the linear fit
        p = res['x']
        x1, x2, k1, k2, k3 = p[0], p[1], p[2], p[3], p[4]
        x0, x3 = 0, np.max(x)  # maximum value for renovation potential

        cutoff_points = np.array([x0, x1, x2, x3])
        marginal_costs = np.array([k1, k2, k3])
        for r in range(number_of_segments):
            linearized_renovation_costs[f"{archetype}_{r}"] = marginal_costs[r]
            threshold_linearized_renovation_costs[f"{archetype}_{r}"] = cutoff_points[r+1] - cutoff_points[r]

    linearized_renovation_costs.to_csv("inputs/linearized_renovation_costs.csv", header=False)
    threshold_linearized_renovation_costs.to_csv("inputs/threshold_linearized_renovation_costs.csv", header=False)
    # TODO: peut être changer les chemins pour sauver les fichiers
    return linearized_renovation_costs, threshold_linearized_renovation_costs


if __name__ == '__main__':

    # Test on one sample
    cost_efficiency_stock = pd.read_csv("inputs/cost_efficiency_stock.csv", index_col=0)
    cost_efficiency_stock = cost_efficiency_stock.reset_index()
    sns.lineplot(cost_efficiency_stock, x='Consumption saved (%/initial) cumulated', y='Cost (Billion euro) cumulated')
    plt.show()

    x = np.array(cost_efficiency_stock[['Consumption saved (%/initial) cumulated']]).reshape((-1,))
    y = np.array(cost_efficiency_stock[['Cost (Billion euro) cumulated']]).reshape((-1,))

    # cons = ({'type': 'ineq', 'fun': lambda x: x[1] - x[0]},
    #         {'type': 'ineq', 'fun': lambda x: x[4] - x[3]},
    #         {'type': 'ineq', 'fun': lambda x: x[5] - x[4]})
    #
    # res = scipy.optimize.minimize(lambda params: mse_piecewise_linear(params, x, y),
    #                               x0=(0.3, 0.6, 500, 700, 800, 900), bounds=((0, 1), (0, 1), (0, None), (0, None), (0, None), (0, None)),
    #                               constraints=cons)

    cons = ({'type': 'ineq', 'fun': lambda x: x[1] - x[0]},
            {'type': 'ineq', 'fun': lambda x: x[3] - x[2]},
            {'type': 'ineq', 'fun': lambda x: x[4] - x[3]})

    res = scipy.optimize.minimize(lambda params: mse_piecewise_linear_zero_interpolate(params, x, y),
                                  x0=(0.3, 0.6, 700, 800, 900), bounds=((0, 1), (0, 1), (0, None), (0, None), (0, None)),
                                  constraints=cons)

    p = res['x']
    xd = np.linspace(0, 0.8, 100)
    plt.plot(x, y, "o")
    plt.plot(xd, piecewise_linear_zero_interpolate(xd, *p))
    plt.show()

    # Automize saving piecewise approximation
    cost_efficiency_stock = pd.read_csv("inputs/cost_efficiency_stock.csv", index_col=0)
    dict_cost_efficiency_archetype = {
        "all_stock": cost_efficiency_stock
    }

    linearized_renovation_costs, threshold_linearized_renovation_costs = piecewise_linearization_cost_efficiency(dict_cost_efficiency_archetype, number_of_segments=3)