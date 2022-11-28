import pandas as pd
import scipy.optimize
# from project.model import social_planner
from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from project.model import social_planner

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


def piecewise_linear(x, x1, x2, y0, k1, k2, k3):
    # return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
    return np.piecewise(x, [(x < x1), (x >= x1) & (x < x2), x >= x2],
                        [lambda x: k1 * x + y0, lambda x: k2 * (x - x1) + y0 + k1 * (x1), lambda x: k3 * (x - x2) + k1 * (x1) + k2 * (x2 - x1) + y0])


def piecewise_linear_zero_interpolate(x, x1, x2, k1, k2, k3):
    # return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
    return np.piecewise(x, [(x < x1), (x >= x1) & (x < x2), x >= x2],
                        [lambda x: k1 * x, lambda x: k2 * (x - x1) + k1 * x1, lambda x: k3 * (x - x2) + k1 * x1 + k2 * (x2 - x1)])


def piecewise_linear_zero_interpolate_4segments(x, x1, x2, x3, k1, k2, k3, k4):
    # return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
    return np.piecewise(x, [(x < x1), (x >= x1) & (x < x2), (x >= x2) & (x < x3), x >= x3],
                        [lambda x: k1 * x, lambda x: k2 * (x - x1) + k1 * x1, lambda x: k3 * (x - x2) + k1 * x1 + k2 * (x2 - x1),
                         lambda x: k4 * (x - x3) + k1 * x1 + k2 * (x2 - x1) + k3 * (x3 - x2)])


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


def mse_piecewise_linear_zero_interpolate_4segments(params,x,y):
    x1, x2, x3, k1, k2, k3, k4 = params
    y_tilde = piecewise_linear_zero_interpolate_4segments(x,  x1, x2, x3, k1, k2, k3, k4)
    return np.mean(1/2*(y_tilde - y)**2)


def piecewise_linearization_cost_efficiency_4segments(dict_cost_efficiency_archetype, number_of_segments=4, n_init=50, plot=False, plot_tot=False):
    """Computes piecewise linearization for all costs functions given as input.

    :param dict_cost_efficiency_archetype: dict
        Includes all costs functions for all considered archetypes"""
    number_of_archetypes = len(list(dict_cost_efficiency_archetype.keys()))
    index_archetype = [f"{archetype}_{r}" for archetype in list(dict_cost_efficiency_archetype.keys()) for r in range(number_of_segments)]

    linearized_renovation_costs = pd.Series(index=index_archetype, dtype=float)
    threshold_linearized_renovation_costs = pd.Series(index=index_archetype, dtype=float)

    if plot and plot_tot:
        fig = plt.figure()
        ax = plt.subplot()

    for archetype in list(dict_cost_efficiency_archetype.keys()):
        cost_efficiency_stock = dict_cost_efficiency_archetype[archetype]
        cost_efficiency_stock = cost_efficiency_stock.reset_index()

        x = np.array(cost_efficiency_stock[['Consumption saved cumulated']]).reshape((-1,))
        y = np.array(cost_efficiency_stock[['Cost (Billion euro) cumulated']]).reshape((-1,))
        bound_x = 1
        x_max = np.max(x)
        x_min = np.min(x)

        if x_max < 0.4:
            bound_x = x_max * 1.1

        average_k = np.max(y) / np.max(x)

        cons = ({'type': 'ineq', 'fun': lambda x: x[1] - x[0]},
                {'type': 'ineq', 'fun': lambda x: x[3] - x[2]},
                {'type': 'ineq', 'fun': lambda x: x[4] - x[3]})

        # Attention, this only includes the option of 3 segments for the time being
        random_init_x1 = np.random.uniform(low=0, high=bound_x, size=(n_init, 1))
        random_init_x2 = np.random.uniform(low=0, high=bound_x, size=(n_init, 1))
        random_init_x = np.concatenate((random_init_x1, random_init_x2), axis=1)
        random_init_x = np.sort(random_init_x,
                                axis=1)  # we sort to make sure that in initialization we consider increasing coefficients
        random_init_k0 = np.random.uniform(low=0, high=0.8 * average_k, size=(n_init, 1))
        random_init_k1 = np.random.uniform(low=0.8 * average_k, high=3 * average_k, size=(n_init, 1))
        random_init_k2 = np.random.uniform(low=1.5 * average_k, high=4 * average_k, size=(n_init, 1))
        random_init_k = np.concatenate((random_init_k0, random_init_k1, random_init_k2), axis=1)
        random_init_k = np.sort(random_init_k,
                                axis=1)  # we sort to make sure that in initialization we consider increasing coefficients

        loss = 1e6
        best_fit = -1
        for i in range(n_init):
            x0_init = (random_init_x[i][0], random_init_x[i][1]) + (
            random_init_k[i][0], random_init_k[i][1], random_init_k[i][2])
            # k0, k1, k2 = random_init_k[0], random_init_k[1], random_init_k[2]
            # x0_init = (x1_init, x2_init) + (k0, k1, k2)
            res = scipy.optimize.minimize(lambda params: mse_piecewise_linear_zero_interpolate(params, x, y),
                                          x0=x0_init,
                                          bounds=((0, bound_x), (0, bound_x), (0, None), (0, None), (0, None)),
                                          constraints=cons)  # optimize the linear fit
            new_loss = res["fun"]
            if new_loss < loss:
                best_fit = i
                loss = new_loss

        x0_init = (random_init_x[best_fit][0], random_init_x[best_fit][1]) + (
            random_init_k[best_fit][0], random_init_k[best_fit][1], random_init_k[best_fit][2])

        res = scipy.optimize.minimize(lambda params: mse_piecewise_linear_zero_interpolate(params, x, y),
                                      x0=x0_init,
                                      bounds=((0, bound_x), (0, bound_x), (0, None), (0, None), (0, None)),
                                      constraints=cons)  # optimize the linear fit

        p = res['x']
        x1, x2, k1, k2, k3 = p[0], p[1], p[2], p[3], p[4]
        x0, x3 = 0, np.max(x)  # maximum value for renovation potential

        if plot:
            xd = np.linspace(0, x_max*1.1, 100)
            if not plot_tot:
                plt.plot(x, y, "o")  # plot initial points
            plt.plot(xd, piecewise_linear_zero_interpolate(xd, *p), label=archetype)
            if not plot_tot:
                plt.title(f"Renovation costs: {archetype}")
                plt.show()


        cutoff_points = np.array([x0, x1, x2, x3])
        marginal_costs = np.array([k1, k2, k3])
        for r in range(number_of_segments):
            linearized_renovation_costs[f"{archetype}_{r}"] = marginal_costs[r]
            threshold_linearized_renovation_costs[f"{archetype}_{r}"] = cutoff_points[r+1] - cutoff_points[r]

    if plot and plot_tot:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f"Renovation costs")
        plt.show()

    # linearized_renovation_costs.to_csv("inputs/linearized_renovation_costs.csv", header=False)
    # threshold_linearized_renovation_costs.to_csv("inputs/threshold_linearized_renovation_costs.csv", header=False)
    # TODO: peut être changer les chemins pour sauver les fichiers
    return linearized_renovation_costs, threshold_linearized_renovation_costs


def piecewise_linearization_cost_efficiency(dict_cost_efficiency_archetype, number_of_segments=3, n_init=50, plot=False, plot_tot=False):
    """Computes piecewise linearization for all costs functions given as input.
    :param dict_cost_efficiency_archetype: dict
        Includes all costs functions for all considered archetypes"""
    number_of_archetypes = len(list(dict_cost_efficiency_archetype.keys()))
    index_archetype = [f"{archetype}_{r}" for archetype in list(dict_cost_efficiency_archetype.keys()) for r in range(number_of_segments)]

    linearized_renovation_costs = pd.Series(index=index_archetype, dtype=float)
    threshold_linearized_renovation_costs = pd.Series(index=index_archetype, dtype=float)

    if plot and plot_tot:
        fig = plt.figure()
        ax = plt.subplot()

    for archetype in list(dict_cost_efficiency_archetype.keys()):
        cost_efficiency_stock = dict_cost_efficiency_archetype[archetype]
        cost_efficiency_stock = cost_efficiency_stock.reset_index()

        x = np.array(cost_efficiency_stock[['Consumption saved cumulated']]).reshape((-1,))
        y = np.array(cost_efficiency_stock[['Cost (Billion euro) cumulated']]).reshape((-1,))
        bound_x = 1
        x_max = np.max(x)
        x_min = np.min(x)

        if x_max < 0.4:
            bound_x = x_max * 1.1

        average_k = np.max(y) / np.max(x)

        cons = ({'type': 'ineq', 'fun': lambda x: x[1] - x[0]},
                {'type': 'ineq', 'fun': lambda x: x[3] - x[2]},
                {'type': 'ineq', 'fun': lambda x: x[4] - x[3]})

        # Attention, this only includes the option of 3 segments for the time being
        random_init_x1 = np.random.uniform(low=0, high=bound_x, size=(n_init, 1))
        random_init_x2 = np.random.uniform(low=0, high=bound_x, size=(n_init, 1))
        random_init_x = np.concatenate((random_init_x1, random_init_x2), axis=1)
        random_init_x = np.sort(random_init_x,
                                axis=1)  # we sort to make sure that in initialization we consider increasing coefficients
        random_init_k0 = np.random.uniform(low=0, high=0.8 * average_k, size=(n_init, 1))
        random_init_k1 = np.random.uniform(low=0.8 * average_k, high=3 * average_k, size=(n_init, 1))
        random_init_k2 = np.random.uniform(low=1.5 * average_k, high=4 * average_k, size=(n_init, 1))
        random_init_k = np.concatenate((random_init_k0, random_init_k1, random_init_k2), axis=1)
        random_init_k = np.sort(random_init_k,
                                axis=1)  # we sort to make sure that in initialization we consider increasing coefficients

        loss = 1e6
        best_fit = -1
        for i in range(n_init):
            x0_init = (random_init_x[i][0], random_init_x[i][1]) + (
            random_init_k[i][0], random_init_k[i][1], random_init_k[i][2])
            # k0, k1, k2 = random_init_k[0], random_init_k[1], random_init_k[2]
            # x0_init = (x1_init, x2_init) + (k0, k1, k2)
            res = scipy.optimize.minimize(lambda params: mse_piecewise_linear_zero_interpolate(params, x, y),
                                          x0=x0_init,
                                          bounds=((0, bound_x), (0, bound_x), (0, None), (0, None), (0, None)),
                                          constraints=cons)  # optimize the linear fit
            new_loss = res["fun"]
            if new_loss < loss:
                best_fit = i
                loss = new_loss

        x0_init = (random_init_x[best_fit][0], random_init_x[best_fit][1]) + (
            random_init_k[best_fit][0], random_init_k[best_fit][1], random_init_k[best_fit][2])

        res = scipy.optimize.minimize(lambda params: mse_piecewise_linear_zero_interpolate(params, x, y),
                                      x0=x0_init,
                                      bounds=((0, bound_x), (0, bound_x), (0, None), (0, None), (0, None)),
                                      constraints=cons)  # optimize the linear fit

        p = res['x']
        x1, x2, k1, k2, k3 = p[0], p[1], p[2], p[3], p[4]
        x0, x3 = 0, np.max(x)  # maximum value for renovation potential

        if plot:
            xd = np.linspace(0, x_max*1.1, 100)
            if not plot_tot:
                plt.plot(x, y, "o")  # plot initial points
            plt.plot(xd, piecewise_linear_zero_interpolate(xd, *p), label=archetype)
            if not plot_tot:
                plt.title(f"Renovation costs: {archetype}")
                plt.show()


        cutoff_points = np.array([x0, x1, x2, x3])
        marginal_costs = np.array([k1, k2, k3])
        for r in range(number_of_segments):
            linearized_renovation_costs[f"{archetype}_{r}"] = marginal_costs[r]
            threshold_linearized_renovation_costs[f"{archetype}_{r}"] = cutoff_points[r+1] - cutoff_points[r]

    if plot and plot_tot:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f"Renovation costs")
        plt.show()

    # linearized_renovation_costs.to_csv("inputs/linearized_renovation_costs.csv", header=False)
    # threshold_linearized_renovation_costs.to_csv("inputs/threshold_linearized_renovation_costs.csv", header=False)
    # TODO: peut être changer les chemins pour sauver les fichiers
    return linearized_renovation_costs, threshold_linearized_renovation_costs


if __name__ == '__main__':

    # # Test on one sample
    # cost_efficiency_stock = pd.read_csv("inputs/cost_efficiency_stock.csv", index_col=0)
    # cost_efficiency_stock = cost_efficiency_stock.reset_index()
    # sns.lineplot(cost_efficiency_stock, x='Consumption saved (%/initial) cumulated', y='Cost (Billion euro) cumulated')
    # plt.show()
    #
    # x = np.array(cost_efficiency_stock[['Consumption saved (%/initial) cumulated']]).reshape((-1,))
    # y = np.array(cost_efficiency_stock[['Cost (Billion euro) cumulated']]).reshape((-1,))
    #
    # # cons = ({'type': 'ineq', 'fun': lambda x: x[1] - x[0]},
    # #         {'type': 'ineq', 'fun': lambda x: x[4] - x[3]},
    # #         {'type': 'ineq', 'fun': lambda x: x[5] - x[4]})
    # #
    # # res = scipy.optimize.minimize(lambda params: mse_piecewise_linear(params, x, y),
    # #                               x0=(0.3, 0.6, 500, 700, 800, 900), bounds=((0, 1), (0, 1), (0, None), (0, None), (0, None), (0, None)),
    # #                               constraints=cons)
    #
    # cons = ({'type': 'ineq', 'fun': lambda x: x[1] - x[0]},
    #         {'type': 'ineq', 'fun': lambda x: x[3] - x[2]},
    #         {'type': 'ineq', 'fun': lambda x: x[4] - x[3]})
    #
    # res = scipy.optimize.minimize(lambda params: mse_piecewise_linear_zero_interpolate(params, x, y),
    #                               x0=(0.3, 0.6, 700, 800, 900), bounds=((0, 1), (0, 1), (0, None), (0, None), (0, None)),
    #                               constraints=cons)
    #
    # p = res['x']
    # xd = np.linspace(0, 0.8, 100)
    # plt.plot(x, y, "o")
    # plt.plot(xd, piecewise_linear_zero_interpolate(xd, *p))
    # plt.show()

    # # Automize saving piecewise approximation
    # cost_efficiency_stock = pd.read_csv("inputs/cost_efficiency_stock.csv", index_col=0)
    # dict_cost_efficiency_archetype = {
    #     "all_stock": cost_efficiency_stock
    # }
    #
    dict_cost, dict_heat = social_planner(aggregation_archetype=['Wall class', 'Housing type'], climate=2006,
                                          smooth=False, building_stock="medium_3")

    # dict_cost, dict_heat = social_planner(aggregation_archetype=['Income owner'], climate=2006,
    #                                       smooth=False)
    linearized_renovation_costs, threshold_linearized_renovation_costs = \
        piecewise_linearization_cost_efficiency(dict_cost, number_of_segments=3, n_init=300, plot=True, plot_tot=False)

    # df_heat = pd.DataFrame(dict_heat).stack().reset_index().rename(
    #     columns={"level_0": "date", "level_1": "archetype", 0: "heat_demand"})
    # sns.lineplot(df_heat, x="date", y="heat_demand", hue="archetype")
    # plt.show()

    # df_heat = pd.DataFrame(dict_heat).stack(
    #     level=[i for i in range(pd.DataFrame(dict_heat).columns.nlevels)]).reset_index()
    # df_heat["archetype"] = df_heat.apply(lambda row: row["level_1"] + ' ' + row["level_2"], axis=1)  # should be changed depending on level of multiindex
    # df_heat = df_heat.rename(columns={"level_0": "date", 0: "heat_demand"})
    # sns.lineplot(df_heat, x="date", y="heat_demand", hue="archetype")
    # plt.show()

    # cost_efficiency_stock = dict_cost[('Wood fuel-Performance boiler', 'Multi-family')]
    # cost_efficiency_stock = cost_efficiency_stock.reset_index()
    #
    # sns.scatterplot(cost_efficiency_stock, x='Consumption saved (%/initial) cumulated', y='Cost (Billion euro) cumulated')
    # plt.show()
    #
    # x = np.array(cost_efficiency_stock[['Consumption saved (%/initial) cumulated']]).reshape((-1,))
    # y = np.array(cost_efficiency_stock[['Cost (Billion euro) cumulated']]).reshape((-1,))
    #
    # # index_without_costly_renov = np.where(x < 0.6)
    # # x = x[index_without_costly_renov]
    # # y = y[index_without_costly_renov]
    #
    # average_k = np.max(y) / np.max(x)
    #
    # cons = ({'type': 'ineq', 'fun': lambda x: x[1] - x[0]},
    #         {'type': 'ineq', 'fun': lambda x: x[3] - x[2]},
    #         {'type': 'ineq', 'fun': lambda x: x[4] - x[3]})
    #
    # # Attention, this only includes the option of 3 segments for the time being
    # random_init = np.random.uniform(low=0, high=average_k*4, size=(20, 3))
    # random_init_k0 = np.random.uniform(low=0, high=0.8*average_k, size=(50,1))
    # random_init_k1 = np.random.uniform(low=0.5*average_k, high=3*average_k, size=(50,1))
    # random_init_k2 = np.random.uniform(low=1.5*average_k, high=4*average_k, size=(50,1))
    # random_init = np.concatenate((random_init_k0, random_init_k1, random_init_k2), axis=1)
    # random_init = np.sort(random_init, axis=1)
    #
    # loss = 1e6
    # best_fit = -1
    # for i in range(random_init.shape[0]):
    #     random_init_k = random_init[i]
    #     k0, k1, k2 = random_init_k[0], random_init_k[1], random_init_k[2]
    #     res = scipy.optimize.minimize(lambda params: mse_piecewise_linear_zero_interpolate(params, x, y),
    #                                   x0=(0.3, 0.6, k0, k1, k2),
    #                                   bounds=((0, 1), (0, 1), (0, None), (0, None), (0, None)),
    #                                   constraints=cons)  # optimize the linear fit
    #     new_loss = res["fun"]
    #     if new_loss < loss:
    #         best_fit = i
    #         loss = new_loss
    #
    # random_init_k = random_init[best_fit]
    # k0, k1, k2 = random_init_k[0], random_init_k[1], random_init_k[2]
    # res = scipy.optimize.minimize(lambda params: mse_piecewise_linear_zero_interpolate(params, x, y),
    #                               x0=(0.3, 0.6, k0, k1, k2),
    #                               bounds=((0, 1), (0, 1), (0, None), (0, None), (0, None)),
    #                               constraints=cons)  # optimize the linear fit
    #
    # p = res['x']
    # x1, x2, k1, k2, k3 = p[0], p[1], p[2], p[3], p[4]
    # x0, x3 = 0, np.max(x)  # maximum value for renovation potential
    # xd = np.linspace(0, 0.8, 100)
    # plt.plot(x, y, "o")
    # plt.plot(xd, piecewise_linear_zero_interpolate(xd, *p))
    # plt.show()
    #
    # linearized_renovation_costs, threshold_linearized_renovation_costs = piecewise_linearization_cost_efficiency(dict_cost_efficiency_archetype, number_of_segments=3)