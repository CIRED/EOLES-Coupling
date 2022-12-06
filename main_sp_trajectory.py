import logging

from project.model import social_planner

from eoles.model_heat_coupling import ModelEOLES
from eoles.utils import get_config, heating_hourly_profile, process_heating_need
from eoles.process_cost_efficiency import piecewise_linearization_cost_efficiency


if __name__ == '__main__':

    config = get_config(spec="greenfield")

    LOG_FORMATTER = '%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # consoler handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
    logger.addHandler(console_handler)

    hourly_profile_test = heating_hourly_profile(method="very_extreme", percentage=0)

    # TODO: attention au choix de hourly profile !!
    # ['Occupancy status', 'Income owner', 'Income tenant', 'Housing type', 'Heating system']
    dict_cost, dict_heat = social_planner(aggregation_archetype=['Wall class', "Housing type"], climate=2006,
                                          smooth=False, building_stock="medium_3",
                                          hourly_profile=hourly_profile_test)
    dict_heat = process_heating_need(dict_heat, climate=2006)
    linearized_renovation_costs, threshold_linearized_renovation_costs = piecewise_linearization_cost_efficiency(
        dict_cost, number_of_segments=3, plot=True, plot_tot=True)

    threshold_linearized_renovation_costs[
        threshold_linearized_renovation_costs < 0] = 0  # handles cases where the linearization was not perfect

    # TODO: il faudra ajouter les capacités pour charging et discharging
    year = [2025, 2030, 2035, 2040, 2045, 2050]
    new_capa_previous = 0
    new_capa_tot = 0
    for y in year:
        existing_capa_historical = None
        existing_capacity = existing_capa_historical + new_capa_tot  # existing capacity are equal to newly built
        # capacities over the whole time horizon + existing capacity (from before 2020)
        m_scc = ModelEOLES(name="test", config=config, path="eoles/outputs", logger=logger, nb_years=1,
                           heating_demand=dict_heat, nb_linearize=3,
                           linearized_renovation_costs=linearized_renovation_costs,
                           threshold_linearized_renovation_costs=threshold_linearized_renovation_costs,
                           existing_capacity=existing_capacity,
                           social_cost_of_carbon=0, year=y, scenario_cost=None, hp_hourly=True,
                           renov=None,
                           hourly_heat_gas=None)
        m_scc.build_model()
        solver_results, status, termination_condition = m_scc.solve(solver_name="gurobi")

        new_capa_previous = m_scc.capacities - m_scc.existing_capacity  # we get the newly installed capacities at time t
        new_capa_tot = new_capa_tot + new_capa_previous  # total newly built capacity over the time horizon