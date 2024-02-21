# Description: This file contains the mapping between the scenarios and the configuration files

map_values_eoles = {
    'Biogas-': 'S2',
    'Nuc-': 'N1nuc',
    'Ren-': 'N1ren2',
    'Sufficiency': 'Sobriete',
    'Reindustrialisation': 'Reindustrialisation',
    'cold': {'load_factors': 'vre_profiles_2012',
              'lake_inflows': 'lake_2012',
              'nb_years': 1,
              'input_years': [2012]}
}

map_values_resirf = {
    'Ban': {
        "restriction_gas": {
            "start": 2025,
            "end": 2051,
            "value": "Natural gas",
            "policy": "restriction_energy"
    }},
    'NoPolicy':  {
        "file": "project/input/policies/policies_calibration.json"
    },
    'Learning+': {
          "technical_progress": {
            "heater": {
              "activated": True,
              "start": 2019,
              "end": 2035,
              "value_end": -0.5
            }
          }
        },
    'Learning-': {
          "technical_progress": {
            "heater": {
              "activated": False,
              "start": 2019,
              "end": 2035,
              "value_end": -0.5
            }
          }
    },
    'Elasticity-': {
          "scale": {
            "option": "price_elasticity",
            "target": -0.5
          },
    },
    'Elasticity+': {
      "scale": {
        "option": "price_elasticity",
        "target": 1.5
      }
    },
    'ProfileFlat': {
            "hourly_profile": "project/input/technical/hourly_profile_flat.csv"
        },
    'PriceGas+':  {
        "energy_prices": {
        "ini": "project/input/energy/energy_prices_wt_ame2021.csv",
        "rate": {
          "Electricity": 0.0135,
          "Natural gas": 0.0104 * 2,
          "Wood fuel": 0.0127,
          "Oil fuel": 0.0173,
          "Heating": 0.0104
        }
      }},
    'PriceWood+': {
        "energy_prices": {
            "ini": "project/input/energy/energy_prices_wt_ame2021.csv",
            "rate": {
                "Electricity": 0.0135,
                "Natural gas": 0.0104,
                "Wood fuel": 0.0127 * 2,
                "Oil fuel": 0.0173,
                "Heating": 0.0104
            }
        }},
}
# concatenate the two dictionaries
map_values = {**map_values_eoles, **map_values_resirf}

map_scenarios_to_configs = {
    'biogas': ['supply', 'biomass_potential_scenario'],
    'capacity_ren': ['supply', 'maximum_capacity_scenario'],
    'capacity_nuc': ['supply', 'maximum_capacity_scenario'],
    'weather': ['supply', 'weather'],
    'ban': ['policies'],
    'insulation': ['policies'],
    'demand': ['supply', 'demand_scenario'],
    'technical': ['demand'],
    'learning': ['demand', 'technical'],
    'profile': ['demand', 'technical'],
    'elasticity': ['demand', 'switch_heater'],
    'gasprices': ['demand', 'energy'],
    'woodprices': ['demand', 'energy']
}
