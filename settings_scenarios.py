# Description: This file contains the mapping between the scenarios and the configuration files

map_values_eoles = {
    'Biogas-': 'S2',
    'Nuc-': 'N1nuc',
    'Ren-': 'N1ren',
    'Offshore-': 'N1offshore',
    'Onshore-': 'N1onshore',
    'PV-': 'N1pv',
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
        "target": 1.2
      }
    },
    'COP+' : {"temp_sink": "project/input/technical/temp_sink_progress.csv"},
    'ProfileFlat': {
            "hourly_profile": "project/input/technical/hourly_profile_flat.csv"
        },
    'PriceGas+':  {
        "eoles": {
            "rate": {
                "natural_gas": 0.0158 * 2,
            }
        },
        "resirf": {
            "rate": {
                "Natural gas": 0.0104 * 2,
            }
      }},
    'PriceWood+': {
        "eoles": {
            "rate": {
                "wood": 0.0126 * 2,
            }
        },
        "resirf": {
            "rate": {
                "Wood fuel": 0.0127 * 2,
            }
        }},
}
# concatenate the two dictionaries
map_values = {**map_values_eoles, **map_values_resirf}

map_scenarios_to_configs = {
    'biogas': ['supply', 'biomass_potential_scenario'],
    'capacity_ren': ['supply', 'maximum_capacity_scenario'],
    'capacity_nuc': ['supply', 'maximum_capacity_scenario'],
    'capacity_offshore': ['supply', 'maximum_capacity_scenario'],
    'capacity_onshore': ['supply', 'maximum_capacity_scenario'],
    'capacity_pv': ['supply', 'maximum_capacity_scenario'],
    'weather': ['supply', 'weather'],
    'ban': ['policies'],
    'insulation': ['policies'],
    'demand': ['supply', 'demand_scenario'],
    'technical': ['demand'],
    'learning': ['demand', 'technical'],
    'cop': ['demand', 'technical'],
    'profile': ['demand', 'technical'],
    'elasticity': ['demand', 'switch_heater'],
    'gasprices': ['prices', 'prices'],
    'woodprices': ['prices', 'prices']
}
