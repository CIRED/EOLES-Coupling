import matplotlib

resources_data = dict()

colors_resirf = {
    "Owner-occupied": "lightcoral",
    "Privately rented": "chocolate",
    "Social-housing": "orange",
    "Single-family": "brown",
    "Multi-family": "darkolivegreen",
    "Single-family - Owner-occupied": "firebrick",
    "Multi-family - Owner-occupied": "salmon",
    "Single-family - Privately rented": "darkgreen",
    "Multi-family - Privately rented": "mediumseagreen",
    "Single-family - Social-housing": "darkorange",
    "Multi-family - Social-housing": "chocolate",
    "G": "black",
    "F": "dimgrey",
    "E": "grey",
    "D": "darkgrey",
    "C": "darkgreen",
    "B": "forestgreen",
    "A": "limegreen",
    "C1": "black",
    "C2": "darkred",
    "C3": "firebrick",
    "C4": "orangered",
    "C5": "lightcoral",
    "D1": "black",
    "D2": "maroon",
    "D3": "darkred",
    "D4": "brown",
    "D5": "firebrick",
    "D6": "orangered",
    "D7": "tomato",
    "D8": "lightcoral",
    "D9": "lightsalmon",
    "D10": "darksalmon",
    "Electricity": "darkorange",
    "Natural gas": "slategrey",
    "Oil fuel": "black",
    "Wood fuel": "saddlebrown",
    "Electricity-Heat pump water": "sandybrown",
    "Electricity-Heat pump air": "gold",
    "Electricity-Performance boiler": "darkorange",
    "Natural gas-Performance boiler": "slategrey",
    "Natural gas-Standard boiler": "grey",
    "Oil fuel-Performance boiler": "black",
    "Oil fuel-Standard boiler": "black",
    "Wood fuel-Performance boiler": "saddlebrown",
    "Wood fuel-Standard boiler": "saddlebrown",
    "VTA": "grey",
    "Energy taxes": "blue",
    "Energy vta": "red",
    "Taxes expenditure": "darkorange",
    "Subsidies heater": "orangered",
    "Subsidies insulation": "darksalmon",
    "Reduced tax": "darkolivegreen",
    "Cee": "tomato",
    "Cee tax": "red",
    "Cite": "blue",
    "Zero interest loan": "darkred",
    "Over cap": "grey",
    "Carbon tax": "rebeccapurple",
    "Mpr": "darkmagenta",
    "Mpr serenite": "violet",
    'Sub ad volarem': "darkorange",
    "Sub merit": "slategrey",
    "Sub obligation": "darkorange",
    "Existing": "tomato",
    "New": "lightgrey",
    "Renovation": "brown",
    "Construction": "dimgrey",
    'Investment': 'firebrick',
    'Embodied emission additional': 'darkgreen',
    'Cofp': 'grey',
    'Energy saving': 'darkorange',
    'Emission saving': 'forestgreen',
    'Well-being benefit': 'royalblue',
    'Health savings': 'blue',
    'Mortality reduction benefit': 'lightblue',
    'Social NPV': 'black',
    'Windows': 'royalblue',
    'Roof': 'darkorange',
    'Floor': 'grey',
    'Wall': 'darkslategrey',
    "Consumption saving heater (TWh)": '#029E73',
    "Consumption saving insulation (TWh)": '#FBAFE4',
    "Consumption saving heater (TWh/year)": '#029E73',
    "Consumption saving insulation (TWh/year)": '#FBAFE4',
    "Consumption saving heater cumulated (TWh)": '#029E73',
    "Consumption saving insulation cumulated (TWh)": '#FBAFE4'
}

colors_eoles = {
    "offshore_f": '#0173B2',
    "offshore floating wind": '#0173B2',
    "offshore_g": '#56B4E9',
    "offshore": '#56B4E9',
    "wind": '#029E73',
    "offshore ground wind": '#56B4E9',
    "onshore": '#029E73',
    "pv_g": '#DE8F05',
    "pv": '#DE8F05',
    "pv_c": '#ECE133',
    "river": '#FBAFE4',
    "lake": '#CC78BC',
    "hydro": '#FBAFE4',
    "phs": '#949494',
    "phs_in": '#CC78BC',
    "battery1": "blue",
    "battery4": "blue",
    "battery": "blue",
    "battery_in": "blue",
    "battery_discharge": "#ECE133",
    "ocgt": "#D55E00",
    "ccgt": "#ECE133",
    "methane": "black",
    "peaking_plants": "brown",
    "h2_ccgt": "sienna",
    "natural_gas": "#949494",
    "methanization": "#CA9161",
    "nuclear": "#D55E00",
    "electrolysis": "#56B4E9",
    "methanation": "#029E73",
    "pyrogazification": "#0173B2",
    "Average electricity price": "#0173B2",
    "Average CH4 price": "#D55E00",
    "Average H2 price": "#CC78BC",
    "LCOE electricity": "#56B4E9",
    "LCOE electricity volume": "#029E73",
    "LCOE electricity value": "#DE8F05",
    "LCOE CH4": 'red',
    "LCOE CH4 value": '#ECE133',
    "LCOE CH4 volume": '#56B4E9',
    "LCOE CH4 noSCC": 'grey',
    "LCOE CH4 volume noSCC": 'black',
    "Gas for heating": "slategrey",
    "Electricity for heating": "darkorange",
    # "Investment heater (Billion euro)" : "firebrick",
    # "Investment insulation (Billion euro)" : "#CC78BC",
    # "Subsidies heater (Billion euro)" : "orangered",
    # "Subsidies insulation (Billion euro)": "#ECE133",
    "Health cost (Billion euro)": "blue",
    "Annualized electricity system costs": '#56B4E9',
    "Annualized investment heater costs": '#029E73',
    "Annualized investment insulation costs": '#FBAFE4',
    "Annualized health costs": '#0173B2',
    "Investment electricity costs": '#56B4E9',
    "Functionment costs": 'yellow',
    "Investment heater costs": '#029E73',
    "Investment insulation costs": '#FBAFE4',
    "Health costs": '#0173B2',
    "Stock Wood fuel (Million)": 'saddlebrown',
     "Stock Oil fuel (Million)": '#949494',
    "Stock Natural gas (Million)": '#CC78BC',
    "Stock Electricity (Million)": '#029E73',
    "Stock Heat pump (Million)": '#DE8F05',
    "Heat pump air": '#ECE133',
    "Heat pump water": '#D55E00',
    "Heat pump": "#ECE133",
    "Electric heating": 'red',
    "Natural gas": '#949494',
    "Oil fuel": "black",
    "Wood fuel": 'saddlebrown',
    "Investment heater (Billion euro)": "#029E73",
    "Investment insulation (Billion euro)": "#FBAFE4",
    "Investment heater WT (Billion euro)": "#029E73",
    "Investment insulation WT (Billion euro)": "#FBAFE4",
    "Subsidies heater (Billion euro)": "#029E73",
    "Subsidies insulation (Billion euro)": "#FBAFE4"

}

resources_data["colors_resirf"] = colors_resirf
resources_data["colors_eoles"] = colors_eoles