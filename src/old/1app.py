# @ Create Time: 2024-02-28 12:14:28.059962

from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objs as go
import numpy as np

app = Dash(__name__, title="MyDashApp")

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

# Load the dataset
file_path = './relevant.csv'
data = pd.read_csv(file_path)

# Handle missing or 'nan' values
data.fillna('nan', inplace=True)

# Encode categorical variables
label_encoders = {}
columns_to_encode = ['input|Opt-AboveGradeWall', 'input|Opt-Ceilings', 'input|Opt-DHWSystem', 
                     'input|Opt-FoundationWallExtIns', 'input|Opt-Heating-Cooling', 'input|Opt-Windows']

for column in columns_to_encode:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

# Function to get estimates
def get_estimates(opt_abovegradewall, opt_ceilings, opt_dhwsystem, opt_foundationwall, opt_heatingcooling, opt_windows):
    # Encode the user's choices
    opt_abovegradewall_encoded = label_encoders['input|Opt-AboveGradeWall'].transform([opt_abovegradewall])[0]
    opt_ceilings_encoded = label_encoders['input|Opt-Ceilings'].transform([opt_ceilings])[0]
    opt_dhwsystem_encoded = label_encoders['input|Opt-DHWSystem'].transform([opt_dhwsystem])[0]
    opt_foundationwall_encoded = label_encoders['input|Opt-FoundationWallExtIns'].transform([opt_foundationwall])[0]
    opt_heatingcooling_encoded = label_encoders['input|Opt-Heating-Cooling'].transform([opt_heatingcooling])[0]
    opt_windows_encoded = label_encoders['input|Opt-Windows'].transform([opt_windows])[0]

    # Find the corresponding row in the dataset
    row = data[(data['input|Opt-AboveGradeWall'] == opt_abovegradewall_encoded) &
               (data['input|Opt-Ceilings'] == opt_ceilings_encoded) &
               (data['input|Opt-DHWSystem'] == opt_dhwsystem_encoded) &
               (data['input|Opt-FoundationWallExtIns'] == opt_foundationwall_encoded) &
               (data['input|Opt-Heating-Cooling'] == opt_heatingcooling_encoded) &
               (data['input|Opt-Windows'] == opt_windows_encoded)]

    if not row.empty:
        total_cost = row['cost-estimates|total'].values[0]
        utility_bill = row['output|Util-Bill-gross'].values[0]
        return total_cost, utility_bill
    else:
        return None, None

# Function to find the optimal combination
def get_optimal_combination():
    # Ensure numeric conversion of cost and utility bill columns, forcing non-numeric values to NaN
    data['cost-estimates|total'] = pd.to_numeric(data['cost-estimates|total'], errors='coerce')
    data['output|Util-Bill-gross'] = pd.to_numeric(data['output|Util-Bill-gross'], errors='coerce')
    
    # Drop rows with NaN values in these columns
    data.dropna(subset=['cost-estimates|total', 'output|Util-Bill-gross'], inplace=True)
    
    # Calculate the total cost with the multiplier for the utility bill
    data['total_cost_with_multiplier'] = data['cost-estimates|total'] + (data['output|Util-Bill-gross'] * 15)
    
    # Find the row with the minimum total cost with multiplier
    optimal_row = data.loc[data['total_cost_with_multiplier'].idxmin()]

    # Decode the optimal options
    opt_abovegradewall = label_encoders['input|Opt-AboveGradeWall'].inverse_transform([optimal_row['input|Opt-AboveGradeWall']])[0]
    opt_ceilings = label_encoders['input|Opt-Ceilings'].inverse_transform([optimal_row['input|Opt-Ceilings']])[0]
    opt_dhwsystem = label_encoders['input|Opt-DHWSystem'].inverse_transform([optimal_row['input|Opt-DHWSystem']])[0]
    opt_foundationwall = label_encoders['input|Opt-FoundationWallExtIns'].inverse_transform([optimal_row['input|Opt-FoundationWallExtIns']])[0]
    opt_heatingcooling = label_encoders['input|Opt-Heating-Cooling'].inverse_transform([optimal_row['input|Opt-Heating-Cooling']])[0]
    opt_windows = label_encoders['input|Opt-Windows'].inverse_transform([optimal_row['input|Opt-Windows']])[0]

    total_cost = optimal_row['cost-estimates|total']
    utility_bill = optimal_row['output|Util-Bill-gross']

    return opt_abovegradewall, opt_ceilings, opt_dhwsystem, opt_foundationwall, opt_heatingcooling, opt_windows, total_cost, utility_bill

# Get initial default values
default_opt_abovegradewall = 'Stud_2x6_R20'
default_opt_ceilings = 'CeilR40'
default_opt_dhwsystem = 'elec_heatpump_ef2.30'
default_opt_foundationwall = 'xps4inEffR20'
default_opt_heatingcooling = 'ASHP'
default_opt_windows = 'NC-3g-LG-u0.85'

# Define layout
app.layout = html.Div(children=[
    html.H1(children='Energy Cost Estimator'),

    html.Div(children='''Select your options:'''),

    dcc.Dropdown(
        id='abovegradewall-dropdown',
        options=[{'label': val, 'value': val} for val in label_encoders['input|Opt-AboveGradeWall'].classes_],
        value=default_opt_abovegradewall
    ),

    dcc.Dropdown(
        id='ceilings-dropdown',
        options=[{'label': val, 'value': val} for val in label_encoders['input|Opt-Ceilings'].classes_],
        value=default_opt_ceilings
    ),

    dcc.Dropdown(
        id='dhwsystem-dropdown',
        options=[{'label': val, 'value': val} for val in label_encoders['input|Opt-DHWSystem'].classes_],
        value=default_opt_dhwsystem
    ),

    dcc.Dropdown(
        id='foundationwall-dropdown',
        options=[{'label': val, 'value': val} for val in label_encoders['input|Opt-FoundationWallExtIns'].classes_],
        value=default_opt_foundationwall
    ),

    dcc.Dropdown(
        id='heatingcooling-dropdown',
        options=[{'label': val, 'value': val} for val in label_encoders['input|Opt-Heating-Cooling'].classes_],
        value=default_opt_heatingcooling
    ),

    dcc.Dropdown(
        id='windows-dropdown',
        options=[{'label': val, 'value': val} for val in label_encoders['input|Opt-Windows'].classes_],
        value=default_opt_windows
    ),

    html.Button('Calculate', id='calculate-button', n_clicks=0),

    html.Div(id='output-container', children='Select options and click "Calculate" to see results'),

    html.H2(children='Recommended Optimal Combination'),
    html.Div(id='optimal-combination', children='Loading optimal combination...'),

    dcc.Graph(id='comparison-graph')
])

@app.callback(
    Output('output-container', 'children'),
    [Input('calculate-button', 'n_clicks')],
    [State('abovegradewall-dropdown', 'value'),
     State('ceilings-dropdown', 'value'),
     State('dhwsystem-dropdown', 'value'),
     State('foundationwall-dropdown', 'value'),
     State('heatingcooling-dropdown', 'value'),
     State('windows-dropdown', 'value')]
)
def update_output(n_clicks, opt_abovegradewall, opt_ceilings, opt_dhwsystem, opt_foundationwall, opt_heatingcooling, opt_windows):
    if n_clicks > 0:
        total_cost, utility_bill = get_estimates(opt_abovegradewall, opt_ceilings, opt_dhwsystem, opt_foundationwall, opt_heatingcooling, opt_windows)
        if total_cost is not None and utility_bill is not None:
            return f"Total Cost Estimates: {total_cost}, Gross Utility Bills (15 years): {utility_bill * 15}"
        else:
            return "No data available for the selected combination"
    return 'Select options and click "Calculate" to see results'


@app.callback(
    [Output('optimal-combination', 'children'),
     Output('comparison-graph', 'figure')],
    [Input('calculate-button', 'n_clicks')],
    [State('abovegradewall-dropdown', 'value'),
     State('ceilings-dropdown', 'value'),
     State('dhwsystem-dropdown', 'value'),
     State('foundationwall-dropdown', 'value'),
     State('heatingcooling-dropdown', 'value'),
     State('windows-dropdown', 'value')]
)
def display_optimal_combination(n_clicks, opt_abovegradewall, opt_ceilings, opt_dhwsystem, opt_foundationwall, opt_heatingcooling, opt_windows):
    if n_clicks > 0:
        # Fetch the optimal combination
        opt_abovegradewall_opt, opt_ceilings_opt, opt_dhwsystem_opt, opt_foundationwall_opt, opt_heatingcooling_opt, opt_windows_opt, total_cost_opt, utility_bill_opt = get_optimal_combination()
        
        # Fetch the selected combination's costs
        total_cost_sel, utility_bill_sel = get_estimates(opt_abovegradewall, opt_ceilings, opt_dhwsystem, opt_foundationwall, opt_heatingcooling, opt_windows)

        optimal_combination_text = (
            f"Above Grade Wall: {opt_abovegradewall_opt}, "
            f"Ceilings: {opt_ceilings_opt}, "
            f"DHW System: {opt_dhwsystem_opt}, "
            f"Foundation Wall: {opt_foundationwall_opt}, "
            f"Heating/Cooling: {opt_heatingcooling_opt}, "
            f"Windows: {opt_windows_opt}, "
            f"Total Cost Estimates: {total_cost_opt}, "
            f"Gross Utility Bills (15 years): {utility_bill_opt * 15}"
        )

        # Create a comparison graph
        figure = {
            'data': [
                go.Bar(name='Selected Combination', x=['Total Cost', 'Utility Bill (15 years)'], 
                       y=[total_cost_sel, utility_bill_sel * 15]),
                go.Bar(name='Optimal Combination', x=['Total Cost', 'Utility Bill (15 years)'], 
                       y=[total_cost_opt, utility_bill_opt * 15])
            ],
            'layout': go.Layout(title='Comparison of Selected and Optimal Combinations',
                                barmode='group')
        }

        return optimal_combination_text, figure
    return 'Loading optimal combination...', {}


if __name__ == '__main__':
    app.run_server(debug=True)


