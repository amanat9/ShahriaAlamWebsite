# @ Create Time: 2024-02-28 12:14:28.059962

from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objs as go

app = Dash(__name__, title="MyDashApp")

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

# Load the dataset
file_path = './relevant.csv'
data = pd.read_csv(file_path)

# Handle missing or 'NA' values in 'input|Opt-Windows'
data['input|Opt-Windows'] = data['input|Opt-Windows'].fillna('NA')

# Encode categorical variables
label_encoders = {}
for column in ['input|Run-Locale', 'input|Opt-Windows', 'input|Opt-ACH']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

# Function to get estimates
def get_estimates(location, opt_ach, opt_windows):
    # Encode the user's choices
    location_encoded = label_encoders['input|Run-Locale'].transform([location])[0]
    opt_ach_encoded = label_encoders['input|Opt-ACH'].transform([opt_ach])[0]
    opt_windows_encoded = label_encoders['input|Opt-Windows'].transform([opt_windows])[0]

    # Find the corresponding row in the dataset
    row = data[(data['input|Run-Locale'] == location_encoded) &
               (data['input|Opt-ACH'] == opt_ach_encoded) &
               (data['input|Opt-Windows'] == opt_windows_encoded)]

    if not row.empty:
        total_cost = row['cost-estimates|total'].values[0]
        utility_bill = row['output|Util-Bill-gross'].values[0]
        return total_cost, utility_bill
    else:
        return None, None

# Function to find the optimal combination
def get_optimal_combination():
    data['total_cost_with_multiplier'] = data['cost-estimates|total'] + (data['output|Util-Bill-gross'] * 15)
    optimal_row = data.loc[data['total_cost_with_multiplier'].idxmin()]

    location = label_encoders['input|Run-Locale'].inverse_transform([optimal_row['input|Run-Locale']])[0]
    opt_ach = label_encoders['input|Opt-ACH'].inverse_transform([optimal_row['input|Opt-ACH']])[0]
    opt_windows = label_encoders['input|Opt-Windows'].inverse_transform([optimal_row['input|Opt-Windows']])[0]

    total_cost = optimal_row['cost-estimates|total']
    utility_bill = optimal_row['output|Util-Bill-gross']

    return location, opt_ach, opt_windows, total_cost, utility_bill

# Get initial default values
default_location = 'HALIFAX'
default_opt_ach = 'New-Const-air_seal_to_1.50_ach'
default_opt_windows = 'NA'

# Define layout
app.layout = html.Div(children=[
    html.H1(children='Energy Cost Estimator'),

    html.Div(children='''Select your options:'''),

    dcc.Dropdown(
        id='location-dropdown',
        options=[
            {'label': 'HALIFAX', 'value': 'HALIFAX'},
            {'label': 'GREENWOOD', 'value': 'GREENWOOD'},
            {'label': 'YARMOUTH', 'value': 'YARMOUTH'}
        ],
        value=default_location
    ),

    dcc.Dropdown(
        id='ach-dropdown',
        options=[
            {'label': 'New-Const-air_seal_to_1.50_ach', 'value': 'New-Const-air_seal_to_1.50_ach'},
            {'label': 'New-Const-air_seal_to_1.00_ach', 'value': 'New-Const-air_seal_to_1.00_ach'},
            {'label': 'New-Const-air_seal_to_0.60_ach', 'value': 'New-Const-air_seal_to_0.60_ach'}
        ],
        value=default_opt_ach
    ),

    dcc.Dropdown(
        id='windows-dropdown',
        options=[
            {'label': 'NA', 'value': 'NA'},
            {'label': 'NC-2g-LG-u1.65', 'value': 'NC-2g-LG-u1.65'},
            {'label': 'NC-2g-HG-u1.65', 'value': 'NC-2g-HG-u1.65'}
        ],
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
    [State('location-dropdown', 'value'),
     State('ach-dropdown', 'value'),
     State('windows-dropdown', 'value')]
)
def update_output(n_clicks, location, opt_ach, opt_windows):
    if n_clicks > 0:
        total_cost, utility_bill = get_estimates(location, opt_ach, opt_windows)
        if total_cost is not None and utility_bill is not None:
            return f"Total Cost Estimates: {total_cost}, Gross Utility Bills: {utility_bill}"
        else:
            return "No data available for the selected combination"
    return 'Select options and click "Calculate" to see results'

@app.callback(
    [Output('optimal-combination', 'children'),
     Output('comparison-graph', 'figure')],
    [Input('calculate-button', 'n_clicks')],
    [State('location-dropdown', 'value'),
     State('ach-dropdown', 'value'),
     State('windows-dropdown', 'value')]
)
def display_optimal_combination(n_clicks, location, opt_ach, opt_windows):
    if n_clicks > 0:
        location_opt, opt_ach_opt, opt_windows_opt, total_cost_opt, utility_bill_opt = get_optimal_combination()
        
        total_cost_sel, utility_bill_sel = get_estimates(location, opt_ach, opt_windows)

        optimal_combination_text = (f"Location: {location_opt}, Opt-ACH: {opt_ach_opt}, Opt-Windows: {opt_windows_opt}, "
                                    f"Total Cost Estimates: {total_cost_opt}, Gross Utility Bills: {utility_bill_opt}")

        # Create a comparison graph
        figure = {
            'data': [
                go.Bar(name='Selected Combination', x=['Total Cost', 'Utility Bill'], 
                       y=[total_cost_sel, utility_bill_sel]),
                go.Bar(name='Optimal Combination', x=['Total Cost', 'Utility Bill'], 
                       y=[total_cost_opt, utility_bill_opt])
            ],
            'layout': go.Layout(title='Comparison of Selected and Optimal Combinations',
                                barmode='group')
        }

        return optimal_combination_text, figure
    return 'Loading optimal combination...', {}

if __name__ == '__main__':
    app.run_server(debug=True)
