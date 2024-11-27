from dash import Dash, html, dcc, Input, Output, State
import pandas as pd

# Load the dataset
file_path = 'Final.csv'  # Update this with the actual path if needed
df = pd.read_csv(file_path)

# Select relevant columns and drop rows with missing values
columns = [
    'input|Opt-Windows', 'input|Opt-AboveGradeWall', 'input|Opt-Heating-Cooling', 'input|Opt-DHWSystem',
    'cost-estimates|byAttribute|Opt-Windows', 'cost-estimates|byAttribute|Opt-AboveGradeWall',
    'cost-estimates|byAttribute|Opt-Heating-Cooling', 'cost-estimates|byAttribute|Opt-DHWSystem',
    'output|Util-Bill-Net'
]
df = df[columns].dropna()

# Initialize Dash app
app = Dash(__name__)

# Define the layout of the app
app.layout = html.Div(children=[
    html.H1(children='Energy Cost Estimator'),
    
    html.Label('Windows:'),
    dcc.Dropdown(
        id='windows-dropdown',
        options=[{'label': val, 'value': val} for val in df['input|Opt-Windows'].unique()],
        value=df['input|Opt-Windows'].unique()[0]
    ),
    
    html.Label('Above Grade Wall:'),
    dcc.Dropdown(
        id='abovegradewall-dropdown',
        options=[{'label': val, 'value': val} for val in df['input|Opt-AboveGradeWall'].unique()],
        value=df['input|Opt-AboveGradeWall'].unique()[0]
    ),
    
    html.Label('Heating/Cooling:'),
    dcc.Dropdown(
        id='heatingcooling-dropdown',
        options=[{'label': val, 'value': val} for val in df['input|Opt-Heating-Cooling'].unique()],
        value=df['input|Opt-Heating-Cooling'].unique()[0]
    ),
    
    html.Label('DHW System:'),
    dcc.Dropdown(
        id='dhwsystem-dropdown',
        options=[{'label': val, 'value': val} for val in df['input|Opt-DHWSystem'].unique()],
        value=df['input|Opt-DHWSystem'].unique()[0]
    ),
    
    html.Button('Calculate', id='calculate-button', n_clicks=0),
    
    html.Div(id='output-container', children='Select options and click "Calculate" to see results'),
    
    html.H2(children='Recommended Upgrades'),
    html.Div(id='recommended-upgrades', children='Loading recommended upgrades...')
])

# Callback to compute results based on user selections
@app.callback(
    [Output('output-container', 'children'),
     Output('recommended-upgrades', 'children')],
    Input('calculate-button', 'n_clicks'),
    [State('windows-dropdown', 'value'),
     State('abovegradewall-dropdown', 'value'),
     State('heatingcooling-dropdown', 'value'),
     State('dhwsystem-dropdown', 'value')]
)
def calculate_recommendations(n_clicks, windows, above_grade_wall, heating_cooling, dhw_system):
    if n_clicks == 0:
        return "Please click the 'Calculate' button.", ""

    # Filter data based on current user selection
    current_row = df[
        (df['input|Opt-Windows'] == windows) &
        (df['input|Opt-AboveGradeWall'] == above_grade_wall) &
        (df['input|Opt-Heating-Cooling'] == heating_cooling) &
        (df['input|Opt-DHWSystem'] == dhw_system)
    ].iloc[0]

    # Retrieve the upgrade and current costs from the CSV for each component
    upgrade_costs = {
        'Windows': current_row['cost-estimates|byAttribute|Opt-Windows'],
        'AboveGradeWall': current_row['cost-estimates|byAttribute|Opt-AboveGradeWall'],
        'HeatingCooling': current_row['cost-estimates|byAttribute|Opt-Heating-Cooling'],
        'DHWSystem': current_row['cost-estimates|byAttribute|Opt-DHWSystem']
    }

    current_costs = {
        'Windows': df[df['input|Opt-Windows'] == windows]['cost-estimates|byAttribute|Opt-Windows'].values[0],
        'AboveGradeWall': df[df['input|Opt-AboveGradeWall'] == above_grade_wall]['cost-estimates|byAttribute|Opt-AboveGradeWall'].values[0],
        'HeatingCooling': df[df['input|Opt-Heating-Cooling'] == heating_cooling]['cost-estimates|byAttribute|Opt-Heating-Cooling'].values[0],
        'DHWSystem': df[df['input|Opt-DHWSystem'] == dhw_system]['cost-estimates|byAttribute|Opt-DHWSystem'].values[0]
    }

    util_bill_net_upgrade = current_row['output|Util-Bill-Net']
    util_bill_net_current = util_bill_net_upgrade  # This should be recalculated with current setup

    # Calculate savings and net savings
    recommendations = []
    for component, upgrade_cost in upgrade_costs.items():
        current_cost = current_costs[component]
        yearly_saving = util_bill_net_current - util_bill_net_upgrade
        fifteen_year_saving = yearly_saving * 15
        net_saving = fifteen_year_saving - (upgrade_cost - current_cost)

        recommendation = (f"For {component} we recommend switching, which costs {upgrade_cost:.2f} CAD to upgrade, "
                          f"while your current option costs {current_cost:.2f} CAD. "
                          f"This upgrade saves {yearly_saving:.2f} CAD annually and "
                          f"{fifteen_year_saving:.2f} CAD in 15 years. The net saving is {net_saving:.2f} CAD.")
        recommendations.append(recommendation)

    # Combine recommendations
    return f"Estimated Utility Bill: {util_bill_net_upgrade:.2f} CAD", "\n".join(recommendations)

if __name__ == '__main__':
    app.run_server(debug=True)
