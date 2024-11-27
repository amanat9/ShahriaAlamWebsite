from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import plotly.graph_objs as go


app = Dash(__name__)
server= app.server

# Load the dataset
file_path = './Final.csv'
df = pd.read_csv(file_path)

# Select relevant columns and drop rows with missing values
columns = [
    'input|Opt-Windows', 'input|Opt-AboveGradeWall', 'input|Opt-Heating-Cooling', 'input|Opt-DHWSystem',
    'cost-estimates|byAttribute|Opt-Windows', 'cost-estimates|byAttribute|Opt-AboveGradeWall', 
    'cost-estimates|byAttribute|Opt-Heating-Cooling', 'cost-estimates|byAttribute|Opt-DHWSystem',
    'output|Util-Bill-Net'
]
df = df[columns].dropna()

# Declare the app
app = Dash(__name__, title="Energy Cost Estimator")

# Define layout
app.layout = html.Div(children=[
    html.H1(children='Energy Cost Estimator'),
    html.Div(children='Select your options:'),
    
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
    html.Div(id='recommended-upgrades', children='Loading recommended upgrades...'),
    
    dcc.Graph(id='savings-graph')
])

# Function to calculate the best upgrades
def find_best_upgrades(df, user_selections):
    results = []
    components = ['input|Opt-Windows', 'input|Opt-AboveGradeWall', 'input|Opt-Heating-Cooling', 'input|Opt-DHWSystem']
    
    for component in components:
        cost_col = f'cost-estimates|byAttribute|{component.split("|")[1]}'
        
        current_value = user_selections[component]
        current_cost = df[df[component] == current_value][cost_col].mode()[0]
        current_util_bill_net = df[(df['input|Opt-Windows'] == user_selections['input|Opt-Windows']) & 
                                   (df['input|Opt-AboveGradeWall'] == user_selections['input|Opt-AboveGradeWall']) & 
                                   (df['input|Opt-Heating-Cooling'] == user_selections['input|Opt-Heating-Cooling']) & 
                                   (df['input|Opt-DHWSystem'] == user_selections['input|Opt-DHWSystem'])]['output|Util-Bill-Net'].mode()[0]
        
        best_saving = float('-inf')
        best_upgrade = None
        
        for upgrade in df[component].unique():
            if upgrade != current_value:
                upgrade_cost = df[df[component] == upgrade][cost_col].mode()[0]
                util_bill_net = df[(df['input|Opt-Windows'] == (user_selections['input|Opt-Windows'] if component != 'input|Opt-Windows' else upgrade)) & 
                                   (df['input|Opt-AboveGradeWall'] == (user_selections['input|Opt-AboveGradeWall'] if component != 'input|Opt-AboveGradeWall' else upgrade)) & 
                                   (df['input|Opt-Heating-Cooling'] == (user_selections['input|Opt-Heating-Cooling'] if component != 'input|Opt-Heating-Cooling' else upgrade)) & 
                                   (df['input|Opt-DHWSystem'] == (user_selections['input|Opt-DHWSystem'] if component != 'input|Opt-DHWSystem' else upgrade))]['output|Util-Bill-Net'].mode()
                if len(util_bill_net) == 0:
                    continue
                util_bill_net = util_bill_net[0]

                yearly_saving = current_util_bill_net - util_bill_net
                saving_15_years = yearly_saving * 15
                net_saving = saving_15_years - upgrade_cost

                if yearly_saving > 0 and saving_15_years > 0 and net_saving > best_saving:
                    best_saving = net_saving
                    best_upgrade = upgrade

        if best_upgrade is not None and yearly_saving > 0 and saving_15_years > 0:
            results.append({
                'Component': component,
                'Current Value': current_value,
                'Best Upgrade': best_upgrade,
                'Current Cost': current_cost,
                'Upgrade Cost': upgrade_cost,
                'Yearly Saving': yearly_saving,
                'Saving 15 Years': saving_15_years,
                'Net Saving': best_saving
            })
        else:
            results.append({
                'Component': component,
                'Current Value': current_value,
                'Best Upgrade': 'No Upgrade Found',
                'Current Cost': current_cost,
                'Upgrade Cost': 0,
                'Yearly Saving': 0,
                'Saving 15 Years': 0,
                'Net Saving': 0
            })
    
    return results

@app.callback(
    Output('output-container', 'children'),
    Output('recommended-upgrades', 'children'),
    Output('savings-graph', 'figure'),
    Input('calculate-button', 'n_clicks'),
    State('windows-dropdown', 'value'),
    State('abovegradewall-dropdown', 'value'),
    State('heatingcooling-dropdown', 'value'),
    State('dhwsystem-dropdown', 'value')
)
def update_output(n_clicks, windows, abovegradewall, heatingcooling, dhwsystem):
    if n_clicks > 0:
        user_selections = {
            'input|Opt-Windows': windows,
            'input|Opt-AboveGradeWall': abovegradewall,
            'input|Opt-Heating-Cooling': heatingcooling,
            'input|Opt-DHWSystem': dhwsystem
        }
        
        best_upgrades = find_best_upgrades(df, user_selections)
        best_upgrades_df = pd.DataFrame(best_upgrades)
        
        current_cost = df[(df['input|Opt-Windows'] == windows) & 
                          (df['input|Opt-AboveGradeWall'] == abovegradewall) & 
                          (df['input|Opt-Heating-Cooling'] == heatingcooling) & 
                          (df['input|Opt-DHWSystem'] == dhwsystem)][['cost-estimates|byAttribute|Opt-Windows', 
                                                                      'cost-estimates|byAttribute|Opt-AboveGradeWall', 
                                                                      'cost-estimates|byAttribute|Opt-Heating-Cooling', 
                                                                      'cost-estimates|byAttribute|Opt-DHWSystem']].sum().sum()
        current_util_bill_net = df[(df['input|Opt-Windows'] == windows) & 
                                   (df['input|Opt-AboveGradeWall'] == abovegradewall) & 
                                   (df['input|Opt-Heating-Cooling'] == heatingcooling) & 
                                   (df['input|Opt-DHWSystem'] == dhwsystem)]['output|Util-Bill-Net'].mode()[0]
        
        recommended_text_elements = []
        for _, row in best_upgrades_df.iterrows():
            if row['Best Upgrade'] == 'No Upgrade Found':
                recommended_text_elements.append(html.P(f"For {row['Component']} No Upgrade Found"))
            else:
                recommended_text_elements.append(
                    html.P(
                        f"For {row['Component']} we recommend switching from {row['Current Value']} to {row['Best Upgrade']} which saves {row['Yearly Saving']:.2f} CAD annually and {row['Saving 15 Years']:.2f} CAD in 15 years."
                    )
                )
        
        savings_data = [
            go.Bar(
                x=best_upgrades_df['Component'],
                y=best_upgrades_df['Yearly Saving'],
                name='Yearly Saving'
            ),
            go.Bar(
                x=best_upgrades_df['Component'],
                y=best_upgrades_df['Saving 15 Years'],
                name='Saving 15 Years'
            ),
            go.Bar(
                x=best_upgrades_df['Component'],
                y=best_upgrades_df['Net Saving'],
                name='Net Saving'
            )
        ]
        
        savings_layout = go.Layout(
            title='Savings by Component Upgrade',
            barmode='group'
        )
        
        figure = go.Figure(data=savings_data, layout=savings_layout)
        
        return (
            f"Current Cost: {current_cost}, Current Util Bill (15 years): {current_util_bill_net * 15}",
            html.Div(recommended_text_elements),
            figure
        )
    
    return 'Select options and click "Calculate" to see results', 'Loading recommended upgrades...', {}















if __name__ == '__main__':
	app.run_server()
