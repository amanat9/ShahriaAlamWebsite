# combined_page.py
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import plotly.graph_objs as go
import dash
import dash_bootstrap_components as dbc

# Register the page with Dash
dash.register_page(__name__, path="/combined")

# Define the layout
layout = dbc.Container([
    html.H1("Energy Cost Estimator"),

    # House Type Dropdown
    html.Div([
        html.Label('Select House Type:'),
        dcc.Dropdown(
            id='house-type-dropdown',
            options=[
                {'label': 'Rancher', 'value': 'Rancher'},
                {'label': 'Two Storey', 'value': '2storey'},
                {'label': 'Upslopes', 'value': 'upslope'}
            ],
            value='Rancher',
            style={'color': 'black'}  # Set the text color to black
        )
    ], style={'margin-bottom': '3cm'}),  # Add space after the dropdown

    dbc.Row([
        dbc.Col([
            html.Img(
                src='/assets/image1.png',
                id='image1',
                n_clicks=0,
                style={'width': '100%', 'cursor': 'pointer'}
            ),
            html.Br(),
            dbc.Button('See Plan', id='button-image1', n_clicks=0)
        ], width=4),
        dbc.Col([
            html.Img(
                src='/assets/image2.png',
                id='image2',
                n_clicks=0,
                style={'width': '100%', 'cursor': 'pointer'}
            ),
            html.Br(),
            dbc.Button('See Plan', id='button-image2', n_clicks=0)
        ], width=4),
        dbc.Col([
            html.Img(
                src='/assets/image3.png',
                id='image3',
                n_clicks=0,
                style={'width': '100%', 'cursor': 'pointer'}
            ),
            html.Br(),
            dbc.Button('See Plan', id='button-image3', n_clicks=0)
        ], width=4),
    ]),

    # Modal for PDF viewer
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("PDF Viewer")),
            dbc.ModalBody(
                html.Iframe(id="pdf-iframe", src="", style={"width": "100%", "height": "500px"})
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-pdf-modal", className="ml-auto")
            ),
        ],
        id="pdf-modal",
        size="xl",
        is_open=False,
    ),

    # Hidden stores to keep track of selected house and area
    dcc.Store(id='selected-house', storage_type='session'),
    dcc.Store(id='selected-area', storage_type='session'),
    dcc.Store(id='data-store', storage_type='memory'),

    # Area dropdown, initially hidden
    html.Div(id='area-selection', children=[
        html.Label('Select House Area:'),
        dcc.Dropdown(
            id='area-dropdown',
            options=[
                {'label': '1000', 'value': '1000'},
                {'label': '2000', 'value': '2000'}
            ],
            value='1000',
            style={'color': 'black'}  # Set the text color to black
        )
    ], style={'display': 'none'}),  # Initially hidden

    # Options container, initially hidden
    html.Div(id='options-container', children=[
        html.Div(children='Select your options:'),

        # Placeholder Dropdowns
        html.Label('Number of Windows:'),
        dcc.Dropdown(
            id='number-of-windows-dropdown',
            options=[
                {'label': '3', 'value': '3'},
                {'label': '4', 'value': '4'},
                {'label': '5', 'value': '5'}
            ],
            value='3',
            style={'color': 'black'}  # Set the text color to black
        ),

        html.Label('Optimization Criteria:'),
        dcc.Dropdown(
            id='optimization-criteria-dropdown',
            options=[
                {'label': 'Operation Cost', 'value': 'operation cost'},
                {'label': 'Construction Cost', 'value': 'construction cost'},
                {'label': 'Emissions', 'value': 'emissions'},
                {'label': 'Overall', 'value': 'overall'}
            ],
            value='operation cost',
            style={'color': 'black'}  # Set the text color to black
        ),

        # Single RadioItems for Component Selection
        html.Label('Select Component to Optimize:'),
        dcc.RadioItems(
            id='component-radio',
            options=[
                {'label': 'Windows', 'value': 'Windows'},
                {'label': 'Foundation Wall Ext Ins', 'value': 'Foundation Wall Ext Ins'},
                {'label': 'Heating/Cooling', 'value': 'Heating/Cooling'}
            ],
            value='Windows',  # Default selection
            labelStyle={'display': 'block'},
            inputStyle={"margin-right": "10px"}
        ),

        # Options Dropdowns
        html.Label('Windows:'),
        dcc.Dropdown(
            id='windows-dropdown',
            options=[],  # Options will be populated after data is loaded
            value=None,
            style={'color': 'black'}  # Set the text color to black
        ),

        html.Label('Foundation Wall Ext Ins:'),
        dcc.Dropdown(
            id='foundationwall-dropdown',
            options=[],
            value=None,
            style={'color': 'black'}  # Set the text color to black
        ),

        html.Label('Heating/Cooling:'),
        dcc.Dropdown(
            id='heatingcooling-dropdown',
            options=[],
            value=None,
            style={'color': 'black'}  # Set the text color to black
        ),

        # Buttons
        html.Button('Optimize Selected', id='optimize-selected-button', n_clicks=0),
        html.Button('Optimize All', id='optimize-all-button', n_clicks=0),

        html.Div(id='output-container', children='Select options and click "Optimize All" to see results'),

        html.H2(children='Recommended Upgrades'),
        html.Div(id='recommended-upgrades', children='Loading recommended upgrades...'),

        dcc.Graph(id='savings-graph')
    ], style={'display': 'none'})  # Initially hidden

], fluid=True)

# Callback to update images based on house type
@callback(
    Output('image1', 'src'),
    Output('image2', 'src'),
    Output('image3', 'src'),
    Input('house-type-dropdown', 'value')
)
def update_images(house_type):
    if house_type == 'Rancher':
        prefix = ''
    else:
        prefix = f'{house_type}-'
    src1 = f'/assets/{prefix}image1.png'
    src2 = f'/assets/{prefix}image2.png'
    src3 = f'/assets/{prefix}image3.png'
    return src1, src2, src3

# Callback to handle PDF modal
@callback(
    Output("pdf-modal", "is_open"),
    Output("pdf-iframe", "src"),
    [Input("button-image1", "n_clicks"),
     Input("button-image2", "n_clicks"),
     Input("button-image3", "n_clicks"),
     Input("close-pdf-modal", "n_clicks")],
    [State("pdf-modal", "is_open"),
     State('house-type-dropdown', 'value')],
)
def toggle_pdf_modal(n1, n2, n3, close_clicks, is_open, house_type):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open, ""
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == "close-pdf-modal":
            return False, ""
        elif button_id in ["button-image1", "button-image2", "button-image3"]:
            prefix = ''
            if house_type != 'Rancher':
                prefix = f'{house_type}-'
            pdf_src = ""
            if button_id == "button-image1":
                pdf_src = f"/assets/{prefix}plan1.pdf"
            elif button_id == "button-image2":
                pdf_src = f"/assets/{prefix}plan2.pdf"
            elif button_id == "button-image3":
                pdf_src = f"/assets/{prefix}plan3.pdf"
            return True, pdf_src
        else:
            # Do not open the modal for any other triggers
            return is_open, ""

# Callback to handle house selection and display the area dropdown
@callback(
    Output('selected-house', 'data'),
    Output('area-selection', 'style'),
    Input('image1', 'n_clicks'),
    Input('image2', 'n_clicks'),
    Input('image3', 'n_clicks')
)
def select_house(n_clicks_img1, n_clicks_img2, n_clicks_img3):
    ctx = dash.callback_context

    if not ctx.triggered:
        return dash.no_update, {'display': 'none'}
    else:
        # Get the ID of the element that triggered the callback
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'image1' and n_clicks_img1:
            selected_house = 'house1'
        elif button_id == 'image2' and n_clicks_img2:
            selected_house = 'house2'
        elif button_id == 'image3' and n_clicks_img3:
            selected_house = 'house3'
        else:
            return dash.no_update, {'display': 'none'}

        # Display the area selection
        return selected_house, {'display': 'block'}

# Callback to load data and populate options after area is selected
@callback(
    Output('options-container', 'style'),
    Output('windows-dropdown', 'options'),
    Output('foundationwall-dropdown', 'options'),
    Output('heatingcooling-dropdown', 'options'),
    Output('windows-dropdown', 'value'),
    Output('foundationwall-dropdown', 'value'),
    Output('heatingcooling-dropdown', 'value'),
    Output('data-store', 'data'),
    Input('selected-house', 'data'),
    Input('area-dropdown', 'value'),
    Input('house-type-dropdown', 'value')
)
def load_data(selected_house, area, house_type):
    if not selected_house or not area or not house_type:
        return {'display': 'none'}, [], [], [], None, None, None, None

    # Determine the prefix based on the house type
    if house_type == 'Rancher':
        prefix = ''
    else:
        prefix = f'{house_type}-'

    # Determine the base_name based on selected_house
    if selected_house == 'house1':
        base_name = 'One-story_Full_Basement'
    elif selected_house == 'house2':
        base_name = 'Two-story_Full_Basement'
    elif selected_house == 'house3':
        base_name = 'Three-story_Full_Basement'
    else:
        return {'display': 'none'}, [], [], [], None, None, None, None

    file_path = f'assets/{prefix}{base_name}_{area}SF.csv'

    # Load the data with error handling
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return {'display': 'none'}, [], [], [], None, None, None, None

    # Now, get the unique values for each option
    windows_options = [{'label': val, 'value': val} for val in df['input|Opt-Windows'].unique()]
    foundationwall_options = [{'label': val, 'value': val} for val in df['input|Opt-FoundationWallExtIns'].unique()]
    heatingcooling_options = [{'label': val, 'value': val} for val in df['input|Opt-Heating-Cooling'].unique()]

    # Set default values
    windows_value = df['input|Opt-Windows'].unique()[0]
    foundationwall_value = df['input|Opt-FoundationWallExtIns'].unique()[0]
    heatingcooling_value = df['input|Opt-Heating-Cooling'].unique()[0]

    # Serialize the data
    data_json = df.to_json(date_format='iso', orient='split')

    # Display the options container
    return {'display': 'block'}, windows_options, foundationwall_options, heatingcooling_options, windows_value, foundationwall_value, heatingcooling_value, data_json

# Function to find the best upgrades
def find_best_upgrades(df, user_selections, selected_component=None):
    results = []
    components = ['input|Opt-Windows', 'input|Opt-FoundationWallExtIns', 'input|Opt-Heating-Cooling']

    # If a specific component is selected, optimize only that component
    if selected_component:
        components = [selected_component]

    for component in components:
        cost_col = f'cost-estimates|byAttribute|{component.split("|")[1]}'

        current_value = user_selections[component]
        current_df = df[
            (df['input|Opt-Windows'] == user_selections['input|Opt-Windows']) &
            (df['input|Opt-FoundationWallExtIns'] == user_selections['input|Opt-FoundationWallExtIns']) &
            (df['input|Opt-Heating-Cooling'] == user_selections['input|Opt-Heating-Cooling'])
        ]
        current_cost = current_df[cost_col].mode()[0]
        current_util_bill_net = current_df['output|Util-Bill-Net'].mode()[0]

        best_saving = float('-inf')
        best_upgrade = None
        best_yearly_saving = 0
        best_saving_15_years = 0
        best_net_saving = 0

        for upgrade in df[component].unique():
            if upgrade != current_value:
                upgrade_df = df.copy()
                upgrade_selections = user_selections.copy()
                upgrade_selections[component] = upgrade
                upgrade_df = upgrade_df[
                    (upgrade_df['input|Opt-Windows'] == upgrade_selections['input|Opt-Windows']) &
                    (upgrade_df['input|Opt-FoundationWallExtIns'] == upgrade_selections['input|Opt-FoundationWallExtIns']) &
                    (upgrade_df['input|Opt-Heating-Cooling'] == upgrade_selections['input|Opt-Heating-Cooling'])
                ]
                if upgrade_df.empty:
                    continue
                upgrade_cost = upgrade_df[cost_col].mode()[0]
                upgrade_util_bill_net = upgrade_df['output|Util-Bill-Net'].mode()[0]

                yearly_saving = current_util_bill_net - upgrade_util_bill_net
                saving_15_years = yearly_saving * 15
                upgrade_diff = upgrade_cost - current_cost
                net_saving = saving_15_years - upgrade_diff

                if yearly_saving > 0 and net_saving > best_saving:
                    best_saving = net_saving
                    best_upgrade = upgrade
                    best_yearly_saving = yearly_saving
                    best_saving_15_years = saving_15_years
                    best_net_saving = net_saving

        if best_upgrade is not None:
            results.append({
                'Component': component.split('|')[1],
                'Current Value': current_value,
                'Best Upgrade': best_upgrade,
                'Yearly Saving': best_yearly_saving,
                'Saving 15 Years': best_saving_15_years,
                'Net Saving': best_net_saving
            })
        else:
            results.append({
                'Component': component.split('|')[1],
                'Current Value': current_value,
                'Best Upgrade': 'No Upgrade Found',
                'Yearly Saving': 0,
                'Saving 15 Years': 0,
                'Net Saving': 0
            })

    return results

# Callback for the 'Optimize All' button
@callback(
    Output('output-container', 'children'),
    Output('recommended-upgrades', 'children'),
    Output('savings-graph', 'figure'),
    Input('optimize-all-button', 'n_clicks'),
    State('windows-dropdown', 'value'),
    State('foundationwall-dropdown', 'value'),
    State('heatingcooling-dropdown', 'value'),
    State('data-store', 'data')
)
def optimize_all(n_clicks, windows, foundationwall, heatingcooling, data_json):
    if n_clicks > 0 and data_json:
        # Load the data from JSON
        df = pd.read_json(data_json, orient='split')

        user_selections = {
            'input|Opt-Windows': windows,
            'input|Opt-FoundationWallExtIns': foundationwall,
            'input|Opt-Heating-Cooling': heatingcooling
        }

        best_upgrades = find_best_upgrades(df, user_selections)
        best_upgrades_df = pd.DataFrame(best_upgrades)

        recommended_text_elements = []
        for _, row in best_upgrades_df.iterrows():
            recommended_text_elements.append(
                html.P(
                    f"For {row['Component']} we recommend switching from {row['Current Value']} to {row['Best Upgrade']} "
                    f"which saves {row['Yearly Saving']:.2f} CAD annually and {row['Saving 15 Years']:.2f} CAD in 15 years."
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

        current_df = df[
            (df['input|Opt-Windows'] == user_selections['input|Opt-Windows']) &
            (df['input|Opt-FoundationWallExtIns'] == user_selections['input|Opt-FoundationWallExtIns']) &
            (df['input|Opt-Heating-Cooling'] == user_selections['input|Opt-Heating-Cooling'])
        ]

        total_cost = current_df['output|Util-Bill-Net'].mode()[0]

        return (
            f"Total Cost: {total_cost:.2f} CAD",
            html.Div(recommended_text_elements),
            figure
        )

    return 'Select options and click "Optimize All" to see results', 'Loading recommended upgrades...', {}

# Callback for the 'Optimize Selected' button (Placeholder functionality)
@callback(
    Output('output-container', 'children', allow_duplicate=True),
    Output('recommended-upgrades', 'children', allow_duplicate=True),
    Output('savings-graph', 'figure', allow_duplicate=True),
    Input('optimize-selected-button', 'n_clicks'),
    State('component-radio', 'value'),
    State('windows-dropdown', 'value'),
    State('foundationwall-dropdown', 'value'),
    State('heatingcooling-dropdown', 'value'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def optimize_selected(n_clicks, selected_component, windows, foundationwall, heatingcooling, data_json):
    if n_clicks > 0 and data_json:
        # Map the selected component to the appropriate key
        component_key_map = {
            'Windows': 'input|Opt-Windows',
            'Foundation Wall Ext Ins': 'input|Opt-FoundationWallExtIns',
            'Heating/Cooling': 'input|Opt-Heating-Cooling'
        }
        selected_key = component_key_map.get(selected_component)

        # Load the data from JSON
        df = pd.read_json(data_json, orient='split')

        user_selections = {
            'input|Opt-Windows': windows,
            'input|Opt-FoundationWallExtIns': foundationwall,
            'input|Opt-Heating-Cooling': heatingcooling
        }

        best_upgrades = find_best_upgrades(df, user_selections, selected_component=selected_key)
        best_upgrades_df = pd.DataFrame(best_upgrades)

        recommended_text_elements = []
        for _, row in best_upgrades_df.iterrows():
            recommended_text_elements.append(
                html.P(
                    f"For {row['Component']} we recommend switching from {row['Current Value']} to {row['Best Upgrade']} "
                    f"which saves {row['Yearly Saving']:.2f} CAD annually and {row['Saving 15 Years']:.2f} CAD in 15 years."
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

        current_df = df[
            (df['input|Opt-Windows'] == user_selections['input|Opt-Windows']) &
            (df['input|Opt-FoundationWallExtIns'] == user_selections['input|Opt-FoundationWallExtIns']) &
            (df['input|Opt-Heating-Cooling'] == user_selections['input|Opt-Heating-Cooling'])
        ]

        total_cost = current_df['output|Util-Bill-Net'].mode()[0]

        return (
            f"Total Cost: {total_cost:.2f} CAD",
            html.Div(recommended_text_elements),
            figure
        )

    return dash.no_update, dash.no_update, dash.no_update
