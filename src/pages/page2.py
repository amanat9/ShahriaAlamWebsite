# combined_page.py

from dash import html, dcc, Input, Output, State, callback
import dash
import dash_bootstrap_components as dbc

import pandas as pd
import plotly.graph_objs as go

# For the SVM model
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Register the page with Dash
dash.register_page(__name__, path="/combined")

###############################################################################
# 1) Pre-load and combine data, then train SVM models for each house type
###############################################################################
# We'll assume you have 2 CSVs per house type for demonstration:
#   assets/{house_type_prefix}One-story_Full_Basement_1000SF.csv
#   assets/{house_type_prefix}One-story_Full_Basement_2000SF.csv
#   assets/{house_type_prefix}Two-story_Full_Basement_1000SF.csv
#   assets/{house_type_prefix}Two-story_Full_Basement_2000SF.csv
#   assets/{house_type_prefix}Three-story_Full_Basement_1000SF.csv
#   assets/{house_type_prefix}Three-story_Full_Basement_2000SF.csv
# etc., based on your naming.
#
# We'll build a dict of SVM models keyed by (house_type, base_name).
# Then whenever the user picks "Machine Learning", we can use the correct SVM
# for that combination.

def load_and_combine_csvs(house_type, base_name):
    """
    Loads two CSVs: one for 1000 SF, one for 2000 SF,
    combines them into a single DataFrame with 'Area' column indicating 1000 or 2000.
    """
    # The house_type might be '', '2storey-', 'upslope-', etc.
    prefix = '' if house_type == 'Rancher' else f'{house_type}-'
    
    # For example, base_name might be 'One-story_Full_Basement'
    df_combined = pd.DataFrame()
    for area_val in [1000, 2000]:
        path = f'assets/{prefix}{base_name}_{area_val}SF.csv'
        try:
            df_temp = pd.read_csv(path)
            # Add an "Area" column to each row
            df_temp['Area'] = area_val
            df_combined = pd.concat([df_combined, df_temp], ignore_index=True)
        except FileNotFoundError:
            pass
    return df_combined

def train_svms_for_house_type(house_type, base_name):
    """
    Train three separate SVM classifiers to predict:
    - Windows
    - FoundationWallExtIns
    - Heating/Cooling
    from the 'Area' feature.
    Returns a dict with the three trained models plus label encoders.
    """
    df = load_and_combine_csvs(house_type, base_name)
    if df.empty:
        return None  # No data for this combination

    # Features: just the 'Area' as numeric
    # Target columns:
    windows_col = 'input|Opt-Windows'
    foundation_col = 'input|Opt-FoundationWallExtIns'
    heating_col = 'input|Opt-Heating-Cooling'

    # Prepare X, y
    X = df[['Area']].values

    # Prepare label encoders for each target (to convert strings to numeric classes)
    le_windows = LabelEncoder()
    le_foundation = LabelEncoder()
    le_heating = LabelEncoder()

    y_windows = le_windows.fit_transform(df[windows_col])
    y_foundation = le_foundation.fit_transform(df[foundation_col])
    y_heating = le_heating.fit_transform(df[heating_col])

    # Train an SVM for each target
    svm_windows = SVC()
    svm_foundation = SVC()
    svm_heating = SVC()

    svm_windows.fit(X, y_windows)
    svm_foundation.fit(X, y_foundation)
    svm_heating.fit(X, y_heating)

    return {
        'svm_windows': svm_windows,
        'svm_foundation': svm_foundation,
        'svm_heating': svm_heating,
        'le_windows': le_windows,
        'le_foundation': le_foundation,
        'le_heating': le_heating
    }

# Precompute possible combinations of house_image -> base_name
house_to_basename_map = {
    'house1': 'One-story_Full_Basement',
    'house2': 'Two-story_Full_Basement',
    'house3': 'Three-story_Full_Basement'
}

# Similarly map the "house-type-dropdown" to prefix strings used in the filenames
house_type_prefix = {
    'Rancher': '',
    '2storey': '2storey-',
    'upslope': 'upslope-'
}

# We'll keep a big dictionary of models, keyed by (houseTypeDropdownValue, selectedHouseImage)
# e.g. models_cache[('Rancher','house1')] = { 'svm_windows':..., 'le_windows':... }
models_cache = {}

def get_svm_models_for(house_type, selected_house):
    """
    Load or return from cache the SVM models for the given combination
    (house_type, selected_house).
    """
    prefix_key = house_type_prefix.get(house_type, '')  # e.g. '2storey-' or ''
    base_name = house_to_basename_map.get(selected_house, '')
    cache_key = (house_type, selected_house)
    if cache_key not in models_cache:
        # Train the SVM models
        models = train_svms_for_house_type(house_type, base_name)
        models_cache[cache_key] = models
    return models_cache[cache_key]


###############################################################################
# 2) Layout
###############################################################################
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
            style={'color': 'black'}
        )
    ], style={'margin-bottom': '3cm'}),

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
    dcc.Store(id='data-store', storage_type='memory'),  # For the original CSV data

    # ============== NEW: Radio to choose "Predefined" or "Machine Learning" =============
    html.Div([
        html.Label("Select how to choose House Area:"),
        dcc.RadioItems(
            id='area-input-mode',
            options=[
                {'label': 'Use Predefined Values (1000 / 2000)', 'value': 'predefined'},
                {'label': 'Use Machine Learning (SVM Prediction)', 'value': 'ml'}
            ],
            value='predefined',
            style={'margin-bottom': '20px'}
        ),
    ], style={'margin-top': '20px'}),

    # Area selection (Predefined) dropdown
    html.Div(id='area-selection', children=[
        html.Label('Select House Area:'),
        dcc.Dropdown(
            id='area-dropdown',
            options=[
                {'label': '1000', 'value': '1000'},
                {'label': '2000', 'value': '2000'}
            ],
            value='1000',
            style={'color': 'black'}
        )
    ], style={'display': 'none'}),  # We'll show or hide this dynamically

    # Slider for ML-based area
    html.Div(id='ml-area-slider-container', children=[
        html.Label("Select House Area (SVM will predict Windows/Foundation/Heating):"),
        dcc.Slider(
            id='ml-area-slider',
            min=800,
            max=5000,
            step=50,
            value=1000,
            marks={n: str(n) for n in range(800, 5001, 400)},
            tooltip={"placement": "bottom"}
        ),
        html.Div(id='ml-slider-value', style={'margin-top': '10px'}),
    ], style={'display': 'none'}),

    # Options container
    html.Div(id='options-container', children=[
        html.Div(children='Select your options:'),

        # (Placeholder) dropdown for number of windows
        html.Label('Number of Windows:'),
        dcc.Dropdown(
            id='number-of-windows-dropdown',
            options=[
                {'label': '3', 'value': '3'},
                {'label': '4', 'value': '4'},
                {'label': '5', 'value': '5'}
            ],
            value='3',
            style={'color': 'black'}
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
            style={'color': 'black'}
        ),

        # Single RadioItems for which component to optimize
        html.Label('Select Component to Optimize:'),
        dcc.RadioItems(
            id='component-radio',
            options=[
                {'label': 'Windows', 'value': 'Windows'},
                {'label': 'Foundation Wall Ext Ins', 'value': 'Foundation Wall Ext Ins'},
                {'label': 'Heating/Cooling', 'value': 'Heating/Cooling'}
            ],
            value='Windows',
            labelStyle={'display': 'block'},
            inputStyle={"margin-right": "10px"}
        ),

        html.Label('Windows:'),
        dcc.Dropdown(
            id='windows-dropdown',
            options=[],
            value=None,
            style={'color': 'black'}
        ),

        html.Label('Foundation Wall Ext Ins:'),
        dcc.Dropdown(
            id='foundationwall-dropdown',
            options=[],
            value=None,
            style={'color': 'black'}
        ),

        html.Label('Heating/Cooling:'),
        dcc.Dropdown(
            id='heatingcooling-dropdown',
            options=[],
            value=None,
            style={'color': 'black'}
        ),

        html.Button('Optimize Selected', id='optimize-selected-button', n_clicks=0),
        html.Button('Optimize All', id='optimize-all-button', n_clicks=0),

        html.Div(id='output-container', children='Select options and click "Optimize All" to see results'),

        html.H2(children='Recommended Upgrades'),
        html.Div(id='recommended-upgrades', children='Loading recommended upgrades...'),

        dcc.Graph(
            id='savings-graph',
            style={'height': '500px'}  # We'll adjust via callback
        )
    ], style={'display': 'none'}),

    html.Div(style={'height': '50px'})
], fluid=True)

###############################################################################
# 3) Callbacks
###############################################################################

# Update images based on house type
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

# Handle PDF modal
@callback(
    Output("pdf-modal", "is_open"),
    Output("pdf-iframe", "src"),
    [Input("button-image1", "n_clicks"),
     Input("button-image2", "n_clicks"),
     Input("button-image3", "n_clicks"),
     Input("close-pdf-modal", "n_clicks")],
    [State("pdf-modal", "is_open"),
     State('house-type-dropdown', 'value')]
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
            return is_open, ""

# Handle house selection (clicking on images) and show the area choice
@callback(
    Output('selected-house', 'data'),
    Output('area-selection', 'style'),
    Output('ml-area-slider-container', 'style'),
    Input('image1', 'n_clicks'),
    Input('image2', 'n_clicks'),
    Input('image3', 'n_clicks'),
    State('area-input-mode', 'value')
)
def select_house(n_clicks_img1, n_clicks_img2, n_clicks_img3, area_mode):
    ctx = dash.callback_context
    if not ctx.triggered:
        # If no click yet, hide both area selection and slider
        return dash.no_update, {'display': 'none'}, {'display': 'none'}
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'image1' and n_clicks_img1:
            selected_house = 'house1'
        elif button_id == 'image2' and n_clicks_img2:
            selected_house = 'house2'
        elif button_id == 'image3' and n_clicks_img3:
            selected_house = 'house3'
        else:
            return dash.no_update, {'display': 'none'}, {'display': 'none'}

        # Show the correct area input widget based on area_mode
        if area_mode == 'predefined':
            return selected_house, {'display': 'block'}, {'display': 'none'}
        else:
            return selected_house, {'display': 'none'}, {'display': 'block'}

# Show/hide the Predefined dropdown vs. the Slider based on radio choice
@callback(
    Output('area-selection', 'style', allow_duplicate=True),
    Output('ml-area-slider-container', 'style', allow_duplicate=True),
    Input('area-input-mode', 'value'),
    prevent_initial_call=True
)
def toggle_area_input_widgets(area_mode):
    if area_mode == 'predefined':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}

# Display the slider value
@callback(
    Output('ml-slider-value', 'children'),
    Input('ml-area-slider', 'value')
)
def update_slider_label(value):
    return f"Selected area: {value} ft²"

# Callback to set the chosen area in a store (either from the dropdown or slider)
@callback(
    Output('selected-area', 'data'),
    Input('area-dropdown', 'value'),
    Input('ml-area-slider', 'value'),
    State('area-input-mode', 'value'),
    prevent_initial_call=True
)
def store_selected_area(predef_area, slider_area, area_mode):
    if area_mode == 'predefined':
        return predef_area  # e.g., "1000" or "2000"
    else:
        return str(slider_area)  # Convert to string just to keep things consistent

###############################################################################
# 4) Load data into dropdowns or (for ML) predict them using SVM
###############################################################################
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
    Input('selected-area', 'data'),
    Input('house-type-dropdown', 'value'),
    State('area-input-mode', 'value'),
    prevent_initial_call=True
)
def load_or_predict_factors(selected_house, area, house_type, area_mode):
    """
    If area_mode == 'predefined', we read the CSV for that area (1000 or 2000)
    and populate the factor dropdowns with real data from the CSV.
    If area_mode == 'ml', we use the trained SVM to predict a factor combination
    for the user-specified (slider) area, then we set the dropdown values to
    the predicted combination. For consistency, we'll still gather all unique
    options from the 1000/2000 CSV, so the user can manually change them if desired.
    """
    if not selected_house or not area or not house_type:
        return {'display': 'none'}, [], [], [], None, None, None, None

    # 1) Build the prefix (house_type) and base_name (from house1, house2, house3)
    prefix = '' if house_type == 'Rancher' else f'{house_type}-'
    base_name = house_to_basename_map.get(selected_house, '')

    # If we fail to find valid combos, hide everything
    if not base_name:
        return {'display': 'none'}, [], [], [], None, None, None, None

    # 2) Attempt to load the CSV for the "area" the user selected (only relevant if predefined)
    file_path = f'assets/{prefix}{base_name}_{area}SF.csv'
    df = pd.DataFrame()
    try:
        if area_mode == 'predefined':
            df = pd.read_csv(file_path)
        else:
            # We won't load the single CSV for the custom area;
            # we only load data from the 1000/2000 CSVs to populate the "options".
            # So let's just load 1000 or 2000 as a fallback. We'll pick 1000 for simplicity.
            # (Alternatively, you could load both 1000 and 2000 and merge.)
            fallback_file_path = f'assets/{prefix}{base_name}_1000SF.csv'
            df = pd.read_csv(fallback_file_path)
    except FileNotFoundError:
        # If the file doesn't exist, hide everything
        return {'display': 'none'}, [], [], [], None, None, None, None

    # 3) Gather unique values for the three factor dropdowns
    windows_options = sorted(df['input|Opt-Windows'].unique())
    foundationwall_options = sorted(df['input|Opt-FoundationWallExtIns'].unique())
    heatingcooling_options = sorted(df['input|Opt-Heating-Cooling'].unique())

    windows_options_dd = [{'label': val, 'value': val} for val in windows_options]
    foundationwall_options_dd = [{'label': val, 'value': val} for val in foundationwall_options]
    heatingcooling_options_dd = [{'label': val, 'value': val} for val in heatingcooling_options]

    # Default each dropdown to the first in the list
    # (We may override these if we do an ML prediction)
    windows_value = windows_options[0] if windows_options else None
    foundationwall_value = foundationwall_options[0] if foundationwall_options else None
    heatingcooling_value = heatingcooling_options[0] if heatingcooling_options else None

    # 4) If area_mode == 'ml', we do an SVM prediction for the user’s slider area
    if area_mode == 'ml':
        models = get_svm_models_for(house_type, selected_house)
        if models is not None:
            # Convert area to float or int
            area_numeric = float(area)
            X_new = np.array([[area_numeric]])  # shape (1,1)

            # Predict
            pred_win_num = models['svm_windows'].predict(X_new)[0]
            pred_fnd_num = models['svm_foundation'].predict(X_new)[0]
            pred_heat_num = models['svm_heating'].predict(X_new)[0]

            # Convert numeric back to labels
            windows_value = models['le_windows'].inverse_transform([pred_win_num])[0]
            foundationwall_value = models['le_foundation'].inverse_transform([pred_fnd_num])[0]
            heatingcooling_value = models['le_heating'].inverse_transform([pred_heat_num])[0]

    # 5) Serialize the DataFrame for later use in cost calculations
    data_json = df.to_json(date_format='iso', orient='split')

    # 6) Show the options container
    return (
        {'display': 'block'},
        windows_options_dd,
        foundationwall_options_dd,
        heatingcooling_options_dd,
        windows_value,
        foundationwall_value,
        heatingcooling_value,
        data_json
    )

###############################################################################
# 5) Optimization logic and final callbacks
###############################################################################
def find_best_upgrades(df, user_selections, selected_component=None):
    """
    Provided in your original code, unchanged.
    """
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
        if current_df.empty or cost_col not in df.columns:
            # If we can't find any row with the user's selection (or cost_col missing),
            # skip it gracefully.
            results.append({
                'Component': component.split('|')[1],
                'Current Value': current_value,
                'Best Upgrade': 'No Data',
                'Yearly Saving': 0,
                'Saving 15 Years': 0,
                'Net Saving': 0
            })
            continue

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
                if upgrade_df.empty or cost_col not in upgrade_df.columns:
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

# "Optimize All" button
@callback(
    Output('output-container', 'children'),
    Output('recommended-upgrades', 'children'),
    Output('savings-graph', 'figure'),
    Output('savings-graph', 'style'),
    Input('optimize-all-button', 'n_clicks'),
    State('windows-dropdown', 'value'),
    State('foundationwall-dropdown', 'value'),
    State('heatingcooling-dropdown', 'value'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def optimize_all(n_clicks, windows, foundationwall, heatingcooling, data_json):
    if n_clicks > 0 and data_json:
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
                    f"For {row['Component']} we recommend switching from {row['Current Value']} to "
                    f"{row['Best Upgrade']} which saves {row['Yearly Saving']:.2f} CAD annually "
                    f"and {row['Saving 15 Years']:.2f} CAD in 15 years."
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
            barmode='group',
            height=500,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        figure = go.Figure(data=savings_data, layout=savings_layout)
        graph_style = {'height': '500px', 'width': '100%'}

        current_df = df[
            (df['input|Opt-Windows'] == user_selections['input|Opt-Windows']) &
            (df['input|Opt-FoundationWallExtIns'] == user_selections['input|Opt-FoundationWallExtIns']) &
            (df['input|Opt-Heating-Cooling'] == user_selections['input|Opt-Heating-Cooling'])
        ]
        if current_df.empty or 'output|Util-Bill-Net' not in current_df.columns:
            total_cost = 0
        else:
            total_cost = current_df['output|Util-Bill-Net'].mode()[0]

        return (
            f"Total Cost: {total_cost:.2f} CAD",
            html.Div(recommended_text_elements),
            figure,
            graph_style
        )

    return 'Select options and click "Optimize All" to see results', 'Loading recommended upgrades...', {}, {}

# "Optimize Selected" button
@callback(
    Output('output-container', 'children', allow_duplicate=True),
    Output('recommended-upgrades', 'children', allow_duplicate=True),
    Output('savings-graph', 'figure', allow_duplicate=True),
    Output('savings-graph', 'style', allow_duplicate=True),
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
                    f"For {row['Component']} we recommend switching from {row['Current Value']} to "
                    f"{row['Best Upgrade']} which saves {row['Yearly Saving']:.2f} CAD annually "
                    f"and {row['Saving 15 Years']:.2f} CAD in 15 years."
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
            barmode='group',
            height=500,
            width=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        figure = go.Figure(data=savings_data, layout=savings_layout)

        graph_style = {'height': '500px', 'width': '600px'}

        current_df = df[
            (df['input|Opt-Windows'] == user_selections['input|Opt-Windows']) &
            (df['input|Opt-FoundationWallExtIns'] == user_selections['input|Opt-FoundationWallExtIns']) &
            (df['input|Opt-Heating-Cooling'] == user_selections['input|Opt-Heating-Cooling'])
        ]
        if current_df.empty or 'output|Util-Bill-Net' not in current_df.columns:
            total_cost = 0
        else:
            total_cost = current_df['output|Util-Bill-Net'].mode()[0]

        return (
            f"Total Cost: {total_cost:.2f} CAD",
            html.Div(recommended_text_elements),
            figure,
            graph_style
        )

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update
