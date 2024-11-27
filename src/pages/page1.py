import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/")

layout = dbc.Container([
    html.H1('Energy Cost Estimator - Input Page'),

    dbc.Row([
        dbc.Col(html.Label('Enter your name:')),
        dbc.Col(dcc.Input(id='name-input', type='text', placeholder='Your Name')),
    ], className='mb-3'),

    dbc.Row([
        dbc.Col(html.Label('Enter your budget:')),
        dbc.Col(dcc.Input(id='budget-input', type='number', placeholder='Your Budget')),
    ], className='mb-3'),

    dbc.Row([
        dbc.Col(dcc.Link('Proceed to Component Selection', href='/recommendations', className='btn btn-primary')),
    ], className='mb-3')
])
