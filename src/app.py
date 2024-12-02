import dash
from dash import html, Dash
import dash_bootstrap_components as dbc

# Initialize the app with pages
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SUPERHERO])
server = app.server

# Header without navigation links
header = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("Energy Cost Estimator"),
            # Navigation links have been removed
        ],
        fluid=True,
    ),
    dark=True,
    color='dark'
)

# Main layout
app.layout = dbc.Container([header, dash.page_container], fluid=False)

if __name__ == '__main__':
    app.run_server(debug=True)