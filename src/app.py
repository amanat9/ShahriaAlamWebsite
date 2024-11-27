import dash
from dash import html, Dash
import dash_bootstrap_components as dbc

# Initialize the app with pages
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SUPERHERO])
server = app.server

# Header for navigation between pages
header = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("Energy Cost Estimator"),
            dbc.Nav([
                dbc.NavLink(page["name"], href=page["path"])
                for page in dash.page_registry.values()
                if not page["path"].startswith("/app")
            ])
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
