### Import Packages ###
from dash import Dash
from dash import dcc
from dash import html
from dash.dependencies import (
    Input,
    Output,
)
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import numpy as np

from igf.base import Identity
from igf.coulomb import CoulombHamiltonian
from igf.tbh import G_TBH_1D

# Make sure the bootstrap css styles are loaded
external_stylesheets = [dbc.themes.BOOTSTRAP]

# Create the Dash app
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=external_stylesheets,
    title="CT absorption",
)

# Input parameters - See also input parameters for update_visualization()
parameters = {
    "E": {"min": -10.0, "max": 10.0, "default": 0.0, "step": 0.1, "alias": "Energy"},
    "B": {"min": 0.0, "max": 10.0, "default": 2.0, "step": 0.1, "alias": "Bandwidth"},
    "delta": {"min": 0.0, "max": 5.0, "default": 0.5, "step": 0.1, "alias": "Detuning"},
    "A": {
        "min": 0.0,
        "max": 10.0,
        "default": 3.0,
        "step": 0.1,
        "alias": "Coulomb well depth",
    },
    "gamma": {
        "min": 0.01,
        "max": 1.0,
        "default": 0.1,
        "step": 0.01,
        "alias": "Linewidth",
    },
    "R": {
        "min": 0,
        "max": 30,
        "default": 15,
        "step": 1,
        "alias": "Truncation distance",
    },
}

# Inputs - convert all parameters to an input for generating a figure
callback_inputs = []
for name in parameters:
    callback_inputs.append(Input(name, "value"))

callback_inputs.append(Input("wrange", "value"))

# Sliders - create a slider for each parameter
input_sliders = []
for name in parameters:
    value = parameters[name]
    input_sliders.append(
        dbc.FormGroup(
            [
                dbc.Label(parameters[name]["alias"]),
                dcc.Slider(
                    id=name,
                    min=value["min"],
                    max=value["max"],
                    step=value["step"],
                    value=value["default"],
                    updatemode="drag",
                    tooltip={"always_visible": True, "placement": "right"},
                ),
            ]
        )
    )

# Create separate range slider
input_sliders.append(
    dbc.FormGroup(
        [
            dbc.Label("Range"),
            dcc.RangeSlider(
                id="wrange",
                min=-25,
                max=25,
                step=1,
                value=[-4, 6],
                tooltip={"always_visible": True, "placement": "right"},
            ),
        ]
    )
)

# Main figure creator
@app.callback(
    Output("ABS", "figure"),
    callback_inputs,
)
def update_visualization(E, B, delta, A, gamma, R, wrange):
    omega = np.linspace(wrange[0], wrange[1], 200)
    G = green_function(
        float(E), float(B), float(delta), float(A), float(gamma), int(R), omega
    )
    Gabs = DOS(omega, G, R=R)
    Gdos = DOS(omega, G)
    Abs = go.Figure()
    Abs.add_trace(go.Scatter(x=omega, y=np.real(Gabs), name="Absorption"))
    Abs.add_trace(go.Scatter(x=omega, y=np.real(Gdos), name="Density of States"))
    Abs.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        # paper_bgcolor="LightSteelBlue",
        height=700,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        xaxis_title="Energy",
        yaxis_title="Absorption / DOS",
    )

    return Abs


# Generate a Green function in matrix representation
def green_function(E, B, delta, A, gamma, R, w):

    # Create Green function
    G0 = G_TBH_1D(E + A + delta, B, R)
    Hp = CoulombHamiltonian(delta, A, R)
    Id = Identity(2 * R + 1)

    T = Hp / (Id - G0 * Hp)

    G = G0 + G0 * T * G0

    Green = np.zeros((len(w), 2 * R + 1, 2 * R + 1), dtype=np.complex_)
    for i in range(len(w)):
        Green[i, :, :] = G(w[i] + 1j * gamma)

    return Green


# Compute the density of states for a given Green function
def DOS(w, G, R=None):
    dos = np.zeros_like(w, dtype=np.float_)

    sG = np.shape(G)
    for i in range(len(w)):
        if R is None:
            dos[i] = -np.imag(np.trace(G[i])) / np.pi / sG[1]
        else:
            dos[i] = -np.imag(G[i, R, R]) / np.pi

    return dos


# Main jumbotron
jumbotron = dbc.Jumbotron(
    [
        html.H3(html.B("Theory of Condensed Matter Group"), className="display-5"),
        html.Hr(),
        html.H2("CT Absorption", className="display-4"),
        html.P(
            "How does the spectrum of a 1D charge transfer (CT) system look like?",
            className="lead blue",
        ),
    ]
)

# Page layout
page_container = html.Div(
    children=[
        html.Div(
            className="container",
            children=[
                # content will be rendered in this element
                jumbotron,
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [html.H4("Controls:"), *input_sliders],
                                body=True,
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            dcc.Loading(
                                children=dcc.Graph(id="ABS"),
                                parent_className="loading_wrapper",
                                style={"backgroundColor": "transparent"},
                            ),
                            # dcc.Graph(id="ABS"),
                            md=8,
                        ),
                    ]
                ),
            ],
        ),
    ]
)

### Set app layout to page container ###
app.layout = page_container  ### Assemble all layouts ###
app.validation_layout = html.Div(children=[page_container])

# Run server
if __name__ == "__main__":
    app.run_server(debug=True)
