import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import numpy as np

from environments.env_generation import GeneralEnvironmentGenerator

SAVE_PATH = "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/MAPS/"

START_DIMENSION = 16
START_RES_T = 2
START_RES_O = 2
START_FILL_RATIO = 0.17

EXTRA_SPACING = 8

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash("ENVIRONMENT GENERATION", external_stylesheets=external_stylesheets)

env_generator = GeneralEnvironmentGenerator((START_DIMENSION, START_DIMENSION))
env_generator.set_height_frequency((START_RES_T, START_RES_T))
env_generator.set_obstacle_frequency((START_RES_O, START_RES_O))

start_representation = env_generator.generate_environment(extra_spacing=EXTRA_SPACING)

MAP = {
    "representation": start_representation,
    "obstacle_map": start_representation.get_obstacle_map(),
    "terrain_map": start_representation.get_terrain_map(),
    "nb_free_tiles": start_representation.get_nb_free_tiles()
}

"""
TERRAIN
"""
terrain_fig = px.imshow(MAP["terrain_map"], binary_string=True)

terrain_component = html.Div(
    children=[
        dcc.Graph(
            id="terrain_map",
            figure=terrain_fig
        ),
        html.Div(
            children=[
                html.Button(
                    id="generate_terrain",
                    n_clicks=0,
                    children="GENERATE TERRAIN MAP"
                ),
                html.Span("map size"),
                dcc.Dropdown(
                    id="terrain_map_size",
                    options=[
                        {'label': str(2*i), 'value': 2*i} for i in range(4, 11)
                    ],
                    value=START_DIMENSION
                ),
                html.Span("extra spacing"),
                dcc.Dropdown(
                    id="extra_spacing",
                    options=[
                        {'label': str(2 * i), 'value': 2 * i} for i in range(0, 11)
                    ],
                    value=EXTRA_SPACING
                ),
                html.Span("frequency - x"),
                dcc.Dropdown(
                    id="terrain_map_freq_x",
                    options=[
                        {'label': str(i), 'value': i} for i in range(1, 21)
                    ],
                    value=START_RES_T
                ),
                html.Span("frequency - y"),
                dcc.Dropdown(
                    id="terrain_map_freq_y",
                    options=[
                        {'label': str(i), 'value': i} for i in range(1,21)
                    ],
                    value=START_RES_T
                ),
                html.Span("save map"),
                dcc.Input(
                    id="terrain_save_name",
                    type="text"
                ),
                html.Button(
                    "SAVE",
                    id="terrain_save_btn",
                    n_clicks=0
                )
            ],
            style={
                'display': 'flex',
                'flex-direction': 'column'
            }
        )
    ],
    style={
        'display': 'flex',
        'align-items': 'center',
        'justify-content': 'center'
    }
)


@app.callback(
    Output(component_id="terrain_map", component_property="figure"),
    Output(component_id="obstacle_map", component_property="figure"),
    Output(component_id="nb_free_tiles", component_property="children"),
    Input(component_id="generate_terrain", component_property="n_clicks"),
    Input(component_id="terrain_map_size", component_property="value"),
    Input(component_id="terrain_map_freq_x", component_property="value"),
    Input(component_id="terrain_map_freq_y", component_property="value"),
    Input(component_id="obstacle_map_freq_x", component_property="value"),
    Input(component_id="obstacle_map_freq_y", component_property="value"),
    Input(component_id="obstacle_fill_ratio", component_property="drag_value"),
    Input(component_id="extra_spacing", component_property="value")
)
def update_terrain_map(n_clicks, size, freq_xt, freq_yt, freq_xo, freq_yo, fill_ratio, spacing):
    if size is not None:
        env_generator.set_dimension((size, size))
    if freq_xt is not None and freq_yt is not None:
        env_generator.set_height_frequency((freq_xt, freq_yt))
    if freq_xo is not None and freq_yo is not None:
        env_generator.set_obstacle_frequency((freq_xo, freq_yo))
    if fill_ratio is not None:
        env_generator.set_fill_ratio(fill_ratio)

    extra_spacing = 0
    if spacing is not None:
        extra_spacing = spacing

    n_representation = env_generator.generate_environment(extra_spacing)
    MAP["representation"] = n_representation
    MAP["obstacle_map"] = n_representation.get_obstacle_map()
    MAP["terrain_map"] = n_representation.get_terrain_map()
    MAP["nb_free_tiles"] = n_representation.get_nb_free_tiles()

    return [px.imshow(MAP["terrain_map"], binary_string=True),
            px.imshow(MAP["obstacle_map"]),
            f"nb free tiles: {MAP['nb_free_tiles']}"]


@app.callback(
    Output(component_id='terrain_save_btn', component_property='children'),
    Input(component_id='terrain_save_btn', component_property='n_clicks'),
    State(component_id='terrain_save_name', component_property='value')
)
def save_height_map(n_clicks, name):
    if name is not None:
        print(name)
        np.save(f"{SAVE_PATH}{name}.npy", MAP["terrain_map"])
    return "SAVE"

"""
OBSTACLES
"""
obstacle_fig = px.imshow(MAP["obstacle_map"])

obstacle_component = html.Div(
    children=[
        dcc.Graph(
            id="obstacle_map",
            figure=obstacle_fig
        ),
        html.Div(
            children=[
                html.Span(
                    id="nb_free_tiles",
                    children=f"nb free tiles: TODO <free tiles>"
                ),
                html.Span("frequency - x"),
                dcc.Dropdown(
                    id="obstacle_map_freq_x",
                    options=[
                        {'label': str(i), 'value': i} for i in range(2, 21)
                    ],
                    value=START_RES_O
                ),
                html.Span("frequency - y"),
                dcc.Dropdown(
                    id="obstacle_map_freq_y",
                    options=[
                        {'label': str(i), 'value': i} for i in range(2,21)
                    ],
                    value=START_RES_O
                ),
                html.Span("agent size"),
                dcc.Dropdown(
                    id="agent_size",
                    options=[
                        {'label': str(2 * i + 1), 'value': 2 * i + 1} for i in range(5)
                    ],
                    value=1
                ),
                html.Span("fill ratio"),
                dcc.Slider(
                    id='obstacle_fill_ratio',
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    value=START_FILL_RATIO,
                ),
                html.Span("", id='obstacle_fill_ratio_span'),
                html.Span("save map"),
                dcc.Input(
                    id="obstacle_save_name",
                    type="text"
                ),
                html.Button(
                    "SAVE",
                    id="obstacle_save_btn",
                    n_clicks=0
                )
            ],
            style={
                'display': 'flex',
                'flex-direction': 'column'
            }
        )
    ],
    style={
        'display': 'flex',
        'align-items': 'center',
        'justify-content': 'center'
    }
)


@app.callback(
    Output(component_id='obstacle_fill_ratio_span', component_property='children'),
    Input(component_id='obstacle_fill_ratio', component_property='drag_value')
)
def update_fill_ratio_span2(fill_ratio):
    return str(fill_ratio)


@app.callback(
    Output(component_id='obstacle_save_btn', component_property='children'),
    Input(component_id='obstacle_save_btn', component_property='n_clicks'),
    State(component_id='obstacle_save_name', component_property='value')
)
def save_obstacle_map(n_clicks, name):
    if name is not None:
        print(name)
        env_repr = MAP["representation"]
        env_repr.save(SAVE_PATH, name)
    return "SAVE"


app.layout = html.Div(
    children=[
        html.H1(
            children="Environment Dashboard",
            style={
                'textAlign': 'center'
            }
        ),
        html.Div(
            children="Visualisation of the environment generation parameters.",
            style={
                'textAlign': 'center'
            }
        ),
        terrain_component,
        obstacle_component
    ]
)


if __name__ == '__main__':
    app.run_server(debug=True)