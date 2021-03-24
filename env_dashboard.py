import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px

from environments.noise_generation import NoiseGenerator
from environments.obstacle_generation import ObstacleMapGenerator
from environments.env_representation import EnvironmentRepresentation

SAVE_PATH = "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/"

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash("ENVIRONMENT GENERATION", external_stylesheets=external_stylesheets)

terrain_gen = NoiseGenerator()
terrain_fig = px.imshow(terrain_gen.generate_noise_map(), binary_string=True)

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
                        {'label': str(2**i), 'value': 2**i} for i in range(3, 10)
                    ],
                    value=terrain_gen.dim[0]
                ),
                html.Span("frequency - x"),
                dcc.Dropdown(
                    id="terrain_map_freq_x",
                    options=[
                        {'label': str(2**i), 'value': 2**i} for i in range(4)
                    ],
                    value=terrain_gen.res[0]
                ),
                html.Span("frequency - y"),
                dcc.Dropdown(
                    id="terrain_map_freq_y",
                    options=[
                        {'label': str(2**i), 'value': 2**i} for i in range(4)
                    ],
                    value=terrain_gen.res[1]
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
    Input(component_id="generate_terrain", component_property="n_clicks"),
    Input(component_id="terrain_map_size", component_property="value"),
    Input(component_id="terrain_map_freq_x", component_property="value"),
    Input(component_id="terrain_map_freq_y", component_property="value")
)
def update_terrain_map(n_clicks, size, freq_x, freq_y):
    if size is not None:
        terrain_gen.dim = (size, size)
    if freq_x is not None and freq_y is not None:
        terrain_gen.res = (freq_x, freq_y)

    return px.imshow(terrain_gen.generate_noise_map(), binary_string=True)


obstacle_gen = ObstacleMapGenerator()
obstacle_features = obstacle_gen.generate_obstacle_map()
obstacle_dict = {
    "map": obstacle_features[0],
    "free_tiles": obstacle_features[1],
    "start_pos": obstacle_features[2]
}
obstacle_fig = px.imshow(obstacle_features[0])

obstacle_component = html.Div(
    children=[
        dcc.Graph(
            id="obstacle_map",
            figure=obstacle_fig
        ),
        html.Div(
            children=[
                html.Button(
                    id="generate_obstacles",
                    n_clicks=0,
                    children="GENERATE OBSTACLE MAP"
                ),
                html.Span(
                    id="nb_free_tiles",
                    children=f"nb free tiles: {obstacle_features[1]}"
                ),
                html.Span("map size"),
                dcc.Dropdown(
                    id="obstacle_map_size",
                    options=[
                        {'label': str(2**i), 'value': 2**i} for i in range(3, 10)
                    ],
                    value=obstacle_gen.dim[0]
                ),
                html.Span("frequency - x"),
                dcc.Dropdown(
                    id="obstacle_map_freq_x",
                    options=[
                        {'label': str(2**i), 'value': 2**i} for i in range(4)
                    ],
                    value=terrain_gen.res[0]
                ),
                html.Span("frequency - y"),
                dcc.Dropdown(
                    id="obstacle_map_freq_y",
                    options=[
                        {'label': str(2 ** i), 'value': 2 ** i} for i in range(4)
                    ],
                    value=terrain_gen.res[1]
                ),
                html.Span("fill ratio"),
                dcc.Slider(
                    id='obstacle_fill_ratio',
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    value=obstacle_gen.fill_ratio,
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
    Output(component_id="obstacle_map", component_property="figure"),
    Output(component_id="nb_free_tiles", component_property="children"),
    Input(component_id="generate_obstacles", component_property="n_clicks"),
    Input(component_id="obstacle_map_size", component_property="value"),
    Input(component_id="obstacle_map_freq_x", component_property="value"),
    Input(component_id="obstacle_map_freq_y", component_property="value"),
    Input(component_id="obstacle_fill_ratio", component_property="drag_value")
)
def update_terrain_map(n_clicks, size, freq_x, freq_y, fill_ratio):
    if size is not None:
        obstacle_gen.set_dimension((size, size))
    if freq_x is not None and freq_y is not None:
        obstacle_gen.set_frequency((freq_x, freq_y))
    if fill_ratio is not None:
        obstacle_gen.fill_ratio = fill_ratio

    obstacle_features = obstacle_gen.generate_obstacle_map()
    obstacle_dict["map"] = obstacle_features[0]
    obstacle_dict["free_tiles"] = obstacle_features[1]
    obstacle_dict["start_pos"] = obstacle_features[2]

    ### uncomment to show start positions
    # obst_map = obstacle_dict["map"]
    # list1 , list2 = zip(*obstacle_dict["start_pos"])
    # obst_map[(list1, list2)] = 0.5

    return px.imshow(obstacle_dict["map"]), f"nb free tiles: {obstacle_dict['free_tiles']}"


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
        env_repr = EnvironmentRepresentation()
        env_repr.obstacle_map = obstacle_dict["map"]
        env_repr.start_positions = obstacle_dict["start_pos"]
        env_repr.nb_free_tiles = obstacle_dict["free_tiles"]
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