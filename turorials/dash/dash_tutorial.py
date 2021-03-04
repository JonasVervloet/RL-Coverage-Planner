import math
import dash
import  dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import numpy as np

from env_generation.height_generator import Height_Generator
from env_generation.obstacle_generator import ObstacleGenerator
from env_generation.obstacle_generator import ObstacleGenerator2

SAVE_PATH = "D:/Documenten/Studie/2020-2021/Masterproef/Reinforcement-Learner-For-Coverage-Path-Planning/data/"

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

height_gen = Height_Generator()
height_grid = height_gen.generate_grid()
height_fig = px.imshow(height_grid, binary_string=True)

obstacle_gen = ObstacleGenerator()
obstacle_grid = obstacle_gen.generate_grid()
obstacle_fig = px.imshow(obstacle_grid)

obstacle_gen2 = ObstacleGenerator2()
obstacle_grid2 = obstacle_gen2.generate_grid()
obstacle_fig2 = px.imshow(obstacle_grid2)

app = dash.Dash("DASH TUTORIAL", external_stylesheets=external_stylesheets)


class GridManager:
    def __init__(self, grid=None):
        self.grid = grid


grid_mgr = GridManager(obstacle_grid2)


@app.callback(
    Output(component_id='height_grid', component_property='figure'),
    Input(component_id='height_reset_btn', component_property='n_clicks'),
    Input(component_id='height_size_drop_down', component_property='value'),
    Input(component_id='height_freq_drop_down', component_property='value')
)
def update_height_grid(nb_clicks, grid_size, frequency):
    if grid_size is not None:
        height_gen.dim = (grid_size, grid_size)
    if frequency is not None:
        height_gen.res = (frequency, frequency)

    grid = height_gen.generate_grid()
    return px.imshow(grid, binary_string=True)


@app.callback(
    Output(component_id='height_freq_drop_down', component_property='options'),
    Input(component_id='height_size_drop_down', component_property='value')
)
def update_height_freq_drop_down(height_grid_size):
    return [
        {'label': str(2**i), 'value': 2**i} for i in range(int(math.log(height_grid_size, 2) + 1))
    ]


@app.callback(
    Output(component_id='obstacle_grid', component_property='figure'),
    Input(component_id='obstacle_reset_btn', component_property='n_clicks'),
    Input(component_id='obstacle_size_drop_down', component_property='value'),
    Input(component_id='fill_ratio_slider', component_property='drag_value'),
    Input(component_id='nb_smoothing_slider', component_property='drag_value')
)
def update_obstacle_grid(nb_clicks, obstacle_size, fill_ratio, nb_smoothing):
    obstacle_gen.dim = (obstacle_size, obstacle_size)
    if fill_ratio is not None:
        obstacle_gen.fill_ratio = fill_ratio
    if nb_smoothing is not None:
        obstacle_gen.nb_smoothing = nb_smoothing

    grid = obstacle_gen.generate_grid()
    return px.imshow(grid)


@app.callback(
    Output(component_id='fill_ratio_span', component_property='children'),
    Input(component_id='fill_ratio_slider', component_property='drag_value')
)
def update_fill_ratio_span(fill_ratio):
    return str(fill_ratio)


@app.callback(
    Output(component_id='nb_smoothing_span', component_property='children'),
    Input(component_id='nb_smoothing_slider', component_property='drag_value')
)
def update_fill_ratio_span(nb_smoothing):
    return str(nb_smoothing)


@app.callback(
    Output(component_id='obstacle_grid2', component_property='figure'),
    Input(component_id='obstacle_reset_btn2', component_property='n_clicks'),
    Input(component_id='obstacle_size_drop_down2', component_property='value'),
    Input(component_id='fill_ratio_slider2', component_property='drag_value'),
    Input(component_id='boundaries_radio', component_property='value')
)
def update_obstacle_grid(nb_clicks, obstacle_size, fill_ratio, radio_value):
    if obstacle_size is not None:
        obstacle_gen2.set_dim((obstacle_size, obstacle_size))
    if fill_ratio is not None:
        obstacle_gen2.fill_ratio = fill_ratio
    if radio_value is not None:
        if radio_value == 'True':
            obstacle_gen2.boundaries = True
        else:
            obstacle_gen2.boundaries = False

    grid = obstacle_gen2.generate_grid()
    grid_mgr.grid = grid
    return px.imshow(grid)


@app.callback(
    Output(component_id='fill_ratio_span2', component_property='children'),
    Input(component_id='fill_ratio_slider2', component_property='drag_value')
)
def update_fill_ratio_span2(fill_ratio):
    return str(fill_ratio)


@app.callback(
    Output(component_id='hidden_div', component_property='children'),
    Input(component_id='save_map_btn', component_property='n_clicks'),
    State(component_id='map_save_name_input', component_property='value')
)
def save_obstacle_map(nb_clicks, file_name):
    if nb_clicks == 0:
        return ""

    if file_name is None or file_name == "":
        return ""

    np.save(SAVE_PATH + file_name, grid_mgr.grid)

    return ""


height_grid_component = html.Div(
    children=[
        dcc.Graph(
            figure=height_fig,
            id="height_grid"
        ),
        html.Div(
            children=[
                html.Button(
                    id="height_reset_btn",
                    n_clicks=0,
                    children="GENERATE HEIGHT MAP"
                ),
                html.Span("grid size"),
                dcc.Dropdown(
                    id='height_size_drop_down',
                    options=[
                        {'label': str(2**i), 'value': 2**i} for i in range(3, 10)
                    ],
                    value=16,
                ),
                html.Span("frequency"),
                dcc.Dropdown(
                    id='height_freq_drop_down',
                    options=[
                        {'label': str(2**i), 'value': 2**i} for i in range(4)
                    ],
                    value=2,
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

obstacle_grid_component = html.Div(
    children=[
        dcc.Graph(
            figure=obstacle_fig,
            id="obstacle_grid"
        ),
        html.Div(
            children=[
                html.Button(
                    id="obstacle_reset_btn",
                    n_clicks=0,
                    children="GENERATE OBSTACLES"
                ),
                html.Span("grid size"),
                dcc.Dropdown(
                    id='obstacle_size_drop_down',
                    options=[
                        {'label': str(2**i), 'value': 2**i} for i in range(3, 10)
                    ],
                    value=16,
                ),
                html.Span("fill ratio"),
                dcc.Slider(
                    id='fill_ratio_slider',
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    value=0.35,
                ),
                html.Span("", id='fill_ratio_span'),
                html.Span("nb smooth operations"),
                dcc.Slider(
                    id='nb_smoothing_slider',
                    min=0,
                    max=10,
                    step=1,
                    value=2,
                ),
                html.Span("", id='nb_smoothing_span')
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

obstacle_grid_component2 = html.Div(
    children=[
        dcc.Graph(
            figure=obstacle_fig2,
            id="obstacle_grid2"
        ),
        html.Div(
            children=[
                html.Button(
                    id="obstacle_reset_btn2",
                    n_clicks=0,
                    children="GENERATE OBSTACLES"
                ),
                html.Span("grid size"),
                dcc.Dropdown(
                    id='obstacle_size_drop_down2',
                    options=[
                        {'label': str(2**i), 'value': 2**i} for i in range(3, 10)
                    ],
                    value=16,
                ),
                html.Span("fill ratio"),
                dcc.Slider(
                    id='fill_ratio_slider2',
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    value=0.35,
                ),
                html.Span("", id='fill_ratio_span2'),
                html.Span("boundaries"),
                dcc.RadioItems(
                    id='boundaries_radio',
                    options=[
                        {'label': 'no', 'value': 'False'},
                        {'label': 'yes', 'value': 'True'}
                    ],
                    value='False',
                    labelStyle={'display': 'inline-block'}
                ),
                html.Span("save map"),
                dcc.Input(id="map_save_name_input", type="text"),
                html.Button("SAVE", id="save_map_btn", n_clicks=0)
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

app.layout = html.Div(
    children=[
        html.H1(
            children="Hello Dash",
            style={
                'textAlign': 'center',
                'color': colors['background']
            }
        ),
        html.Div(
            children="Dash: a web application framework for python.",
            style={
                'textAlign': 'center',
                'color': colors['background']
            }
        ),
        height_grid_component,
        obstacle_grid_component,
        obstacle_grid_component2,
        html.Div(
            id="hidden_div", style={"display": "none"}
        )
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)