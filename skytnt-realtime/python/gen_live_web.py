import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import threading
import time
import datetime

import os
from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer
from gen_utils import ImproviserAgent, MIDIScheduler, ModelHandler
import mido


# Dash app setup
# app = dash.Dash(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Dash app layout
app.layout = html.Div([
    html.H1("Barcelona System 2024"),
    html.Div(id='improviser-status-display', style={'font-size': '48px'}),
    dcc.Interval(
        id='update-time-trigger',
        interval=1000,  # in milliseconds (1 second)
        n_intervals=0
    ), 

    html.Div([
        html.Label("Set Input Length"),
        dcc.Slider(
            id='input-length-slider',
            min=0,
            max=4,
            marks={i: str(v) for i, v in enumerate([8, 16, 32, 64, 128])},
            value=0,  # Default value: corresponding to 8
        ),
    ], style={'margin-bottom': '50px'}),
    
    html.Div([
        html.Label("Set Output Length"),
        dcc.Slider(
            id='output-length-slider',
            min=0,
            max=4,
            marks={i: str(v) for i, v in enumerate([8, 16, 32, 64, 128])},
            value=0,  # Default value: corresponding to 8
        ),
    ]),
    
    html.Button('Start improviser', id='start-button', n_clicks=0),
    html.Div(id='improviser-start-status', style={'font-size': '24px', 'color': 'green'}),
    html.Button('Stop improviser', id='stop-button', n_clicks=0),
    html.Div(id='improviser-stop-status', style={'font-size': '24px', 'color': 'red'})

])


@app.callback(Output('improviser-status-display', 'children'),[Input('update-time-trigger', 'n_intervals')])
def update_time(n):
    return f"Improviser status: {improviser.get_status()}"

# Callback to stop the improviser when the button is clicked
@app.callback(
    Output('improviser-stop-status', 'children'),
    [Input('stop-button', 'n_clicks')]
)
def stop_improviser_button_clicked(n_clicks):
    if n_clicks > 0:
        improviser.stop()
        # return "improviser has been stopped."
        return ""
    return ""

# Callback to stop the improviser when the button is clicked
@app.callback(
    Output('improviser-start-status', 'children'),
    [Input('start-button', 'n_clicks')]
)
def start_improviser_button_clicked(n_clicks):
    if n_clicks > 0:
        improviser.initMIDI()
        improviser.start()
        return ""
    return ""

# Callback to handle input length slider
@app.callback(
    Output('input-length-slider', 'value'),
    [Input('input-length-slider', 'value')]
)
def update_input_length(slider_val):
    slider_val = int(slider_val)
    length = [8, 16, 32, 64, 128][slider_val]
    improviser.setInputLength(length)
    return slider_val

# Callback to handle output length slider
@app.callback(
    Output('output-length-slider', 'value'),
    [Input('output-length-slider', 'value')]
)
def update_output_length(slider_val):
    slider_val = int(slider_val)
    length = [8, 16, 32, 64, 128][slider_val]
    improviser.setOutputLength(length)
    return slider_val

# Run the Dash app
if __name__ == '__main__':
    ckpt = "../../trained-models/skytnt/version_703-la-hawthorne-finetune.ckpt"

    assert os.path.exists(ckpt), "Cannot find checkpoint file " + ckpt

    tokenizer = MIDITokenizer()
    model = MIDIModel(tokenizer).to(device='cuda')
    ModelHandler.load_model(ckpt, model)

    ## this commented out one was the version mark first used on 10th Oct '24
    ## with 32 memory length and 5 second auto-regen mode
    ## and the la->hawthorne fine tune model: version_703-la-hawthorne-finetune.ckpt 
    #improviser = ImproviserAgent(memory_length=32, model=model, tokenizer=tokenizer, test_mode=False) 
    improviser = ImproviserAgent(input_length=32, output_length=32, model=model, tokenizer=tokenizer, test_mode=False) 

    # improviser.initMIDI() # select MIDI inputs and outputs
    # improviser.start() # start responding to MIDI input 
    app.run_server(debug=True, host='0.0.0.0')
    

