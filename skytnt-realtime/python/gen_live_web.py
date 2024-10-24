import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

# import threading
# import time
# import datetime

import os
from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer
from gen_utils import ImproviserAgent, MIDIScheduler, ModelHandler
import mido
import pandas as pd 


# # Example data
# musical_events = [

# ]
# # Convert list of events to a pandas DataFrame
# data = pd.DataFrame(musical_events, columns=['event', 'start_time', 'duration', 'channel', 'note', 'velocity'])

# Create the piano roll figure
def create_piano_roll(data):
    traces = []
    for _, row in data.iterrows():
        traces.append(
            go.Scatter(
                x=[row['start_time'], row['start_time'] + row['duration']],
                y=[row['note'], row['note']],
                mode='lines',
                line=dict(width=10),
                name=f"Note {row['note']}"
            )
        )

    layout = go.Layout(
        title='Piano Roll',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Note', autorange='reversed'),
        showlegend=False
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig

# Dash app setup
# app = dash.Dash(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Dash app layout
app.layout = html.Div([
    # html.Div(id='improviser-status-display', style={'font-size': '12px'}),
    html.Div([
    html.H1("Barcelona System 2024"),
    dcc.Markdown(id='improviser-status-display', style={'font-size': '12px'}),
    dcc.Interval(
        id='update-time-trigger',
        interval=1000,  
        n_intervals=0
    )]), 
    dcc.Graph(id='piano-roll-plot'),
    dcc.Interval(
        id='update-interval',
        interval=1000,  # in milliseconds
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
    # Adding a checkbox (Checklist) to toggle self-listen mode
    html.Div([
        dcc.Checklist(
            id='feedback-checkbox',
            options=[{'label': 'Enable feedback', 'value': 'enable'}],
            value=[],  # Initially not checked
        )
    ]),
    # Adding a checkbox (Checklist) to toggle self-listen mode
    html.Div([
        # html.Label("Toggle Self-listen Mode"),
        dcc.Checklist(
            id='overlap-checkbox',
            options=[{'label': 'Enable overlap mode', 'value': 'enable'}],
            value=[],  # Initially not checked
        )
    ]),
    
    html.Button('Start improviser', id='start-button', n_clicks=0),
    html.Div(id='improviser-start-status'),
    html.Button('Reset improviser', id='reset-button', n_clicks=0),
    html.Div(id='improviser-reset-status'),
    html.Button('Stop improviser', id='stop-button', n_clicks=0),
    html.Div(id='improviser-stop-status')

])


# callback to periodically poll and update improviser status
@app.callback(Output('improviser-status-display', 'children'),[Input('update-time-trigger', 'n_intervals')])
def update_time(n):
    return f"Improviser status: {improviser.get_status(markdown_mode=True)}"

# callback to plot the piano roll data
@app.callback(
    Output('piano-roll-plot', 'figure'),
    [Input('update-interval', 'n_intervals')]
)
def update_piano_roll(n_intervals):
    musical_events = improviser.get_last_input()
    # Convert list of events to a pandas DataFrame
    data = pd.DataFrame(musical_events, columns=['event', 'start_time', 'duration', 'channel', 'note', 'velocity'])
    return create_piano_roll(data)

# Callback to start the improviser when the button is clicked
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

# Callback to reset the improviser's memory 
@app.callback(
    Output('improviser-reset-status', 'children'),
    [Input('reset-button', 'n_clicks')]
)
def reset_improviser_button_clicked(n_clicks):
    if n_clicks > 0:
        improviser.resetMemory()
        
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

# Callback to handle the feedback mode checkbox
@app.callback(
    Output('feedback-checkbox', 'value'),
    [Input('feedback-checkbox', 'value')]
)
def toggle_self_listen_mode(checkbox_value):
    enabled = 'enable' in checkbox_value
    improviser.setFeedbackMode(enabled)  # Set self-listen mode on improviser
    return checkbox_value

# Callback to handle the overlap mode
@app.callback(
    Output('overlap-checkbox', 'value'),
    [Input('overlap-checkbox', 'value')]
)
def toggle_overlap_mode(checkbox_value):
    enabled = 'enable' in checkbox_value
    improviser.setOverlapMode(enabled)  # Set self-listen mode on improviser
    return checkbox_value

# Run the Dash app
if __name__ == '__main__':
    #ckpt = "../../trained-models/skytnt-pre-trained-la-dataset.ckpt"
    ckpt = "../../trained-models/skytnt/version_703-la-hawthorne-finetune.ckpt"

    assert os.path.exists(ckpt), "Cannot find checkpoint file " + ckpt

    tokenizer = MIDITokenizer()
    model = MIDIModel(tokenizer).to(device='cpu')
    ModelHandler.load_model(ckpt, model)

    ## this commented out one was the version mark first used on 10th Oct '24
    ## with 32 memory length and 5 second auto-regen mode
    ## and the la->hawthorne fine tune model: version_703-la-hawthorne-finetune.ckpt 
    #improviser = ImproviserAgent(memory_length=32, model=model, tokenizer=tokenizer, test_mode=False) 
    improviser = ImproviserAgent(input_length=32, 
                                output_length=32, 
                                feedback_mode=True, 
                                allow_gen_overlap=False, 
                                model=model, tokenizer=tokenizer, test_mode=False) 
    # print(improviser.get_status())

    # improviser.initMIDI() # select MIDI inputs and outputs
    # improviser.start() # start responding to MIDI input 
    app.run_server(debug=True, host='0.0.0.0')
    

