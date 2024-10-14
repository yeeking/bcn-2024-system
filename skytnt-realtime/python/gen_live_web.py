import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import threading
import time
import datetime

import os
from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer
from gen_utils import ImproviserAgent, MIDIScheduler, ModelHandler
import mido


# Dash app setup
app = dash.Dash(__name__)


# Dash app layout
app.layout = html.Div([
    html.H1("Barcelona System 2024"),
    html.Div(id='improviser-status-display', style={'font-size': '48px'}),
    dcc.Interval(
        id='update-time-trigger',
        interval=1000,  # in milliseconds (1 second)
        n_intervals=0
    ), 
    html.Button('Start improviser', id='start-button', n_clicks=0),
    html.Div(id='improviser-start-status', style={'font-size': '24px', 'color': 'green'}),
    html.Button('Stop improviser', id='stop-button', n_clicks=0),
    html.Div(id='improviser-stop-status', style={'font-size': '24px', 'color': 'red'})

])


@app.callback(Output('improviser-status-display', 'children'),[Input('update-time-trigger', 'n_intervals')])
def update_time(n):
    return f"The current time is: {improviser.get_status()}"

# Callback to stop the improviser when the button is clicked
@app.callback(
    Output('improviser-stop-status', 'children'),
    [Input('stop-button', 'n_clicks')]
)
def stop_improviser_button_clicked(n_clicks):
    if n_clicks > 0:
        improviser.stop()
        return "improviser has been stopped."
    return ""

# Callback to stop the improviser when the button is clicked
@app.callback(
    Output('improviser-start-status', 'children'),
    [Input('start-button', 'n_clicks')]
)
def start_improviser_button_clicked(n_clicks):
    if n_clicks > 0:
        improviser.start()
        return "improviser has been started."
    return ""



ckpt = "../../trained-models/skytnt/version_703-la-hawthorne-finetune.ckpt"

assert os.path.exists(ckpt), "Cannot find checkpoint file " + ckpt

tokenizer = MIDITokenizer()
model = MIDIModel(tokenizer).to(device='cuda')
ModelHandler.load_model(ckpt, model)


## this commented out one was the version mark first used on 10th Oct '24
## with 32 memory length and 5 second auto-regen mode
## and the la->hawthorne fine tune model: version_703-la-hawthorne-finetune.ckpt 
#improviser = ImproviserAgent(memory_length=32, model=model, tokenizer=tokenizer, test_mode=False) 
improviser = ImproviserAgent(memory_length=32, model=model, tokenizer=tokenizer, test_mode=False) 

improviser.initMIDI() # select MIDI inputs and outputs
# improviser.start() # start responding to MIDI input 


# Run the Dash app
if __name__ == '__main__':


    # try:
    #     # Block main thread, simulate waiting for user input or other tasks
    #     input("Press Enter to stop the model runner...\n")
    # except KeyboardInterrupt:
    #     print("Keyboard interrupt received.")
        
    # improviser.stop()

    app.run_server(debug=True, host='0.0.0.0')
    

