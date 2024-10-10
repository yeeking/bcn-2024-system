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


# Clock class that runs in a background thread and calls a UI update callback
class Clock:
    def __init__(self):
        self.current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.running = False

    def start(self):
        # exit old thread if needed 
        if self.running:
            self.running = False
            if self.thread is not None:
                self.thread.join()
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        while self.running:
            self.current_time = datetime.datetime.now().strftime("%H:%M:%S")
            time.sleep(1)

    def get_time(self):
        return self.current_time 
    
    def stop(self):
        self.running = False

# Dash app setup
app = dash.Dash(__name__)

# Create the clock instance and pass it the update_ui_time callback
clock = Clock()

# Dash app layout
app.layout = html.Div([
    html.H1("Barcelona System 2024"),
    html.Div(id='time-display', style={'font-size': '48px'}),
    dcc.Interval(
        id='update-time-trigger',
        interval=1000,  # in milliseconds (1 second)
        n_intervals=0
    ), 
    html.Button('Start Clock', id='start-button', n_clicks=0),
    html.Div(id='clock-start-status', style={'font-size': '24px', 'color': 'green'}),
    html.Button('Stop Clock', id='stop-button', n_clicks=0),
    html.Div(id='clock-stop-status', style={'font-size': '24px', 'color': 'red'})

])


@app.callback(Output('time-display', 'children'),[Input('update-time-trigger', 'n_intervals')])
def update_time(n):
    return f"The current time is: {clock.get_time()}"

# Callback to stop the clock when the button is clicked
@app.callback(
    Output('clock-stop-status', 'children'),
    [Input('stop-button', 'n_clicks')]
)
def stop_clock_button_clicked(n_clicks):
    if n_clicks > 0:
        clock.stop()
        return "Clock has been stopped."
    return ""

# Callback to stop the clock when the button is clicked
@app.callback(
    Output('clock-start-status', 'children'),
    [Input('start-button', 'n_clicks')]
)
def start_clock_button_clicked(n_clicks):
    if n_clicks > 0:
        clock.start()
        return "Clock has been started."
    return ""

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')

