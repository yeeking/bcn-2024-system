import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import threading
import time
import datetime

# Clock class that runs in a background thread and calls a UI update callback
class Clock:
    def __init__(self):
        self.running = True
        self.current_time = datetime.datetime.now().strftime("%H:%M:%S")
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
    html.H1("Real-Time Clock with Background Thread and Callback"),
    html.Div(id='time-display', style={'font-size': '48px'}),
    dcc.Interval(
        id='update-time-trigger',
        interval=1000,  # in milliseconds (1 second)
        n_intervals=0
    )
])

@app.callback(Output('time-display', 'children'),[Input('update-time-trigger', 'n_intervals')])
def update_time(n):
    return f"The current time is: {clock.get_time()}"
    
# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')

