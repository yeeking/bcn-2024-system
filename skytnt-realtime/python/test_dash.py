import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np

# Example data
musical_events = [
    ['note', 145, 78, 0, 72, 59],
    ['note', 212, 87, 0, 70, 66],
    ['note', 151, 190, 0, 58, 17],
    ['note', 298, 47, 0, 65, 58]
]

# Convert list of events to a pandas DataFrame
data = pd.DataFrame(musical_events, columns=['event', 'start_time', 'duration', 'channel', 'note', 'velocity'])

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
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='piano-roll-plot'),
    dcc.Interval(
        id='update-interval',
        interval=1000,  # in milliseconds
        n_intervals=0
    )
])

@app.callback(
    Output('piano-roll-plot', 'figure'),
    [Input('update-interval', 'n_intervals')]
)
def update_piano_roll(n_intervals):
    return create_piano_roll(data)

if __name__ == '__main__':
    app.run_server(debug=True)
