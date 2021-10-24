import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Read CSV data from output file.
filename = "temperature_output.txt"
temp_df = pd.read_csv(filename, header=None)
temp_df = temp_df.iloc[:, 0:100]
print(temp_df.head())
print(temp_df.tail())

# Arange spatial (x) und time (y) values.
x, y = np.arange(0, len(temp_df.columns), 1), np.arange(0, len(temp_df), 1)
z = temp_df.values

# Create 3D surface plot.
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
#fig.update_traces(contours_x=dict(show=True, usecolormap=False,
#                                  highlightcolor="black", project_z=False))
#fig.update_traces(contours_y=dict(show=True, usecolormap=False,
#                                  highlightcolor="black", project_z=False))
fig.update_traces(contours_z=dict(show=True, usecolormap=False,
                                  highlightcolor="black", project_z=False))
fig.update_layout(title='Temprature Distribution', autosize=False,
                  width=1000, height=1000, 
                  scene = dict(
                    xaxis_title='x in m',
                    yaxis_title='t in s',
                    zaxis_title='Temperature in C'),
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()

