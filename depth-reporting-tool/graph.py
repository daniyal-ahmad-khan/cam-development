# import dash
# from dash import dcc, html
# import plotly.express as px
# from dash.dependencies import Input, Output
# import pandas as pd
# from datetime import datetime
# import time
# import random
# import os
# import threading
# import math
# from config import METER_TO_PIXEL

# # Ensure directory exists for saving graphs and data
# data_directory = 'detection_data'
# os.makedirs(data_directory, exist_ok=True)

# # # DataFrame to store detection data
# # data = {
# #     "timestamp": [],
# #     "bbox_area": [],
# #     "bbox_x": [],
# #     "bbox_y": []
# # }
# # df = pd.DataFrame(data)

# # Lock for thread-safe operations on the dataframe and file operations
# # lock = threading.Lock()

# # # Function to simulate detection, save data, and generate static graph hourly
# # def data_collection_and_graphing():
# #     last_saved_time = datetime.now()
# #     while True:
# #         now = datetime.now()
# #         bbox_x = random.randint(50, 150)  # Simulated bbox width
# #         bbox_y = random.randint(50, 150)  # Simulated bbox height
# #         bbox_area = bbox_x * bbox_y  # Calculate area
        
# #         new_data = {'timestamp': now, 'bbox_area': bbox_area, 'bbox_x': bbox_x, 'bbox_y': bbox_y}
# #         with lock:
# #             global df
# #             df.loc[len(df)] = new_data
# #             df.to_csv(f'{data_directory}/detections.csv', index=False)
        
# #         # Save a static graph every hour
# #         if (now - last_saved_time).total_seconds() >= 10:
# #             save_static_graph(df, f'{data_directory}/detections.html')
# #             last_saved_time = now
        
# #         time.sleep(5)  # Simulate every 5 seconds
# def determine_zone(z_point, radius1, radius2, radius3):
#     # x_point_pixel = math.trunc(x_point*m_to_pix)
#     z_point_pixel = math.trunc(z_point * METER_TO_PIXEL)
#     if z_point_pixel < radius1:
#         return "Danger Zone", "red"
#     elif z_point_pixel < radius2:
#         return "Warning Zone", "orange"
#     else:
#         return "Safe Zone", "green"
# def save_static_graph(filename):

#     local_df = pd.read_csv(f'{data_directory}/detections.csv')
#     zone_data = [determine_zone(z, radius1=50, radius2=100, radius3=150) for z in local_df['position_z']]
#     zones, colors = zip(*zone_data)
#     local_df['Zone'] = zones
#     local_df['color'] = colors
#     fig = px.scatter(local_df, x='timestamp', y='position_z', title='Real-Time Detections',
#                      hover_data={'label':True,'position_x': True, 'position_z': True, "Zone": True},
#                      color=local_df['color'],
#                      color_discrete_map={"red": "red", "orange": "orange", "green": "green"})
#     fig.write_html(filename)

# # Background thread for continuous data collection
# # thread = threading.Thread(target=data_collection_and_graphing, daemon=True)
# # thread.start()

# # Initialize the Dash app
# app = dash.Dash(__name__)

# # Layout for the Dash app
# app.layout = html.Div([
#     dcc.Graph(id='live-update-graph'),
#     dcc.Interval(
#         id='interval-component',
#         interval=1000,  # in milliseconds
#         n_intervals=0
#     )
# ])




# # Callback to update the graph
# @app.callback(Output('live-update-graph', 'figure'),
#               [Input('interval-component', 'n_intervals')])
# def update_graph_live(n):
#     local_df = pd.read_csv(f'{data_directory}/detections.csv')
    
#     # Determine zones and colors
#     zone_data = [determine_zone(z, radius1=50, radius2=100, radius3=150) for z in local_df['position_z']]
#     zones, colors = zip(*zone_data)
#     local_df['Zone'] = zones
#     local_df['color'] = colors
    
#     fig = px.scatter(local_df, x='timestamp', y='position_z', title='Real-Time Detections',
#                      hover_data={'position_x': True, 'position_z': True, "Zone": True},
#                      color=local_df['color'],
#                      color_discrete_map={"red": "red", "orange": "orange", "green": "green"}) # Ensuring colors are mapped correctly
#     fig.update_traces(marker=dict(size=10))
#     fig.update_layout(
#         xaxis_title="Time",
#         yaxis_title="Bounding Box Area",
#         hovermode="closest"
#     )
#     return fig

# if __name__ == '__main__':
#     app.run_server(debug=True)
