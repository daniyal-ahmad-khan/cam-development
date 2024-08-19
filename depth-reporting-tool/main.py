import numpy as np
import cv2
import depthai as dai
from pipeline_setup import setup_pipeline
from utils import TextHelper, FPSHandler, displayFrame
from config import bbox_color, text_color, NN_SIZE, nnBlobPath
import os
import pandas as pd
from datetime import datetime
from config import METER_TO_PIXEL
import math
import plotly.express as px



def determine_zone(z_point, radius1, radius2, radius3):
    # x_point_pixel = math.trunc(x_point*m_to_pix)
    z_point_pixel = math.trunc(z_point * METER_TO_PIXEL)
    if z_point_pixel < radius1:
        return "Danger Zone", "red"
    elif z_point_pixel < radius2:
        return "Warning Zone", "orange"
    else:
        return "Safe Zone", "green"
def save_static_graph(filename,data_directory):

    local_df = pd.read_csv(f'{data_directory}/detections.csv')
    zone_data = [determine_zone(z, radius1=50, radius2=100, radius3=150) for z in local_df['position_z']]
    zones, colors = zip(*zone_data)
    local_df['Zone'] = zones
    local_df['color'] = colors
    fig = px.scatter(local_df, x='timestamp', y='position_z', title='Real-Time Detections',
                     hover_data={'label':True,'position_x': True, 'position_z': True, "Zone": True},
                     color=local_df['color'],
                     color_discrete_map={"red": "red", "orange": "orange", "green": "green"})
    fig.write_html(filename)

def process_frames(device, radius1, radius2, radius3, df, data_directory):
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    qPass = device.getOutputQueue(name="pass", maxSize=4, blocking=False)
    text = TextHelper(color=bbox_color, text_color=text_color)
    fps = FPSHandler()
    
    last_saved_time = datetime.now()
    while True:

        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()
        depth = depthQueue.get()
        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame()  # depthFrame values are in millimeters

        depth_downscaled = depthFrame[::4]
        if np.all(depth_downscaled == 0):
            min_depth = 0  # Set a default minimum depth value when all elements are zero
        else:
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
        max_depth = np.percentile(depth_downscaled, 99)
        depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        if inDet is not None:
            detections = inDet.detections
            fps.next_iter()
        
        animmated_frame  = np.zeros(frame.shape, dtype=np.uint8)
        # radius = calculate_radius(5)

        # draw_circle(animmated_frame, radius1, radius2, radius3)
        text.putText(frame, "NN fps: {:.2f}".format(fps.fps()), (2, frame.shape[0] - 4))

        # displayFrame("preview", detections, frame, animmated_frame, text, radius1, radius2, radius3,df)
        if (datetime.now() - last_saved_time).total_seconds() >= 2:
            displayFrame("preview", detections, frame, animmated_frame, text, radius1, radius2, radius3,df)
            df.to_csv(f'{data_directory}/detections.csv', index=False)
            save_static_graph(f'{data_directory}/detections.html', data_directory)
            last_saved_time = datetime.now()
        # cv2.imshow("Animated", animmated_frame)
        
        # Write the frame to the video file
        # video_writer.write(animmated_frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the VideoWriter
    # video_writer.release()

def main():
    data_directory = 'detection_data'
    os.makedirs(data_directory, exist_ok=True)

    data = {
    "timestamp": [],
    "Zone": [],
    "position_x": [],
    "position_z": [],
    "label": []
}
    df = pd.DataFrame(data)
    df.to_csv(f'{data_directory}/detections.csv', index=False)
    pipeline = setup_pipeline(NN_SIZE, nnBlobPath)
    radius1 = 50
    radius2 = 100
    radius3 = 150
    with dai.Device(pipeline) as device:

        process_frames(device, radius1, radius2, radius3, df, data_directory)

if __name__ == '__main__':
    main()
