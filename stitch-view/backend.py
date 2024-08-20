from flask import Flask, render_template, Response, jsonify
import cv2
import pyudev
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

app = Flask(__name__)

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint", filter_threshold=.99).eval().to(device)

def gen_frames(camera_id):
     
    cam = int(camera_id)
    cap=  cv2.VideoCapture(cam)
    
    while True:
        # for cap in caps:
        # # Capture frame-by-frame
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def list_and_verify_video_devices():
    context = pyudev.Context()
    accessible_cameras = []
    for device in context.list_devices(subsystem='video4linux'):
        if 'video' in device.device_node:
            if test_camera(device.device_node):
                accessible_cameras.append(device.device_node)
    return accessible_cameras

def test_camera(device_node):
    cap = cv2.VideoCapture(device_node)
    if not cap.isOpened():
        return False
    ret, frame = cap.read()
    cap.release()
    return ret

def gen_stitched_frames(left_idx, right_idx):
    cap1 = cv2.VideoCapture(left_idx)
    cap2 = cv2.VideoCapture(right_idx)
    initialized = False
    homography_matrix = None

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        if not initialized:
            initialized, homography_matrix = initialize_stitching(frame1, frame2)
            if not initialized:
                continue

        stitched_frame = cv2.warpPerspective(frame2, homography_matrix, (frame1.shape[1] + frame2.shape[1], frame1.shape[0]))
        stitched_frame[0:frame1.shape[0], 0:frame1.shape[1]] = frame1
        
        ret, buffer = cv2.imencode('.jpg', stitched_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def initialize_stitching(frame1, frame2):
    cv2.imwrite("frame1.jpg", frame1)
    cv2.imwrite("frame2.jpg", frame2)
    image0 = load_image("frame1.jpg").to(device)
    image1 = load_image("frame2.jpg").to(device)
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    kpts0 = feats0['keypoints'][matches01['matches'][..., 0]]
    kpts1 = feats1['keypoints'][matches01['matches'][..., 1]]
    homography_matrix, status = cv2.findHomography(kpts1.cpu().numpy(), kpts0.cpu().numpy(), cv2.RANSAC)
    return status.sum() > 0, homography_matrix




@app.route('/list-cameras')
def list_cameras():
    cameras = list_and_verify_video_devices()
    return jsonify(cameras)


@app.route('/video_feed/<string:id>/', methods=["GET"])
def video_feed(id):
   
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stitched_feed/<int:left_idx>/<int:right_idx>/')
def stitched_feed(left_idx, right_idx):
    """Video streaming route for stitched view."""
    return Response(gen_stitched_frames(left_idx, right_idx),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)