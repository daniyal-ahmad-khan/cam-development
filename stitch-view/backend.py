from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import pyudev
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

app = Flask(__name__)
CORS(app)

# Set up camera captures globally
camera_captures = {}

# Initialize device for AI processing
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint", filter_threshold=.99).eval().to(device)

def test_camera(index):
    """ Test the camera by index to check if it can be opened and a frame can be read. """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Camera {index} could not be opened.")
        return False
    ret, frame = cap.read()
    cap.release()
    if ret:
        print(f"Camera {index} is functional.")
    else:
        print(f"Camera {index} could not capture a frame.")
    return ret

def initialize_cameras():
    """ Initializes all cameras and stores their capture objects in a global dictionary. """
    context = pyudev.Context()
    for device in context.list_devices(subsystem='video4linux'):
        if 'video' in device.device_node:
            index = int(device.device_node.rsplit('/', 1)[-1].replace('video', ''))
            if test_camera(index):
                camera_captures[index] = cv2.VideoCapture(index)
                print(f"Camera {index} initialized and stored.")

def gen_frames(camera_id):
    """ Generates frames from a given camera ID. """
    cap = camera_captures.get(camera_id)
    if not cap:
        yield b'Camera not initialized or unavailable\r\n'
        return
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Error generating frames: {e}")

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

def gen_stitched_frames(left_idx, right_idx):
    cap1 = camera_captures.get(left_idx)
    cap2 = camera_captures.get(right_idx)
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




@app.route('/list-cameras')
def list_cameras():
    """ Endpoint to list available camera indexes. """
    cameras = list(camera_captures.keys())
    return jsonify(cameras)

@app.route('/video_feed/<int:id>/', methods=["GET"])
def video_feed(id):
    """ Video streaming route. """
    return Response(gen_frames(id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stitched_feed/<int:left_idx>/<int:right_idx>/')
def stitched_feed(left_idx, right_idx):
    """ Video streaming route for stitched view. """
    return Response(gen_stitched_frames(left_idx, right_idx),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/reinitialize-cameras', methods=['POST'])
def reinitialize_cameras():
    """ Endpoint to reinitialize all cameras. This can be used to recognize newly connected cameras without restarting the server. """
    release_cameras()
    initialize_cameras()
    return jsonify({'message': 'Cameras reinitialized'}), 200

@app.route('/release-cameras', methods=['POST'])
def release_cameras():
    """Endpoint to release all camera resources."""
    for index, cap in camera_captures.items():
        if cap.isOpened():
            cap.release()
            print(f"Released camera {index}")
    camera_captures.clear()
    return jsonify({'message': 'All cameras released'}), 200

@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')



if __name__ == '__main__':
    initialize_cameras()
    app.run(debug=False, port=3001)
