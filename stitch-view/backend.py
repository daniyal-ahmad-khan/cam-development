from flask import Flask, render_template, Response, jsonify
import cv2
import pyudev
app = Flask(__name__)


 





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

@app.route('/list-cameras')
def list_cameras():
    cameras = list_and_verify_video_devices()
    return jsonify(cameras)


@app.route('/video_feed/<string:id>/', methods=["GET"])
def video_feed(id):
   
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)