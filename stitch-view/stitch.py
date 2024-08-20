import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

# Setup for LightGlue
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the extractor and matcher
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint", filter_threshold=.95).eval().to(device)

def crop_to_match(frame_large, frame_small):
    # Dimensions of the smaller frame
    h_small, w_small = frame_small.shape[:2]
    # Dimensions of the larger frame
    h_large, w_large = frame_large.shape[:2]

    # Calculate the new height to maintain the aspect ratio of the smaller frame
    new_height = int((h_small / w_small) * w_large)
    # Ensure the new height does not exceed the original height of the larger frame
    new_height = min(new_height, h_large)

    # Calculate starting points for the crop to center it
    start_y = (h_large - new_height) // 2
    start_x = (w_large - w_small) // 2 if w_large > w_small else 0

    # Crop to the new dimensions, centering the crop
    return frame_large[start_y:start_y + new_height, start_x:start_x + w_small]

def stitch_videos(video_path1, video_path2):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Cannot open one or both video sources.")
        return

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Cannot read from one or both video sources.")
        return

    # Crop the larger frame to match the smaller one
    if frame1.shape[0] != frame2.shape[0] or frame1.shape[1] != frame2.shape[1]:
        if frame1.size < frame2.size:
            frame2 = crop_to_match(frame2, frame1)
        else:
            frame1 = crop_to_match(frame1, frame2)

    cv2.imwrite("frame1.jpg", frame1)
    cv2.imwrite("frame2.jpg", frame2)
    image0 = load_image("frame1.jpg")
    image1 = load_image("frame2.jpg")

    # Extract keypoints and matches
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # Compute homography
    homography_matrix, _ = cv2.findHomography(m_kpts1.cpu().numpy(), m_kpts0.cpu().numpy(), cv2.RANSAC)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    stitched_video = cv2.VideoWriter('stitched_video.mp4', fourcc, 20.0, (frame1.shape[1] + frame2.shape[1], frame1.shape[0]))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Crop on the fly if necessary
        if frame1.shape[0] != frame2.shape[0] or frame1.shape[1] != frame2.shape[1]:
            if frame1.size < frame2.size:
                frame2 = crop_to_match(frame2, frame1)
            else:
                frame1 = crop_to_match(frame1, frame2)

        stitched_frame = cv2.warpPerspective(frame2, homography_matrix, (frame1.shape[1] + frame2.shape[1], frame1.shape[0]))
        stitched_frame[0:frame1.shape[0], 0:frame1.shape[1]] = frame1

        # Write the frame into the file
        stitched_video.write(stitched_frame)
        cv2.imshow("Stitched Video", stitched_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    stitched_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path1 = "stitched_video-02.mp4"
    video_path2 = "stitched_video-03.mp4"
    stitch_videos(video_path1, video_path2)
