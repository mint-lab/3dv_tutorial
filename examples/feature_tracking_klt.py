import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

video_file = '../data/KITTI07/image_0/%06d.png'
min_track_error = 5

# Open a video and get an initial image
video = cv.VideoCapture(video_file)
assert video.isOpened()

_, gray_prev = video.read()
assert gray_prev.size > 0
if gray_prev.ndim >= 3 and gray_prev.shape[2] > 1:
    gray_prev = cv.cvtColor(gray_prev, cv.COLOR_BGR2GRAY)

# Run the KLT feature tracker
while True:
    # Grab an image from the video
    valid, img = video.read()
    if not valid:
        break
    if img.ndim >= 3 and img.shape[2] > 1:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Extract optical flow
    pts_prev = cv.goodFeaturesToTrack(gray_prev, 2000, 0.01, 10)
    pts, status, error = cv.calcOpticalFlowPyrLK(gray_prev, gray, pts_prev, None)
    gray_prev = gray

    # Show the optical flow on the image
    if img.ndim < 3 or img.shape[2] < 3:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for pt, pt_prev, tracked, err in zip(pts, pts_prev, status, error):
        if tracked and err < min_track_error:
            cv.line(img, pt_prev.flatten().astype(np.int32), pt.flatten().astype(np.int32), (0, 255, 0))
    cv.imshow('KLT Feature Tracking', img)
    key = cv.waitKey(1)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()