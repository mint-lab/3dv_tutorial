import numpy as np
import cv2 as cv

# Open a video and get the reference image and feature points
video = cv.VideoCapture('../data/traffic.avi')
assert video.isOpened(), 'Cannot read the given video'

_, gray_ref = video.read()
if gray_ref.ndim >= 3:
    gray_ref = cv.cvtColor(gray_ref, cv.COLOR_BGR2GRAY)
pts_ref = cv.goodFeaturesToTrack(gray_ref, 2000, 0.01, 10)

# Run and show video stabilization
while True:
    # Read an image from 'video'
    valid, img = video.read()
    if not valid:
        break
    if img.ndim >= 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Extract optical flow and calculate planar homography
    pts, status, err = cv.calcOpticalFlowPyrLK(gray_ref, gray, pts_ref, None)
    H, inlier_mask = cv.findHomography(pts, pts_ref, cv.RANSAC)

    # Synthesize a stabilized image
    warp = cv.warpPerspective(img, H, (img.shape[1], img.shape[0]))

    # Show the original and stabilized images together
    for pt, pt_ref, inlier in zip(pts, pts_ref, inlier_mask):
        color = (0, 127, 0)
        if inlier:
            color = (0, 0, 255)
        cv.line(img, pt.flatten().astype(np.int32), pt_ref.flatten().astype(np.int32), color)
    cv.imshow('Video Stabilization', np.hstack((img, warp)))
    key = cv.waitKey(1)
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()
