import numpy as np
import cv2 as cv

video_file, cover_file = '../data/blais.mp4', '../data/blais.jpg'
min_inlier_num = 100

fdetector = cv.ORB_create()
fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')

# Load the object image and extract features
obj_image = cv.imread(cover_file)
assert obj_image is not None
obj_keypoints, obj_descriptors = fdetector.detectAndCompute(obj_image, None)
assert len(obj_keypoints) >= min_inlier_num
fmatcher.add(obj_descriptors)

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given video, ' + video_file

# Prepare a box for simple AR
box_lower = np.array([[30, 145, 0], [30, 200, 0], [200, 200, 0], [200, 145, 0]], dtype=np.float32)
box_upper = np.array([[30, 145, -50], [30, 200, -50], [200, 200, -50], [200, 145, -50]], dtype=np.float32)

# Run pose extimation
calib_param = cv.CALIB_FIX_ASPECT_RATIO | cv.CALIB_FIX_PRINCIPAL_POINT | cv.CALIB_ZERO_TANGENT_DIST | cv.CALIB_FIX_K3 | cv.CALIB_FIX_K4 | cv.CALIB_FIX_K5 | cv.CALIB_FIX_S1_S2_S3_S4 | cv.CALIB_FIX_TAUX_TAUY
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Extract features and match them to the object features
    img_keypoints, img_descriptors = fdetector.detectAndCompute(img, None)
    match = fmatcher.match(img_descriptors, obj_descriptors)
    if len(match) < min_inlier_num:
        continue

    obj_pts, img_pts = [], []
    for m in match:
        obj_pts.append(obj_keypoints[m.trainIdx].pt)
        img_pts.append(img_keypoints[m.queryIdx].pt)
    obj_pts = np.array(obj_pts, dtype=np.float32)
    obj_pts = np.hstack((obj_pts, np.zeros((len(obj_pts), 1), dtype=np.float32))) # Make 2D to 3D
    img_pts = np.array(img_pts, dtype=np.float32)

    # Deterimine whether each matched feature is an inlier or not
    H, inlier_mask = cv.findHomography(obj_pts, img_pts, cv.RANSAC, 2)
    inlier_mask = inlier_mask.flatten()
    img_result = cv.drawMatches(img, img_keypoints, obj_image, obj_keypoints, match, None, (0, 0, 255), (0, 127, 0), inlier_mask)

    # Check whether inliers are enough or not
    inlier_num = sum(inlier_mask)
    if inlier_num > min_inlier_num:
        # Calibrate the camera and estimate its pose with inliers
        ret, K, dist_coeff, rvecs, tvecs = cv.calibrateCamera([obj_pts[inlier_mask.astype(bool)]], [img_pts[inlier_mask.astype(bool)]], (img.shape[0], img.shape[1]), None, None, None, None, calib_param)
        rvec, tvec = rvecs[0], tvecs[0]

        # Draw the box on the image
        line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
        cv.polylines(img_result, [np.int32(line_lower)], True, (255, 0, 0), 2)
        cv.polylines(img_result, [np.int32(line_upper)], True, (0, 0, 255), 2)
        for b, t in zip(line_lower, line_upper):
            cv.line(img_result, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)
        info = f'Inliers: {inlier_num} ({inlier_num*100/len(match):.0f}), Focal length: {K[0,0]:.0f}'
        cv.putText(img_result, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Book)', img_result)
    key = cv.waitKey(1)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()
