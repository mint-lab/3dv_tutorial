import cv2
import numpy as np
import copy
import yaml

input = "data/chessboard.avi"

board_pattern = (10, 7)
select_images = True

capture = cv2.VideoCapture(input)

if capture.isOpened() == False: raise Exception("No video")

images = []

while True:
    ret1, image = capture.read()
    if not ret1: break
    if select_images:
        cv2.imshow("3DV Tutorial: Camera Calibration", image)
        key = cv2.waitKey(1)
        if key == 27: break # 'ESC' key: Exit
        elif key == 32:     # 'Space' key: Pause
            ret2, pts = cv2.findChessboardCorners(image, board_pattern, None) # No flags
            
            # display = image.clone()
            display = copy.deepcopy(image)
            display = cv2.drawChessboardCorners(display, board_pattern, pts, ret2)
            cv2.imshow("3DV Tutorial: Camera Calibration", display)
            key = cv2.waitKey()
            if key == 27: break
            elif key == 13: images.append(image) # 'Enter' key: Save
    else: 
        images.append(image)

capture.release()

if(len(images)) == 0:
    print("no images")
    raise Exception("There is no captured images!")

# Find 2D corner points from given images
img_points = []
h, w = 0,0
for image in images:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ret3, corners = cv2.findChessboardCorners(gray, board_pattern) # No flags
    if ret3 == True:
        img_points.append(corners)

if len(img_points) == 0:
    raise Exception("No 2d Corner pts")

# Prepare 3D points of the chess board
objp = np.zeros((10*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:10].T.reshape(-1, 2)
obj_points = []
for _ in images:
    obj_points.append(objp)

# Calibrate Camera
K = np.eye(3,3, dtype=np.float32)
dist_coeff = np.zeros((4,1))
rms, K, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (h,w), None, None)

# Report calibration results
print("## Camera Calibration Results")
print(f"* The number of applied images = {w}x{h}")
print(f"* RMS error = {rms}")
print(f"* Camera matrix (K) = \n{K}")
print(f"* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff}")

# Save as cam_config.yaml
file_name = "cfg/cam_config.yaml"
cam_dict = {
    "Intrinsic": K.flatten().tolist(),
    "Distortion": dist_coeff.flatten().tolist(),
    "RMS": rms,
}
with open(file_name, 'w') as f:
    yaml.dump(cam_dict, f, sort_keys=False, default_flow_style=False)

print("End!")