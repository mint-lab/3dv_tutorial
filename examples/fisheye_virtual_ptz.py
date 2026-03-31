import numpy as np
from scipy.spatial.transform import Rotation
import cv2 as cv

# The given fisheye image and camera model assumptions
fisheye_file = '../data/fisheye-blog-pics_09.jpg'
fisheye_fov_deg = 180.0

# Open a fisheye image
fisheye_img = cv.imread(fisheye_file)
assert fisheye_img is not None, 'Cannot read the given input, ' + fisheye_file
cv.imshow('Geometric Distortion Correction with Virtual PTZ: Fisheye', fisheye_img)

# Derive the intrinsic parameters for the fisheye camera
img_h, img_w = fisheye_img.shape[:2]
img_cx = (img_w - 1) * 0.5
img_cy = (img_h - 1) * 0.5
fisheye_f = (img_w / 2) / np.deg2rad(fisheye_fov_deg / 2) # Equidistant model: f = r / theta
fisheye_K = np.array([[fisheye_f, 0, img_cx], [0, fisheye_f, img_cy], [0, 0, 1]], dtype=np.float64)
fisheye_D = np.zeros((4, 1), dtype=np.float64)

# Define the undistorted perspective images
undist_fov_deg = 100.0
fov_step, fov_min, fov_max = 2.0, 30.0, 160.0
undist_pan_deg, undist_tilt_deg, undist_roll_deg = 0.0, 0.0, 0.0
rotation_step = 2.0

need_to_update = True
map1, map2 = None, None
while True:
	if need_to_update:
        # Generate the undistortion maps for the current parameters (Alternative: `cv.fisheye.undistortImage()`)
		undist_f = (img_w / 2) / np.tan(np.deg2rad(undist_fov_deg / 2)) # Perspective model: f = r / tan(theta)
		undist_K = np.array([[undist_f, 0, img_cx], [0, undist_f, img_cy], [0, 0, 1]], dtype=np.float64)
		undist_R = Rotation.from_euler('zxy', [undist_roll_deg, undist_tilt_deg, undist_pan_deg], degrees=True).as_matrix()
		map1, map2 = cv.fisheye.initUndistortRectifyMap(fisheye_K, fisheye_D, undist_R, undist_K, (img_w, img_h), cv.CV_32FC1)
		need_to_update = False

    # Generate the undistorted image and show the image
	assert map1 is not None and map2 is not None
	undist_img = cv.remap(fisheye_img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
	info = f'pan:{undist_pan_deg:.0f}, tilt:{undist_tilt_deg:.0f}, roll:{undist_roll_deg:.0f}, fov:{undist_fov_deg:.0f} [deg]'
	cv.putText(undist_img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
	cv.imshow('Geometric Distortion Correction with Virtual PTZ: Undistorted', undist_img)

    # Process the key event
	key = cv.waitKey()
	if key == 27: # ESC
		break
	if key == ord('a') or key == ord('A'):
		undist_pan_deg -= rotation_step
		need_to_update = True
	elif key == ord('d') or key == ord('D'):
		undist_pan_deg += rotation_step
		need_to_update = True
	elif key == ord('w') or key == ord('W'):
		undist_tilt_deg -= rotation_step
		need_to_update = True
	elif key == ord('s') or key == ord('S'):
		undist_tilt_deg += rotation_step
		need_to_update = True
	elif key == ord('q') or key == ord('Q'):
		undist_roll_deg -= rotation_step
		need_to_update = True
	elif key == ord('e') or key == ord('E'):
		undist_roll_deg += rotation_step
		need_to_update = True
	elif key == ord('+') or key == ord('='):
		undist_fov_deg = max(fov_min, undist_fov_deg - fov_step)
		need_to_update = True
	elif key == ord('-') or key == ord('_'):
		undist_fov_deg = min(fov_max, undist_fov_deg + fov_step)
		need_to_update = True
	elif key == ord('\t'):
		undist_pan_deg = 0.0
		undist_tilt_deg = 0.0
		undist_roll_deg = 0.0
		undist_fov_deg = 100.0
		need_to_update = True

cv.destroyAllWindows()