import numpy as np
from scipy.optimize import least_squares
from pose_estimation_implement import project_no_distort

def fcxcy_to_K(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

def reproject_error_calib(unknown, Xs, xs):
    K = fcxcy_to_K(*unknown[0:3])
    err = []
    for i in range(len(xs)):
        offset = 3 + 6 * i
        rvec, tvec = unknown[offset:offset+3], unknown[offset+3:offset+6]
        xp = project_no_distort(Xs[i], rvec, tvec, K)
        err.append(xs[i] - xp)
    return np.vstack(err).ravel()

def calibrateCamera(obj_pts, img_pts, img_size):
    img_n = len(img_pts)
    unknown_init = np.array([img_size[0], img_size[0]/2, img_size[1]/2] + img_n * [0, 0, 0, 0, 0, 1.]) # Sequence: f, cx, cy, img_n * (rvec, tvec)
    result = least_squares(reproject_error_calib, unknown_init, args=(obj_pts, img_pts))
    K = fcxcy_to_K(*result['x'][0:3])
    rvecs = [result['x'][(6*i+3):(6*i+6)] for i in range(img_n)]
    tvecs = [result['x'][(6*i+6):(6*i+9)] for i in range(img_n)]
    return result['cost'], K, np.zeros(5), rvecs, tvecs

if __name__ == '__main__':
    img_size = (640, 480)
    img_files = ['../bin/data/image_formation1.xyz', '../bin/data/image_formation2.xyz']
    img_pts = []
    for file in img_files:
        pts = np.loadtxt('../bin/data/image_formation1.xyz', dtype=np.float32)
        img_pts.append(pts[:,:2])

    pts = np.loadtxt('../bin/data/box.xyz', dtype=np.float32)
    obj_pts = [pts] * len(img_pts) # Copy the object point as much as the number of image observation

    # Calibrate the camera
    _, K, *_ = calibrateCamera(obj_pts, img_pts, img_size)

    print('\n### Ground Truth')
    print('* f, cx, cy = 1000, 320, 240')
    print('\n### My Calibration')
    print(f'* f, cx, cy = {K[0,0]:.1f}, {K[0,2]:.1f}, {K[1,2]:.1f}')