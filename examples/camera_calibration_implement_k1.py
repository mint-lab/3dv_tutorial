import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from camera_calibration_implement import fcxcy_to_K

def project_distort_k1(X, rvec, t, K, k1):
    R = Rotation.from_rotvec(rvec.flatten()).as_matrix()
    XT = X @ R.T + t                     # Transpose of 'X = R @ X + t'
    nT = XT / XT[:,-1].reshape((-1, 1))  # Normalize
    r2 = nT[:,0]**2 + nT[:,1]**2
    nT[:,0] = nT[:,0] * (1 + k1 * r2)
    nT[:,1] = nT[:,1] * (1 + k1 * r2)
    xT = nT @ K.T                        # Transpose of 'x = Kn'
    return xT[:,0:2]

def reproject_error_calib(unknown, Xs, xs):
    K = fcxcy_to_K(*unknown[0:3])
    k1 = unknown[3]
    err = []
    for j in range(len(xs)):
        offset = 4 + 6 * j
        rvec, tvec = unknown[offset:offset+3], unknown[offset+3:offset+6]
        xp = project_distort_k1(Xs[j], rvec, tvec, K, k1)
        err.append(xs[j] - xp)
    return np.vstack(err).ravel()

def calibrateCamera(obj_pts, img_pts, img_size):
    img_n = len(img_pts)
    unknown_init = np.array([img_size[0], img_size[0]/2, img_size[1]/2, 0.] + img_n * [0, 0, 0, 0, 0, 1.]) # Sequence: f, cx, cy, k1, img_n * (rvec, tvec)
    result = least_squares(reproject_error_calib, unknown_init, args=(obj_pts, img_pts))
    K = fcxcy_to_K(*result['x'][0:3])
    dist_coeff = np.array([result['x'][3], 0, 0, 0, 0], dtype=np.float32)
    rvecs = [result['x'][(6*i+4):(6*i+7)] for i in range(img_n)]
    tvecs = [result['x'][(6*i+7):(6*i+10)] for i in range(img_n)]
    return result['cost'], K, dist_coeff, rvecs, tvecs

if __name__ == '__main__':
    img_size = (640, 480)
    img_files = ['../data/image_formation1.xyz', '../data/image_formation2.xyz']
    img_pts = []
    for file in img_files:
        pts = np.loadtxt(file, dtype=np.float32)
        img_pts.append(pts[:,:2])

    pts = np.loadtxt('../data/box.xyz', dtype=np.float32)
    obj_pts = [pts] * len(img_pts) # Copy the object point as much as the number of image observation

    # Calibrate the camera
    _, K, dist_coeff, *_ = calibrateCamera(obj_pts, img_pts, img_size)

    print('\n### Ground Truth')
    print('* f, cx, cy, k1 = 1000, 320, 240, 0')
    print('\n### My Calibration')
    print(f'* f, cx, cy, k1 = {K[0,0]:.1f}, {K[0,2]:.1f}, {K[1,2]:.1f}, {dist_coeff[0]:.4f}')