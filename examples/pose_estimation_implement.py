import cv2 as cv
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def project_no_distort(X, rvec, t, K):
    R = Rotation.from_rotvec(rvec.flatten()).as_matrix()
    XT = X @ R.T + t                     # Transpose of 'X = R @ X + t'
    xT = XT @ K.T                        # Transpose of 'x = KX'
    xT = xT / xT[:,-1].reshape((-1, 1))  # Normalize
    return xT[:,0:2]

def reproject_error_pnp(unknown, X, x, K):
    rvec, tvec = unknown[:3], unknown[3:]
    xp = project_no_distort(X, rvec, tvec, K)
    err = x - xp
    return err.ravel()

def solvePnP(obj_pts, img_pts, K):
    unknown_init = np.array([0, 0, 0, 0, 0, 1.]) # Sequence: rvec(3), tvec(3)
    result = least_squares(reproject_error_pnp, unknown_init, args=(obj_pts, img_pts, K))
    return result['success'], result['x'][:3], result['x'][3:]

if __name__ == '__main__':
    f, cx, cy = 1000., 320., 240.
    obj_pts = np.loadtxt('../bin/data/box.xyz')
    img_pts = np.loadtxt('../bin/data/image_formation1.xyz')[:,:2].copy()
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    dist_coeff = np.zeros(4)

    # Estimate camera pose
    _, rvec, tvec = solvePnP(obj_pts, img_pts, K) # Note) Ignore lens distortion
    R = Rotation.from_rotvec(rvec.flatten()).as_matrix()
    my_ori = Rotation.from_matrix(R.T).as_euler('xyz')
    my_pos = -R.T @ tvec

    # Estimate camera pose using OpenCV
    _, rvec, tvec = cv.solvePnP(obj_pts, img_pts, K, dist_coeff)
    R = Rotation.from_rotvec(rvec.flatten()).as_matrix()
    cv_ori = Rotation.from_matrix(R.T).as_euler('xyz')
    cv_pos = -R.T @ tvec.flatten()

    print('\n### Ground Truth')
    print('* Camera orientation: [-15, 15, 0] [deg]')
    print('* Camera position   : [-2, -2, 0] [m]')
    print('\n### My Camera Pose')
    print(f'* Camera orientation: {np.rad2deg(my_ori)} [deg]')
    print(f'* Camera position   : {my_pos} [m]')
    print('\n### OpenCV Camera Pose')
    print(f'* Camera orientation: {np.rad2deg(cv_ori)} [deg]')
    print(f'* Camera position   : {cv_pos} [m]')