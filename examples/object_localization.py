import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

def mouse_event_handler(event, x, y, flags, param):
    # Change 'mouse_state' (given as 'param') according to the mouse 'event'
    if event == cv.EVENT_LBUTTONDOWN:
        param['dragged'] = True
        param['xy_s'] = (x, y)
        param['xy_e'] = (0, 0)
    elif event == cv.EVENT_MOUSEMOVE:
        if param['dragged']:
            param['xy_e'] = (x, y)
    elif event == cv.EVENT_LBUTTONUP:
        if param['dragged']:
            param['dragged'] = False
            param['xy_e'] = (x, y)

if __name__ == '__main__':
    # The given image and its calibration data
    img_file = '../data/daejeon_station.png'
    f, cx, cy, L = 810.5, 480, 270, 3.31 # Unit: [px], [px], [px], [m]
    cam_ori = [-18.7, -8.2, 2.0]         # Unit: [deg]
    grid_x, grid_z = (-2, 3), (5, 36)    # Unit: [m]

    # Load an image
    img = cv.imread(img_file)
    assert img is not None

    # Register the mouse callback function
    mouse_state = {'dragged': False, 'xy_s': (0, 0), 'xy_e': (0, 0)}
    cv.namedWindow('Object Localization and Measurement')
    cv.setMouseCallback('Object Localization and Measurement', mouse_event_handler, mouse_state)

    # Prepare the camera projection
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    Rc = Rotation.from_euler('zyx', cam_ori[::-1], degrees=True).as_matrix()
    tc = np.array([0, -L, 0])
    R = Rc.T
    t = -Rc.T @ tc.T

    # Draw X- and Z-grids on the ground
    for x in range(*grid_x):
        s, e = [x, 0, grid_z[0]], [x, 0, grid_z[1] - 1]
        p = K @ (R @ s + t)
        q = K @ (R @ e + t)
        cv.line(img, (int(p[0] / p[2]), int(p[1] / p[2])), (int(q[0] / q[2]), int(q[1] / q[2])), (64, 128, 64), 1)
    for z in range(*grid_z):
        s, e = [grid_x[0], 0, z], [grid_x[1] - 1, 0, z]
        p = K @ (R @ s + t)
        q = K @ (R @ e + t)
        cv.line(img, (int(p[0] / p[2]), int(p[1] / p[2])), (int(q[0] / q[2]), int(q[1] / q[2])), (64, 128, 64), 1)

    while True:
        img_copy = img.copy()
        if mouse_state['xy_e'][0] > 0 and mouse_state['xy_e'][1] > 0:
            # Calculate object location and height
            c = R.T @ [mouse_state['xy_s'][0] - cx, mouse_state['xy_s'][1] - cy, f]
            h = R.T @ [mouse_state['xy_e'][0] - cx, mouse_state['xy_e'][1] - cy, f]
            if c[1] < 1e-6: # Skip the degenerate case (beyond the horizon)
                continue
            X = c[0] / c[1] * L                 # Object location X [m]
            Z = c[2] / c[1] * L                 # Object location Y [m]
            H = (c[1] / c[2] - h[1] / h[2]) * Z # Object height [m]

            # Draw the head/contact points and location/height
            cv.line(img_copy, mouse_state['xy_s'], mouse_state['xy_e'], (0, 0, 255), 2)
            cv.circle(img_copy, mouse_state['xy_e'], 4, (255, 0, 0), -1) # Head point
            cv.circle(img_copy, mouse_state['xy_s'], 4, (0, 255, 0), -1) # Contact point
            info = f'X: {X:.3f}, Z: {Z:.3f}, H: {H:.3f}'
            cv.putText(img_copy, info, np.array(mouse_state['xy_s']) + (-20, 20), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

        cv.imshow('Object Localization and Measurement', img_copy)
        key = cv.waitKey(10)
        if key == 27: # ESC
            break

    cv.destroyAllWindows()
