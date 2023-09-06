import numpy as np
import cv2 as cv
import random

def mouse_event_handler(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param.append((x, y))

def draw_straight_line(img, line, color, thickness=1):
    assert img.ndim >= 2
    h, w, *_ = img.shape
    a, b, c = line # Line: ax + by + c = 0
    if abs(a) > abs(b):
        pt1 = (int(c / -a), 0)
        pt2 = (int((b*h + c) / -a), h)
    else:
        pt1 = (0, int(c / -b))
        pt2 = (w, int((a*w + c) / -b))
    cv.line(img, pt1, pt2, color, thickness)

if __name__ == '__main__':
    # Load two images
    img1 = cv.imread('../data/KITTI07/image_0/000000.png', cv.IMREAD_COLOR)
    img2 = cv.imread('../data/KITTI07/image_0/000023.png', cv.IMREAD_COLOR)
    assert (img1 is not None) and (img2 is not None), 'Cannot read the given images'
    # Note) `F` is derived from `fundamental_mat_estimation.py`.
    F = np.array([[ 3.34638533e-07,  7.58547151e-06, -2.04147752e-03],
                  [-5.83765868e-06,  1.36498636e-06,  2.67566877e-04],
                  [ 1.45892349e-03, -4.37648316e-03,  1.00000000e+00]])

    # Register event handlers and show images
    wnd1_name, wnd2_name = 'Epipolar Line: Image #1', 'Epipolar Line: Image #2'
    img1_pts, img2_pts = [], []
    cv.namedWindow(wnd1_name)
    cv.namedWindow(wnd2_name)
    cv.setMouseCallback(wnd1_name, mouse_event_handler, img1_pts)
    cv.setMouseCallback(wnd2_name, mouse_event_handler, img2_pts)
    cv.imshow(wnd1_name, img1)
    cv.imshow(wnd2_name, img2)

    # Get a point from a image and draw its correponding epipolar line on the other image
    while True:
        if len(img1_pts) > 0:
            for x, y in img1_pts:
                color = (random.randrange(256), random.randrange(256), random.randrange(256))
                cv.circle(img1, (x, y), 4, color, -1)
                epipolar_line = F @ [[x], [y], [1]]
                draw_straight_line(img2, epipolar_line, color, 2)
            img1_pts.clear()
        if len(img2_pts) > 0:
            for x, y in img2_pts:
                color = (random.randrange(256), random.randrange(256), random.randrange(256))
                cv.circle(img2, (x, y), 4, color, -1)
                epipolar_line = F.T @ [[x], [y], [1]]
                draw_straight_line(img1, epipolar_line, color, 2)
            img2_pts.clear()
        cv.imshow(wnd2_name, img2)
        cv.imshow(wnd1_name, img1)
        key = cv.waitKey(10)
        if key == 27: # ESC
            break

    cv.destroyAllWindows()