import numpy as np
import cv2 as cv
from homography_estimation_implement import getPerspectiveTransform

def warpPerspective1(src, H, dst_size):
    # Generate an empty image
    width, height = dst_size
    channel = src.shape[2] if src.ndim > 2 else 1
    dst = np.zeros((height, width, channel), dtype=src.dtype)

    # Copy a pixel from `src` to `dst` (forword mapping)
    for py in range(img.shape[0]):
        for px in range(img.shape[1]):
            q = H @ [px, py, 1]
            qx, qy = int(q[0]/q[-1] + 0.5), int(q[1]/q[-1] + 0.5)
            if qx >= 0 and qy >= 0 and qx < width and qy < height:
                dst[qy, qx] = src[py, px]
    return dst

def warpPerspective2(src, H, dst_size):
    # Generate an empty image
    width, height = dst_size
    channel = src.shape[2] if src.ndim > 2 else 1
    dst = np.zeros((height, width, channel), dtype=src.dtype)

    # Copy a pixel from `src` to `dst` (backward mapping)
    H_inv = np.linalg.inv(H)
    for qy in range(height):
        for qx in range(width):
            p = H_inv @ [qx, qy, 1]
            px, py = int(p[0]/p[-1] + 0.5), int(p[1]/p[-1] + 0.5)
            if px >= 0 and py >= 0 and px < src.shape[1] and py < src.shape[0]:
                dst[qy, qx] = src[py, px]
    return dst

if __name__ == '__main__':
    img = cv.imread('../data/sunglok_card.jpg')
    wnd_name = 'Image Warping'
    card_size = (900, 480)
    pts_src = np.array([[95, 243], [743, 121], [157, 652], [969, 372]], dtype=np.float32)
    pts_dst = np.array([[0, 0], [card_size[0], 0], [0, card_size[1]], card_size], dtype=np.float32)

    # Find planar homography and transform the original image
    H = getPerspectiveTransform(pts_src, pts_dst)
    warp1 = warpPerspective1(img, H, card_size)
    warp2 = warpPerspective2(img, H, card_size)

    # Show images generated from two methods
    cv.imshow(wnd_name + ' (Method 1)', warp1)
    cv.imshow(wnd_name + ' (Method 2)', warp2)
    cv.waitKey(0)
    cv.destroyAllWindows()