from re import S
import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

def main():
    input = "data/chessboard.avi"
    cam_params_file = "cam_config.yaml"
    K = np.zeros((3,3))
    dist_coeff = np.zeros((4,1))
    with open(cam_params_file, 'r') as f:
        cam_params = yaml.load(f, Loader=SafeLoader)
        K = np.resize(cam_params["Intrinsic"], (3,3))
        dist_coeff = np.resize(cam_params["Distortion"], (4,1))

    # Open Video
    capture = cv2.VideoCapture(input)
    if capture.isOpened() == False: raise Exception("No video")

    show_rectify = True
    map1 = np.array([])
    map2 = np.array([])

    while True:
        ret, image = capture.read()
        if not ret: break
        
        info = "Original"
        if(show_rectify):
            if not (len(map1) & len(map2)):
                map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeff, None, None, (image.shape[1], image.shape[0]), cv2.CV_32FC1)
            image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
            info = "Rectified"
        cv2.putText(image, info, (5,15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255,0))

        cv2.imshow("3DV Tutorial: Distortion Correction", image)

        key = cv2.waitKey(1)
        if key == 27: break
        elif key == 9: show_rectify = not show_rectify
        elif key == 32:
            key = cv2.waitKey()
            if key == 27: break
            elif key == 9: show_rectify = not show_rectify

    capture.release()

if __name__ == "__main__":
    main()