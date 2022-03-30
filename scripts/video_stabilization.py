from tkinter import image_names
import cv2
import numpy as np
import copy

def main():
    input = "../bin/data/traffic.avi"
    cap = cv2.VideoCapture(input)
    if not cap.isOpened(): raise Exception("No video!")
    
    ret, gray_ref = cap.read()
    print(gray_ref.shape)

    if not ret: raise Exception("Can't get image from Video")
    if gray_ref.shape[2] >1: gray_ref = cv2.cvtColor(gray_ref, cv2.COLOR_BGR2GRAY)
    point_ref = cv2.goodFeaturesToTrack(gray_ref, 2000, 0.01, 10)
    if len(point_ref) < 4:
        cap.release()
        raise Exception("Not good video init image")
    
    while True:
        ret, image = cap.read()
        if not ret: break
        if image.shape[2] > 1: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else: gray = copy.deepcopy(image)

        point, status, err = cv2.calcOpticalFlowPyrLK(gray_ref, gray, point_ref, None)
        H, inliner_mask = cv2.findHomography(point, point_ref, cv2.RANSAC)

        warp = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))

        for i in range(len(point_ref)):
            px_ref, py_ref = point_ref[i][0][0], point_ref[i][0][1]
            px, py = point[i][0][0], point[i][0][1]
            if inliner_mask[i] > 0: image = cv2.line(image, (px_ref, py_ref), (px, py), (0,0,255))
            else: image = cv2.line(image, (px_ref, py_ref), (px, py), (0,127,0))
        
        info = f"image shape: {image.shape[0]} {image.shape[1]} {image.shape[2]}"
        image = cv2.putText(image, info, (5,15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))

        cv2.imshow("3DV Tutorial: Video Stabilization", np.hstack((image,warp)))
        if cv2.waitKey(1) == ord('q'): break

    cap.release()

if __name__ == "__main__":
    main()