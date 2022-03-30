from signal import raise_signal
import cv2
import numpy as np
import glob

def main():
    input = "../bin/data/07/image_0/%06d.png"
    f = 707.0912
    c = (601.8873, 183.1104)
    use_5pt = True
    min_inlier_num = 100

    # Open a file to write camera trajectory
    my_file = "../bin/data/vo_epipolar_v2.xyz"
    camera_trajectory = open(my_file, 'wt')
    if camera_trajectory == 0: raise Exception("Can't make file")

    # Open a video and get initial image
    cap = cv2.VideoCapture(input)
    if cap.isOpened() == False: raise Exception("Cant read images")

    gray_prev = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    
    camera_pose = np.eye(4, dtype=np.float64)

    while cap.isOpened():
        ret, image = cap.read()
        if ret == False: break
        if image.shape[2] == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        point_prev = cv2.goodFeaturesToTrack(gray_prev, 2000, 0.01, 10)
        
        point, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray, point_prev, None)
        gray_prev = gray

        if use_5pt:
            E, inliner_mask = cv2.findEssentialMat(point_prev, point, f, c, cv2.FM_RANSAC, 0.99, 1)
        else:
            F, inliner_mask = cv2.findFundamentalMat(point_prev, point, cv2.FM_RANSAC, 1, 0.99)
            K = np.array([[f, 0, c[0]], [0,f,c[1]], [0,0,1]])
            E = K.T @ F @ K

        inlier_num, R, t, mask = cv2.recoverPose(E, point, point_prev)
        
        if inlier_num > min_inlier_num:
            T = np.eye(4)
            T[0:3, 0:3] = R * 1.0
            T[0:3,[3]] = t* 1.0
            camera_pose = camera_pose @ np.linalg.inv(T) # 왜 역행렬을 그냥 곱함? 그대로 두면 되는데?
            # camera_pose = np.linalg.inv(T)
        
        # Show the iamge and write camera pose
        if image.shape[2] <3: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for i in range(len(point_prev)):
            px_prev, py_prev = point_prev[i][0][0], point_prev[i][0][1]
            px, py = point[i][0][0], point[i][0][1]
            if inliner_mask[i] > 0: image = cv2.line(image, (px_prev, py_prev), (px, py), (0,0,255))
            else: image = cv2.line(image, (px_prev, py_prev), (px, py), (0,127,0))
        
        info = f"Inliers: {inlier_num} ({100*inlier_num/len(point)}), XYZ: [{camera_pose[0][2]:.3f} {camera_pose[1][2]:.3f} {camera_pose[2][2]:.3f}]"
        camera_trajectory.write(f"{camera_pose[0][2]:.6f} {camera_pose[0][2]:.6f} {camera_pose[0][2]:.6f}\n")
        cv2.putText(image, info, (5,15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
        cv2.imshow("test", image)
        key = cv2.waitKey(1)
        if key == ord('q'): break

    cap.release()        
    camera_trajectory.close()

if __name__ == "__main__":
    main()