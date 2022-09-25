import numpy as np
import cv2
from math import cos, sin

def RX(rx):
    return np.array([[1,0,0],[0,cos(rx), -sin(rx)],[0, sin(rx), cos(rx)]])

def RY(ry):
    return np.array([[cos(ry),0,sin(ry)],[0,1,0],[-sin(ry),0,cos(ry)]])

def RZ(rz):
    return np.array([[cos(rz),-sin(rz),0],[sin(rz),cos(rz),0],[0,0,1]])

def read_file(file):
    X = []
    with open(file, 'rt') as f:
        for line in f.read().splitlines():
            x = []
            if line:
                for l in line.split(' '):
                    x.append(float(l))

            if x: X.append(x)
    X = np.array(X)
    ones = np.ones((X.shape[0],1))
    X = np.hstack((X, ones))
    return X

def main():
    f, cx, cy, noise_std = 1000, 320, 240, 1
    img_res = np.array([480, 640])
    cam_pose = np.array([[0,0,0],[-2,-2,0],[2,2,0],[-2,2,0],[2,-2,0]])
    cam_ori = np.array([[0, 0, 0], [-np.pi/12 , np.pi/12, 0], [np.pi/12, -np.pi/12, 0], [np.pi/12, np.pi/12, 0], [-np.pi/12, -np.pi/12, 0]])

    # Load a point cloud in the homogeneous coordinate
    file = "../bin/data/box.xyz" # Nx4
    X = read_file(file).T # 4xN

    # Generate images for each camera pose
    K = np.array([[f,0,cx],[0,f,cy],[0,0,1]])
    for i in range(len(cam_pose)):
        # Derive a projection matrix
        Rc = RZ(cam_ori[i][2]) @ RY(cam_ori[i][1]) @ RX(cam_ori[i][0])
        tc = cam_pose[i]
        x = Rc.T @ tc
        Rt = np.hstack((Rc.T, np.resize(-x, (3,1))))
        P = K @ Rt

        # # Project the points 
        x = P @ X # 3xN
        x /= x[2] # 3xN a, b, 1

        # # Add Gaussian noise
        noise = np.random.normal(0, noise_std, size=(x.shape))
        x += noise

        # Show and store the points
        image = np.zeros(img_res)
        for c in range(x.shape[1]):
            p = x.T[c]
            p = tuple([int(p[0]), int(p[1])])
            if p[0]>=0 and p[0] < image.shape[1] and p[1] >=0 and p[1] < image.shape[0]:
                image = cv2.circle(image, p, 2, 255, -1)
        
        points_file = f"../bin/data/image_formation{i}.xyz"
        with open(points_file, 'wt') as f:
            for c in range(x.shape[1]):
                data = f"{x[0][c]} {x[0][c]} 1\n"
                f.write(data)

        cv2.imshow(f"3DV_Tutorial: Image Formation {i}", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
