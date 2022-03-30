
import cv2
import numpy as np

def main():
    input0, input1 =  "../bin/data/image_formation0.xyz", "../bin/data/image_formation1.xyz"
    points0, points1 = None, None
    file0, file1 = open(input0, 'rt'), open(input1, 'rt')
    
    with open(input0,'rt') as file0: 
        points0 = [ list(xyz.split(' '))[:2] for xyz in file0.read().splitlines() if xyz != None ]
        points0 = np.array(points0, dtype=np.float32)
    with open(input1,'rt') as file0: 
        points1 = [ list(xyz.split(' '))[:2] for xyz in file1.read().splitlines() if xyz != None ]
        points1 = np.array(points1, dtype=np.float32)

    f, cx, cy = 1000, 320, 240
    # print(points0.shape)
    if len(points0) != len(points1): raise Exception("Not matching!")

    # # Estimate relative pose of two view
    F,_ = cv2.findFundamentalMat(points0, points1, cv2.FM_8POINT)
    K = np.array([[f,0,cx],[0,f,cy],[0,0,1]])
    E = K.T @ F @ K
    _, R, t, _ = cv2.recoverPose(E, points0, points1)

    # Reconstruct 3D points (triangulation)
    P0 = K @ np.eye(3,4, dtype=np.float32)
    Rt = np.hstack((R, t))
    P1 = K @ Rt
    X = cv2.triangulatePoints(P0, P1, points0.T, points1.T)
    X /= X[3]
    X = X.T
    
    triangular_file = "../bin/data/triangulation.xyz"
    with open(triangular_file, 'wt') as f:
        f.write(str(X[:,:3]))

if __name__=="__main__":
    main()
