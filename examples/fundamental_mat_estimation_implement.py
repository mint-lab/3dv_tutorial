import numpy as np
import cv2 as cv

def findFundamentalMat(pts1, pts2):
    if len(pts1) == len(pts2):
        # Make homogeneous coordiates if necessary
        if pts1.shape[1] == 2:
            pts1 = np.hstack((pts1, np.ones((len(pts1), 1), dtype=pts1.dtype)))
        if pts2.shape[1] == 2:
            pts2 = np.hstack((pts2, np.ones((len(pts2), 1), dtype=pts2.dtype)))

        # Solve 'Ax = 0'
        A = []
        for p, q in zip(pts1, pts2):
            A.append([q[0]*p[0], q[0]*p[1], q[0]*p[2], q[1]*p[0], q[1]*p[1], q[1]*p[2], q[2]*p[0], q[2]*p[1], q[2]*p[2]])
        _, _, Vt = np.linalg.svd(A, full_matrices=True)
        x = Vt[-1]

        # Reorganize `x` as `F` and enforce 'rank(F) = 2'
        F = x.reshape(3, -1)
        U, S, Vt = np.linalg.svd(F)
        S[-1] = 0
        F = U @ np.diag(S) @ Vt
        return F / F[-1,-1] # Normalize the last element as 1

if __name__ == '__main__':
    pts0 = np.loadtxt('../data/image_formation0.xyz')
    pts1 = np.loadtxt('../data/image_formation1.xyz')

    my_F = findFundamentalMat(pts0, pts1)
    cv_F, _ = cv.findFundamentalMat(pts0, pts1, cv.FM_8POINT)

    print('\n### My Fundamental Matrix')
    print(my_F)
    print('\n### OpenCV Fundamental Matrix')
    print(cv_F) # Note) The result is slightly different because OpenCV considered normalization