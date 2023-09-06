import numpy as np
import cv2 as cv

def getPerspectiveTransform(src, dst):
    if len(src) == len(dst):
        # Make homogeneous coordiates if necessary
        if src.shape[1] == 2:
            src = np.hstack((src, np.ones((len(src), 1), dtype=src.dtype)))
        if dst.shape[1] == 2:
            dst = np.hstack((dst, np.ones((len(dst), 1), dtype=dst.dtype)))

        # Solve 'Ax = 0'
        A = []
        for p, q in zip(src, dst):
            A.append([0, 0, 0, q[2]*p[0], q[2]*p[1], q[2]*p[2], -q[1]*p[0], -q[1]*p[1], -q[1]*p[2]])
            A.append([q[2]*p[0], q[2]*p[1], q[2]*p[2], 0, 0, 0, -q[0]*p[0], -q[0]*p[1], -q[0]*p[2]])
        _, _, Vt = np.linalg.svd(A, full_matrices=True)
        x = Vt[-1]

        # Reorganize `x` as a matrix
        H = x.reshape(3, -1) / x[-1] # Normalize the last element as 1
        return H

if __name__ == '__main__':
    src = np.array([[115, 401], [776, 180], [330, 793], [1080, 383]], dtype=np.float32)
    dst = np.array([[0, 0], [900, 0], [0, 500], [900, 500]], dtype=np.float32)

    my_H = getPerspectiveTransform(src, dst)
    cv_H = cv.getPerspectiveTransform(src, dst) # Note) It accepts only 4 pairs of points.

    print('\n### My Planar Homography')
    print(my_H)
    print('\n### OpenCV Planar Homography')
    print(cv_H)