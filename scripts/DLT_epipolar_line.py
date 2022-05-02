from logging import raiseExceptions
import cv2
import numpy as np
from copy import deepcopy
from math import sqrt
from Homework_02 import CustomCalibrator

def screw(vec1x3):
    return np.array([[0, -vec1x3[2], vec1x3[1]], [vec1x3[2], 0, -vec1x3[0]], [-vec1x3[1], vec1x3[0], 0]])

def inv_intrinsic(intri):
    fx, cx, fy, cy = intri[0, 0], intri[0, 2], intri[1, 1], intri[1, 2]
    return np.array([[1./fx, 0., -cx/fx], [0, 1./fy, -cy/fy], [0., 0., 1.]])

class CustomEpipolarFinder():
    def __init__(self, s1_datas, s2_datas):
        """
        Get scene_1, scene_2 intrinsic, extrinsic datas, original img. Make Camera Matrix Inverse for each scene.
        """
        self.points_2d = []
        if len(s1_datas) == 3:
            s1_Intrinsic = s1_datas[0]
            s1_Extrinsic = s1_datas[1]
            self.s1_img = s1_datas[2]
            self.clean_img_1 = deepcopy(self.s1_img)
        else:
            raise Exception("Short datas. May be no images?")

        if len(s2_datas) == 3:
            s2_Intrinsic = s2_datas[0]
            s2_Extrinsic = s2_datas[1]
            self.s2_img = s2_datas[2]
            self.clean_img_2 = deepcopy(self.s2_img)
        else:
            raise Exception("Short datas. May be no images?")

        # Check Intrinsic Similarity and Get Inv Intrinsic mat
        self.s1_inv_intrinsic = inv_intrinsic(s1_Intrinsic)
        self.s2_inv_intrinsic = inv_intrinsic(s2_Intrinsic)
        
        # Get rotation and translation inform
        c1_M_r = np.vstack((s1_Extrinsic, np.array([[0.0, 0.0, 0.0, 1.0]]))).astype(float)
        c2_M_r = np.vstack((s2_Extrinsic, np.array([[0.0, 0.0, 0.0, 1.0]]))).astype(float)
        c1_M_c2 = c1_M_r @ np.linalg.inv(c2_M_r)
        self.rot_2to1 = c1_M_c2[:3, :3]
        trans_2to1 = c1_M_c2[:3, 3]
        self.trans_2to1 = screw(trans_2to1) # 3x3 screw sym matrix

    def take_points(self, event, x, y, flags, param):
        """
        L = Take interesting point on img!
        R = Delete point before chosen.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            print("left")
            self.points_2d.append((x, y))
            cv2.circle(self.s1_img, (x, y), 3, (0, 0, 255), 3)

        elif event == cv2.EVENT_RBUTTONDOWN:
            print("right")
            self.point_2d[0], self.point_2d[1] = 0, 0
            self.points_2d.pop()
            cv2.circle(self.s1_img, (x, y), 3, (0, 0, 255), 3)

    def get_epipolarline(self):
        """
        Find Epipolar line.
        """
        # Make point 2d shape (u, v, 1)
        self.points_2d = np.array([(u, v, 1) for (u, v) in self.points_2d])

        # Calculate Fundamental Matrix
        Fundamental = self.s1_inv_intrinsic.T @ self.trans_2to1.T @ self.rot_2to1 @ self.s2_inv_intrinsic
        
        # Calculate coefficient of point_2d
        coeffs = np.array([p2d @ Fundamental for p2d in self.points_2d])

        # Draw Line on Img
        x_range = np.linspace(0, self.s2_img.shape[1] - 1, 50)
        ys_range = []
        for coeff in coeffs:
            y_range = -coeff[0] / coeff[1] * x_range - coeff[2] / coeff[1]
            ys_range.append(y_range)

        # 1) Filter Outlier
        good_results = []
        for y_range in ys_range:
            good_result = []
            for i, y in enumerate(y_range):
                if y < 0 or y > self.s1_img.shape[0]:
                    continue
                else:
                    good_result.append((x_range[i], y))
            good_results.append(good_result)

        # 2) Use left and right value of results and draw it.
        for good_result in good_results:
            # get data from it
            left, right = good_result[0], good_result[-1]
            # Make it tuple
            left, right = (int(left[0]), int(left[1])), (int(right[0]), int(right[1]))
            cv2.line(self.s2_img, left, right, (0, 255, 0), 3)

        return coeffs

    def pipeline(self):
        """
        Get images and check point which I want to find epipolar line.
        """
        # From s1_img, pick interesting 2d point.
        cv2.namedWindow('img')
        cv2.setMouseCallback('img', self.take_points)
        while True:
            cv2.imshow("img", self.s1_img)
            if cv2.waitKey(0) == ord('q'):
                break
        # Find epipolar line on s2_img
        coeffs = self.get_epipolarline()
        # Check Correspondence
        print(f"coeffs of epipolar line = \n{coeffs}")
        # Show Img
        img = np.hstack((self.s1_img, self.s2_img))
        cv2.imshow("end", img)
        cv2.waitKey(0)
        # Save Img
        cv2.imwrite("good_epipolar_img.png", img)
        
        return


def main():
    points_3d = [(1,0,0), (3,0,1.9), (5,0,0), (0,1, 1.9), (0, 3, 0), (0, 5, 1.9)]
    mode1 = {'mode': 'image_cal', 'img': 'img1.png'}
    mode2 = {'mode': 'image_cal', 'img': 'img2.png'}

    cc1 = CustomCalibrator(points_3d=points_3d)
    cc2 = CustomCalibrator(points_3d=points_3d)


    _, s1_Intrinsic, s1_Extrinsic, s1_img = cc1.pipeline(**mode1)
    s1_datas = (s1_Intrinsic, s1_Extrinsic, s1_img)
    if s1_datas != None: print("s1 success!")
    
    _, s2_Intrinsic, s2_Extrinsic, s2_img = cc2.pipeline(**mode2)
    s2_datas = (s2_Intrinsic, s2_Extrinsic, s2_img)
    if s2_datas != None: print("s2 success!")

    cef = CustomEpipolarFinder(s1_datas=s1_datas, s2_datas=s2_datas)
    cef.pipeline()

if __name__ == "__main__":
    main()