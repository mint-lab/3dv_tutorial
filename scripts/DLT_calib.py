from copy import deepcopy
import cv2
import numpy as np
from math import sqrt

class CustomCalibrator():
    def __init__(self, points_3d):
        self.points_2d = []
        self.img = None
        self.clean_img = None
        self.video = '/dev/video2'
        self.points_3d = points_3d    

    def take_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points_2d.append((x, y))
            self.img = cv2.circle(self.img, (x, y), 3, (0, 0, 255), 3)

    def find_homography(self):
        """Find homography matrix"""
        # check point's length
        if len(self.points_2d) != 6:
            raise Exception("Not enough points!")
        
        self.points_2d = np.array(self.points_2d, dtype=np.float32)
        self.points_3d = np.array(self.points_3d, dtype=np.float32)
        
        # Make A, B matrix
        A = None
        B = None
        for i in range(len(self.points_2d)):
            rx, ry, rz = self.points_3d[i]
            u, v = self.points_2d[i]
            A_matrix = np.array([[rx, ry, rz, 1, 0, 0, 0, 0, -u * rx, -u * ry, -u * rz], 
                                 [0, 0, 0, 0, rx, ry, rz, 1, -v * rx, -v * ry, -v * rz]], 
                                dtype=np.float32)
            B_matrix = np.array([[u, v]], dtype=np.float32)
            if i == 0:
                A = A_matrix.copy()
                B = B_matrix.copy()
            else:
                A = np.vstack((A, A_matrix))
                B = np.hstack((B, B_matrix))
        
        # Before pseudo inverse, Check det(A.T @ A) 
        det = np.linalg.det((A.T @ A))
        if det == 0:
            # raise Exception("(A.T @ A) is Singular Matrix!!!")
            print(A.T @ A)

        # Do pseudo inverse
        V = np.linalg.inv(A.T @ A) @ A.T @ B.T
        V = V.T
        return V

    def calibrate_homography(self, V):
        # decompose matrix items
        h11, h12, h13, h14, h21, h22, h23, h24, h31, h32, h33 = V[0].tolist() # squeeze를 쓴다면 더 세련될 것.
        print(h11, h12, h13, h14, h21, h22, h23, h24, h31, h32, h33)
        # 1) calculate tz
        tz = 1.0 / sqrt(h31 ** 2 + h32 ** 2 + h33 ** 2)
        print(tz)
        # 2) calculate R3
        R3 = tz * np.array([[h31, h32, h33]])
        # 3) calculate u0
        u0 = tz * np.array([[h11, h12, h13]]) @ R3.T 
        # 4) calculate v0
        v0 = tz * np.array([[h21, h22, h23]]) @ R3.T
        # 5) calculate fx
        fx = tz * np.array([[h11, h12, h13]]) - u0 * R3
        fx = np.linalg.norm(fx)
        # 6) calculate fy
        fy = tz * np.array([[h21, h22, h23]]) - v0 * R3
        fy = np.linalg.norm(fy)
        # 7) ca;culate R1
        R1 = tz / fx * np.array([[h11, h12, h13]]) - u0 / fx * R3
        # 8) calculate R2
        R2 = tz / fy * np.array([[h21, h22, h23]]) - v0 / fy * R3
        # 9) calculate tx
        tx = tz / fx * (h14 - u0)
        # 10) calculate ty
        ty = tz / fy * (h24 - v0)
        
        # Return Homography, Intrinsic and Extrinsic matrix
        R = np.vstack((np.vstack((R1, R2)), R3))
        t = np.array([[tx, ty, tz]], dtype=object)
        H1 = np.array([[h11, h12, h13, h14]])
        H2 = np.array([[h21, h22, h23, h24]])
        H3 = np.array([[h31, h32, h33, 1]])
        homography_matrix = np.vstack((np.vstack((H1, H2)), H3))
        extrinsic_matrix = np.hstack((R, t.T))
        intrinsic_matrix = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]], dtype=np.float32)
        # test_R = R.T
        # test_T = -R.T @ t
        return homography_matrix, intrinsic_matrix, extrinsic_matrix # , test_R, test_T

    def reprojection(self, homography_matrix):
        # Get Reprojected points
        reprojected_points = []
        for point in self.points_3d:
            point = np.append(point, 1)
            reprojected_point = homography_matrix @ point.T
            reprojected_point[0] /= reprojected_point[2]
            reprojected_point[1] /= reprojected_point[2]
            reprojected_points.append(reprojected_point.T[:2])
        
        
        # Show 2d proj positions
        show_points = np.array(reprojected_points, dtype=np.int32)
        for reproj in show_points:
            self.img = cv2.circle(self.img, tuple(reproj), 3, (0, 255, 0), 3)
        cv2.imwrite(f"saved_proj_img.png", self.img)

        # Calc Error
        point_2d = self.points_2d.astype(np.float32)
        err = np.linalg.norm(point_2d - reprojected_points)
        print("points_2d = ", point_2d)
        print("reproj 2d = ", reprojected_points)
        return err

    def pipeline(self, **mode):
        cap = cv2.VideoCapture(self.video)
        cv2.namedWindow('img')
        cv2.setMouseCallback('img', self.take_points)
        n_save = 0
        FIX_IMAGE = False
        print(mode)
        # get points_2d
        if mode['mode'] == 'using_video':
            GET_CLEAN_IMG = True
            while True:
                if not FIX_IMAGE:
                    ret, self.img = cap.read()
                    if ret != None and GET_CLEAN_IMG:
                        self.clean_img = deepcopy(self.img)
                        GET_CLEAN_IMG = False

                # Add info to IMG
                h, w, _ = self.img.shape

                # View Image
                cv2.imshow("img", self.img)
                
                # key event
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    FIX_IMAGE = True
                elif key == ord('e'):
                    n_save += 1
                    cv2.imwrite(f"saved_img_{n_save}.png", self.img)
            cap.release()

        elif mode['mode'] == 'image_cal':
            self.img = cv2.imread(mode['img'])
            self.clean_img = deepcopy(self.img)
            while True:
                h, w, _ = self.img.shape

                # View Image
                cv2.imshow("img", self.img)
                
                # key event
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    FIX_IMAGE = True
                elif key == ord('e'):
                    n_save += 1
                    cv2.imwrite(f"saved_img_{n_save}.png", self.img)
            cap.release()

        elif mode['mode'] == 'done_image':
            self.points_2d = [(402, 324), (434, 300), (451,113), (414, 124), (270,172), (208, 315)]

        # find homography items
        V = self.find_homography()
        
        # From homography, find intrinsic and extrinsic
        homography, intrinsic, extrinsic= self.calibrate_homography(V) # , test_R, test_T 
        print("homography matrix\n", homography)
        print("intrinsic matrix\n", intrinsic)
        print("extrinsic matrix, and shape\n", extrinsic.shape, extrinsic)
        
        # reproject points to image
        err = self.reprojection(homography)
        
        # Show projected image
        if mode['mode'] != 'done_image':
            self.img = cv2.putText(self.img, f"err = {err:.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("img", self.img)
            cv2.waitKey(0)

        return homography, intrinsic, extrinsic, self.clean_img

def main():
    # cm 단위로 했음. m단위로 바꾸기.
    points_3d = [(3,0,0), (5,0,0), (5,0,6), (3,0,6), (0,2,5), (0,5,0)]
    cc = CustomCalibrator(points_3d=points_3d)
    cc.pipeline()

if __name__ == "__main__":
    main()