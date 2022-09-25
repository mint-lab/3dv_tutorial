import cv2
import numpy as np

def main():
    input = "../bin/data/chessboard.avi"
    K = np.array([[432.7390364738057, 0, 476.0614994349778],
                  [0, 431.2395555913084, 288.7602152621297],
                  [0, 0, 1]], dtype=np.float32)
    dist_coeff = np.array([-0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194], dtype=np.float32)
    board_pattern = (10, 7)
    board_cellsize = 0.025
    criteria = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

    # Open a video
    cap = cv2.VideoCapture(input)

    # Prepare a 3D box for simple AR
    box_lower = np.array([[4*board_cellsize, 1*board_cellsize,0], [5*board_cellsize, 2*board_cellsize, 0], [5*board_cellsize, 4*board_cellsize, 0], [4*board_cellsize, 4* board_cellsize,0]], dtype=np.float32)
    box_upper = np.array([[4*board_cellsize, 2*board_cellsize,-board_cellsize], [5*board_cellsize, 2*board_cellsize, -board_cellsize], [5*board_cellsize, 4*board_cellsize, -board_cellsize], [4*board_cellsize, 4* board_cellsize,-board_cellsize]], dtype=np.float32)

    # Prepare 3D points on a chessboard
    obj_points = np.zeros((board_pattern[0]*board_pattern[1],3))
    obj_points_input = []
    for r in range(7):
        for c in range(10):
            obj_points_input.append([board_cellsize * c, board_cellsize * r])
    obj_points_input = np.array(obj_points_input, dtype=np.float32)
    obj_points[:, :2] = obj_points_input

    # Run pose estimation
    while True:
        # Grab an image from the video
        ret, image = cap.read()
        if not ret: break

        # Estimate camera pose
        ret, img_points = cv2.findChessboardCorners(image, board_pattern, criteria)
        if ret:
            ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeff)

            # Draw the box on the image
            line_lower, _ = cv2.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
            line_upper, _ = cv2.projectPoints(box_upper, rvec, tvec, K, dist_coeff)

            # Change 4x1 matrix (CV_64FC2) to 4x2 (CV_32SC1) / np.int32([line_lower]) or line_lower.reshape(-1, 1, 2)
            image = cv2.polylines(image, np.int32([line_lower]), True, (255, 0, 0), 2) 
            image = cv2.polylines(image, np.int32([line_upper]), True, (0, 0, 255), 2)
            for i in range(len(line_lower)):
                image = cv2.line(image, tuple(line_lower[i][0]), tuple(line_upper[i][0]), (0, 255, 0), 2, cv2.LINE_AA)

            # Print camera position
            R,_ = cv2.Rodrigues(rvec)
            p = -R.T @ tvec
            string_info = f"XYZ: [{p.T[0][0]:.3f} {p.T[0][1]:.3f} {p.T[0][2]:.3f}]"
            image = cv2.putText(image, string_info, (5,16), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        cv2.imshow("3DV Tutorial: Pose Estimation (Chess board)", image)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()

if __name__ == "__main__":
    main()