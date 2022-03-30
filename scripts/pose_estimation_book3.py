import cv2
import numpy as np
import copy

def main():
    input, cover = "../bin/data/blais.mp4", "../bin/data/blais.jpg"
    f, cx_init, cy_init = 1000, 320, 240
    min_inlier_num = 100

    # Load the object image and extract features
    obj_image = cv2.imread(cover)

    fdetector = cv2.ORB_create()
    fmatcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")

    obj_keypoint, obj_descriptor = fdetector.detectAndCompute(obj_image, None)
    if (len(obj_keypoint)==0 or len(obj_descriptor)==0): raise Exception("No orb keypoints")
    # obj_descriptor = fmatcher.add(obj_descriptor)

    # Open a video
    cap = cv2.VideoCapture(input)

    # Prepare a box for simple AR
    box_lower = np.array([[30, 145, 0], [30, 200, 0], [200, 200, 0], [200, 145, 0]], dtype=np.float32)
    box_upper = np.array([[30, 145, -50], [30, 200, -50], [200, 200, -50], [200, 145, -50]], dtype = np.float32)

    # Calibrating camera params
    cam_param = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_S1_S2_S3_S4 | cv2.CALIB_FIX_TAUX_TAUY

    # Run pose extimation
    K = np.array([[f, 0, cx_init], [0, f, cy_init], [0, 0, 1]], dtype=np.float32)
    dist_coeff = np.zeros(5)
    while True:
        # Grab  an image from the video
        ret, image = cap.read()
        if not ret: break

        # Extract features and match them to the object features
        img_keypoint, img_descriptor = fdetector.detectAndCompute(image, None)
        if (len(img_keypoint)==0 or len(img_descriptor)==0): continue


        match = fmatcher.match(img_descriptor, obj_descriptor)
        if len(match) < min_inlier_num: continue

        obj_points, obj_project, img_points  = np.zeros((len(match), 3)), [], []
        for i, m in enumerate(match):
            obj_points[i, :2] = obj_keypoint[m.trainIdx].pt
            obj_project.append(obj_keypoint[m.trainIdx].pt)
            img_points.append(img_keypoint[m.queryIdx].pt)

        obj_points = np.array(obj_points, dtype=np.float32)
        obj_project = np.array(obj_project, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)

        # Deterimine whether each matched feature is an inlier or not        
        # inlier_mask = np.zeros(len(match))
        H, inlier_mask = cv2.findHomography(img_points, obj_points, cv2.RANSAC, 2)
        draw_params = dict(matchColor = (0,0, 255), # draw matches in green color
                   singlePointColor = (0, 255, 0),
                   matchesMask = inlier_mask, # draw only inliers
                   flags = 2)
        print(inlier_mask[inlier_mask==1].shape)
        print(inlier_mask.shape)
        print(inlier_mask[inlier_mask==1])
        image_result = cv2.drawMatches(image, img_keypoint, obj_image, obj_keypoint, match, None, **draw_params)

        # Calibrate the camera and estimate camera pose with inliers
        f = 0
        try:
            inlier_num = len(inlier_mask)
        except:
            inlier_num = 0

        if inlier_num > min_inlier_num:
            obj_inlier, img_inlier = [], []
            try:            
                for idx in range(len(inlier_mask)):
                    if inlier_mask[idx]:
                        obj_inlier.append(obj_points[idx])
                        img_inlier.append(img_points[idx])
                obj_inlier = np.array(obj_inlier, dtype=np.float32)
                img_inlier = np.array(img_inlier, dtype=np.float32)

                rms, K, dist_coeff, rvecs, tvecs = cv2.calibrateCamera([obj_inlier], [img_inlier], (image.shape[0], image.shape[1]), cam_param, None)
                rvec = copy.copy(rvecs[0])
                tvec = copy.copy(tvecs[0])
                f = K[0][0]

                # Draw the box on the image
                line_lower, _ = cv2.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
                line_upper, _ = cv2.projectPoints(box_upper, rvec, tvec, K, dist_coeff)

                image_result = cv2.polylines(image_result, np.int32([line_lower]), True, (255, 0, 0), 2)
                image_result = cv2.polylines(image_result, np.int32([line_upper]), True, (0, 0, 255), 2)
                for i in range(len(line_lower)):
                    image_result = cv2.line(image_result, tuple(line_lower[i][0]), tuple(line_upper[i][0]), (0, 255, 0), 2, cv2.LINE_AA)
            except Exception as e:
                # print(e)
                continue

        # Show the image
        info = f"Inliers: {inlier_num*100/len(match):.3f} , Focal length: {f}"
        image_result = cv2.putText(image_result, info, (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        if cv2.waitKey(1) == ord('q'): break
        cv2.imshow("3DV Tutorial: Pose Estimation (Book)", image_result)

    cap.release()

if __name__ == "__main__":
    main()