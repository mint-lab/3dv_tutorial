import cv2
import numpy as np

def main():
    input, cover = "../bin/data/blais.mp4", "../bin/data/blais.jpg"
    f, cx, cy = 1000, 320, 240
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

    # Run pose extimation
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
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
        ret, rvec, tvec, inlier = cv2.solvePnPRansac(objectPoints=obj_points, 
                                                    imagePoints=img_points, 
                                                    cameraMatrix=K, 
                                                    distCoeffs=dist_coeff, 
                                                    useExtrinsicGuess=False, 
                                                    iterationsCount=500, 
                                                    reprojectionError=2., 
                                                    confidence=0.99)
        
        # print("Match data is ", type(match), len(match))
        # inlier_mask = np.zeros(len(match))


        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = (0,127,0),
                   matchesMask = inlier, #inlier_mask, # draw only inliers
                   flags = 2)        
        image_result = cv2.drawMatches(image, img_keypoint, obj_image, obj_keypoint, match, None, **draw_params)

        # Estimate camera pose with inliers
        try:
            inlier_num = len(inlier)
        except:
            inlier_num = 0

        if inlier_num > min_inlier_num:
            ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeff)

            # Draw the box on the image
            line_lower, _ = cv2.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
            line_upper, _ = cv2.projectPoints(box_upper, rvec, tvec, K, dist_coeff)

            image_result = cv2.polylines(image_result, np.int32([line_lower]), True, (255, 0, 0), 2)
            image_result = cv2.polylines(image_result, np.int32([line_upper]), True, (0, 0, 255), 2)
            for i in range(len(line_lower)):
                image_result = cv2.line(image_result, tuple(line_lower[i][0]), tuple(line_upper[i][0]), (0, 255, 0), 2, cv2.LINE_AA)


        # Show the image
        info = f"Inliers: {inlier_num*100/len(match):.3f} , Focal length: {f}"
        image_result = cv2.putText(image_result, info, (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        if cv2.waitKey(1) == ord('q'): break
        cv2.imshow("3DV Tutorial: Pose Estimation (Book)", image_result)

    cap.release()

if __name__ == "__main__":
    main()