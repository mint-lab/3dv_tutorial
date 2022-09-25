from pickletools import read_uint1
from scipy.spatial.transform import Rotation
from copy import deepcopy
import cv2
import numpy as np
import time

def get_camera_mat(cam_vec):
    f, cx, cy = cam_vec[0], cam_vec[1], cam_vec[2]
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)

def update_camera_pose(cam_vec, R, t):
    # 
    t = t.squeeze()
    rvec = Rotation.from_matrix(R).as_rotvec()
    result = np.array([cam_vec[0], cam_vec[1], cam_vec[2], rvec[0], rvec[1], rvec[2], t[0], t[1], t[2]], dtype=np.float32)
    return result

def get_projection_mat(cam_vec):
    K = get_camera_mat(cam_vec=cam_vec)
    R = Rotation.from_rotvec(cam_vec[3:6]).as_matrix()
    t = cam_vec[6:9, np.newaxis]
    Rt = np.hstack((R, t))
    result = K @ Rt
    return result

def isBadPoint(point_3d, camera1, camera2, Z_limit, max_cos_parallax):
    # 무슨 내용인지 모름
    if point_3d[2] < -Z_limit or point_3d[2] > Z_limit:
        return True
    rvec1, rvec2 = np.array([camera1[3], camera1[4], camera1[5]], dtype=np.float32), np.array([camera2[3], camera2[4], camera2[5]], dtype=np.float32)
    R1, R2 = Rotation.from_rotvec(rvec1).as_matrix(), Rotation.from_rotvec(rvec2).as_matrix()
    # rot_vec to rot_mat
    t1, t2 = np.array([[camera1[6], camera1[7], camera1[8]]], dtype=np.float32), np.array([[camera2[6], camera2[7], camera2[8]]], dtype=np.float32)
    p1 = R1 @ point_3d[:, np.newaxis] + t1.T
    p2 = R2 @ point_3d[:, np.newaxis] + t2.T
    if p1[2,0] <= 0 or p2[2,0] <= 0 : return True
    v2 = R1 @ R2.T @ p2
    cos_parallax = p1.T @ v2 / (np.linalg.norm(p1) * np.linalg.norm(v2))
    if cos_parallax > max_cos_parallax: return True
    return False


def main():
    img_path = "./bin/data/relief/%02d.jpg"
    img_resize = 0.25
    f_init, cx_init, cy_init, Z_init, Z_limit = 500, -1, -1, 2, 100
    ba_loss_width = 9
    max_cos_parallax = np.cos(10*np.pi / 180)
    min_inlier_num, ba_num_iter = 200, 200
    SHOW_MATCH = False

    # Load images and extract features
    img_keypoints = []
    img_descriptors = []
    img_set = []
    detector = cv2.BRISK_create()
    cam = cv2.VideoCapture(img_path)

    while True:
        _, img = cam.read()
        if img is None: break
        img = cv2.resize(img, dsize=(0, 0), fx=img_resize, fy=img_resize)
        img_keypoint, img_descriptor = detector.detectAndCompute(img, None)
        img_keypoints.append(img_keypoint)
        img_descriptors.append(img_descriptor)
        img_set.append(img)
    cam.release()

    if cx_init < 0: cx_init = int(img_set[0].shape[1] / 2)
    if cy_init < 0: cy_init = int(img_set[0].shape[0] / 2)

    img_keypoints = np.array(img_keypoints, dtype=object)
    img_descriptors = np.array(img_descriptors, dtype=object)
    img_set = np.array(img_set)

    # Match features and find good matches
    fmatcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
    match_pair, match_inlier = [], []
    for i in range(len(img_set)):
        for j in range(i+1, len(img_set)):
            src, dst, inlier = [], [], []
            match = fmatcher.match(img_descriptors[i], img_descriptors[j])
            match = np.array(match)
            for m in match:
                src.append(img_keypoints[i][m.queryIdx].pt)
                dst.append(img_keypoints[j][m.trainIdx].pt)
            
            src = np.array(src, dtype=np.float32)
            dst = np.array(dst, dtype=np.float32)
            
            F, inlier_mask = cv2.findFundamentalMat(src, dst, cv2.RANSAC) # inlier_mask = 3x3
            for k in range(len(inlier_mask)):
                if inlier_mask[k]: 
                    inlier.append(match[k]) #
            inlier = np.array(inlier)
            print(f"3DV Tutorial: Image {i} - {j} are matched ({inlier.size} / {match.size}).\n")

            # Determin whether the image pair is good or not
            if inlier.size < min_inlier_num: continue
            print(f"3DV Tutorial: Image {i} - {j} are selected.\n")
            match_pair.append((i, j))
            match_inlier.append(inlier)

            if SHOW_MATCH:
                match_image = cv2.drawMatches(img_set[i], img_keypoints[i], img_set[j], img_keypoints[j], match, (0, 255, 0), (255, 0, 0), matchesMask=inlier_mask)
                cv2.imshow("3DV Tutorial: Structure-from-Motion", match_image)
                cv2.waitKey()
    if len(match_pair) < 1: return 

    # Start Initialize cameras(cam_params, rotation, translation)
    cameras = np.full((img_set.shape[0], 9), np.array([f_init, cx_init, cy_init, 0, 0, 0, 0, 0, 0]), dtype=np.float32)
    best_pair = 0
    best_score = [i for i in range(len(match_inlier))]
    best_points_3d = None
    while True:
        # 1) Select the best pair
        for i in best_score:
            if i > best_score[best_pair]:
                best_pair = i
        if best_score[best_pair] == 0:
            print("3DV Tutorial: There is no good match. Try again after reducing 'max_cos_parallax\n")
            return -1
        best_cam0, best_cam1 = match_pair[best_pair][0], match_pair[best_pair][1]

        # 2) Estimate relative pose from the best two view (epipolar geometry)
        src, dst = [], []
        for itr in match_inlier[best_pair]:
            src.append(img_keypoints[best_cam0][itr.queryIdx].pt)
            dst.append(img_keypoints[best_cam1][itr.trainIdx].pt)
        src = np.array(src, dtype=np.float32)
        dst = np.array(dst, dtype=np.float32)
        E, inlier_mask = cv2.findEssentialMat(src, dst, f_init, (cx_init, cy_init), cv2.RANSAC, 0.999, 1.0)
        inlier_num, R, t, mask = cv2.recoverPose(E, src, dst)
        
        for r in range(len(inlier_mask)-1, -1, -1): # 왜 뒤에서 부터 한 것일까...
            if not inlier_mask[r]:
                # Remove additionally detected ouliers
                src = np.delete(src, r, axis=0)
                dst = np.delete(dst, r, axis=0)
                match_inlier[best_pair] = np.delete(match_inlier[best_pair], r)
        cameras[best_cam0] = update_camera_pose(cameras[best_cam0], R, t)

        # 3) Reconstruct 3D points of the best two views (triangulation)
        P0, P1 = get_projection_mat(cameras[best_cam0]), get_projection_mat(cameras[best_cam1])
        best_points_3d = cv2.triangulatePoints(P0, P1, src.T, dst.T)
        best_points_3d = best_points_3d.T
        best_points_3d[0] /= best_points_3d[3]
        best_points_3d[1] /= best_points_3d[3]
        best_points_3d[2] /= best_points_3d[3]
        
        best_score[best_pair] = 0
        for best_point_3d in best_points_3d:
            if isBadPoint(best_point_3d[:3], cameras[best_cam0], cameras[best_cam1], Z_limit, max_cos_parallax): continue
            best_score[best_pair] += 1
        print(f"3DV Tutorial: Image {best_cam0} - {best_cam1} were checked as the best match (# of inliers = {len(match_inlier[best_pair])}, # of good points = {best_score[best_pair]})")
        if best_score[best_pair] > 100: break
        
    # End Initialize cameras
    best_cam0 = match_pair[best_pair]

    # Prepare the initial 3D points

    # Start Prepare BA
    while True:
        # 4) Select the next image to add

        # 5) Estimate relative pose of the next view (PnP)

        # 6) Reconstruct newly observed 3D points (triangulation)
        
        # 7) Optimize camera pose and 3D points together (bundle adjustment)
        break
    # End Prepare BA


    # Store the 3D points to an XYZ file
    # points_3d_name = "sfm_global(point).xyz"
    # with open(points_3d_name, "wt") as f:
    #     for i in range(n_point_3d):
    #         data = f"{opt_points_3d[i,0]:.3f} {opt_points_3d[i,1]:.3f} {opt_points_3d[i,2]:.3f}\n"
    #         f.write(data)

    # points_rgb_name = "sfm_global(rgb).xyz"
    # with open(points_rgb_name, "wt") as f:
    #     for i in range(n_point_3d):
    #         data = f"{points_rgb[i][0]:.3f} {points_rgb[i][1]:.3f} {points_rgb[i][2]:.3f}\n"
    #         f.write(data)

    # camera_file = "sfm_global(camera).xyz"
    # with open(camera_file, 'wt') as f:
    #     for i in range(n_cameras):
    #         data = f"{opt_cam_params[i, 0]:.3f} {opt_cam_params[i, 1]:.3f} {opt_cam_params[i, 2]:.3f} {opt_cam_params[i, 3]:.3f} {opt_cam_params[i, 4]:.3f} {opt_cam_params[i, 5]:.3f}\n"
    #         f.write(data)
    
    # print("END!!!")
if __name__ == "__main__":
    main()