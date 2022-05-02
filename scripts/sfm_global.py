import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from copy import deepcopy
import time
class ProjectError():
    def __init__(self,n_cameras, n_points):
        self.n_cameras = n_cameras
        self.n_points = n_points

    def sparsity_jacobian_inc(self, camera_indices, point_indices):
        
        m = camera_indices.size * 2 # m = 800 x 2
        n = 6 + self.n_points * 3 # n = 6 * cameras = 5 + 3 * points = 160 --> 510
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1

        for s in range(3):
            A[2 * i, 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, 6 + point_indices * 3 + s] = 1

        return A

    def sparsity_jacobian_global(self, camera_indices, point_indices):
        # camera, point indices = params to 2d points
        m = camera_indices.size * 2 # m = 800 x 2
        n = 9 * self.n_cameras + self.n_points * 3 # n = 6 * cameras = 5 + 3 * points = 160 --> 510
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(9):
            A[2 * i, camera_indices * 9 + s] = 1
            A[2 * i + 1, camera_indices * 9 + s] = 1

        for s in range(3):
            A[2 * i, 9  + point_indices * 3 + s] = 1
            A[2 * i + 1, 9 + point_indices * 3 + s] = 1

        return A

    def project2(self, cam_params, points):
        points_proj = []
        for i in range(len(cam_params)):
            # Init params
            f, cx, cy = cam_params[i, 0], cam_params[i, 1], cam_params[i, 2]
            rvec, tvec = cam_params[i, 3:6], cam_params[i, 6:9]
            point_3d = points[i]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)

            # Do Projection
            rot = Rotation.from_rotvec(rvec)
            point_proj = rot.apply(point_3d)
            point_proj += tvec
            point_proj = point_proj @ K.T
            point_proj[0] /= point_proj[2]
            point_proj[1] /= point_proj[2]
            points_proj.append(point_proj[:2])
        return points_proj


    @staticmethod
    def cost_func(params, points_2d, sfm, cam_indices, point_indices,cls):
        """
        params = (camera_params + points_3d).ravel(). 
        n_cameras = cameras number, which is also index of scene
        cls is ba itself
        camera params = f, cx, cy, rx, ry, rz, tx, ty, tz
        """
        cam_params = params[: cls.n_cameras * 9].reshape((cls.n_cameras, 9))
        points_3d = params[cls.n_cameras * 9:].reshape((cls.n_points, 3))

        proj_points_3d = cls.project2(cam_params[cam_indices], points_3d[point_indices])
        result = points_2d - proj_points_3d
        return result.ravel()

class SFM():
    """3D point idx Gen data structure"""
    def __init__(self):
        self._sfm_dict = {}

    def gen_dict(self, cam_idx, point_2d_idx):
        """
        Gen data set. 
        (Cam idx, 2D Pts idx) = 3D Pts idx
        3D Pts Idx = (Cam idx, 2D Pts idx) --> find 하기 어려울 수 있음.
        """
        if self._sfm_dict.get((cam_idx, point_2d_idx)) is None:
            self._sfm_dict[(cam_idx, point_2d_idx)] = None

        return self._sfm_dict[(cam_idx, point_2d_idx)]

    def pop(self, key):
        self._sfm_dict.pop(key)

    def gen_indices(self, img_keypoints):
        cam_indices = []
        point_indices = []
        points_2d = []
        for (cam_idx, m_idx), p3_idx in self._sfm_dict.items():
            points_2d.append(img_keypoints[cam_idx][m_idx].pt)
            cam_indices.append(cam_idx)
            point_indices.append(p3_idx)
        return np.array(cam_indices), np.array(point_indices), np.array(points_2d)

def main():
    img_path = "../bin/data/relief/%02d.jpg"
    img_resize = 0.25
    f_init, cx_init, cy_init, Z_init, Z_limit = 500, -1, -1, 2, 100
    ba_loss_width = 9
    min_inlier_num, ba_num_iter = 200, 200
    SHOW_MATCH = False

    # Load images and extract features
    # USE BRISK feature detection
    img_keypoints = []
    img_descriptors = []
    img_set = []

    detector = cv2.BRISK_create()

    cam = cv2.VideoCapture(img_path)
    h, w = 0, 0
    while True:
        _, img = cam.read()
        if img is None: break
        img = cv2.resize(img, dsize=(0, 0), fx=img_resize, fy=img_resize)
        img_keypoint, img_descriptor = detector.detectAndCompute(img, None)
        img_keypoints.append(img_keypoint)
        img_descriptors.append(img_descriptor)
        img_set.append(img)
        h, w, _ = img.shape
    cam.release()
    if cx_init < 0 or cy_init < 0:
        cx_init = w / 2
        cy_init = h / 2

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
            
            F, inlier_mask = cv2.findFundamentalMat(src, dst, cv2.RANSAC)
            for k in range(len(inlier_mask)):
                if inlier_mask[k]: 
                    inlier.append(match[k]) # 매칭된 index를 넣음.
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
    # if match_pair.size < 1: return 
    
    # Find 0 - 1 - 2 Covisibility Matched points
    points_3d = []
    points_rgb = []
    xs_visited = SFM()
    for i in range(len(match_pair)):
        for j in range(len(match_inlier[i])):
            cam1_idx, cam2_idx = match_pair[i][0], match_pair[i][1] 
            pts1_2d_idx, pts2_2d_idx = match_inlier[i][j].queryIdx, match_inlier[i][j].trainIdx

            value1 = xs_visited.gen_dict(cam_idx=cam1_idx, point_2d_idx=pts1_2d_idx)
            value2 = xs_visited.gen_dict(cam_idx=cam2_idx, point_2d_idx=pts2_2d_idx)

            # 둘 다 값이 있으면
            if value1 != None and value2 != None:
                # 근데 서로 이상한걸 가리키면
                if value1 != value2:
                    xs_visited.pop((cam1_idx, pts1_2d_idx))
                    xs_visited.pop((cam2_idx, pts2_2d_idx))
                
                continue

            X_idx = 0
            if value1 != None:
                X_idx = value1
            elif value2 != None:
                X_idx = value2
            else:
                X_idx = len(points_3d)
                points_3d.append(np.array([0, 0, Z_init]))
                rgb_p = img_keypoints[cam1_idx][pts1_2d_idx].pt
                points_rgb.append(img_set[cam1_idx][int(rgb_p[1]), int(rgb_p[0])])
            if value1 == None:
                xs_visited._sfm_dict[(cam1_idx, pts1_2d_idx)] = X_idx
            if value2 == None:
                xs_visited._sfm_dict[(cam2_idx, pts2_2d_idx)] = X_idx

    print(f"3DV Tutorial: # of 3D points: {len(points_3d)}")
    
    ### Init Parameters ###
    points_3d = np.array(points_3d, dtype=np.float32)

    n_point_3d = len(points_3d)
    n_cameras = len(img_set)
    init_param = np.array([f_init, cx_init, cy_init, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    cam_params = np.full( (len(img_set), len(init_param)), init_param, dtype=np.float32)
    params = np.hstack((cam_params.ravel(), points_3d.ravel()))

    BA = ProjectError(n_cameras=n_cameras, n_points=n_point_3d)

    # Ravel 2d points
    cam_indices, point_indices, points_2d = xs_visited.gen_indices(img_keypoints=img_keypoints)
    J = BA.sparsity_jacobian_global(camera_indices=cam_indices, point_indices=point_indices)

    start = time.time()
    opt = least_squares(BA.cost_func, params, jac_sparsity=J, method='trf', ftol=1e-4, verbose = 2, args=(points_2d, xs_visited, cam_indices, point_indices, BA))
    total = time.time() - start
    print(f"Total time is {total:.3f}")

    # Mark errorneous points to reject them
    opt_cam_params = opt.x[:n_cameras * 9].reshape(n_cameras, 9)
    opt_points_3d = opt.x[n_cameras * 9:].reshape(n_point_3d, 3)

    for i in range(len(opt_cam_params)):
        msg = f"3DV Tutorial: Camera {i+1}'s (f, cx, cy) = {opt_cam_params[i][0]:.1f} {opt_cam_params[i][1]:.1f} {opt_cam_params[i][2]:.1f}"
        print(msg)
    
    # Store the 3D points to an XYZ file
    points_3d_name = "sfm_global(point).xyz"
    with open(points_3d_name, "wt") as f:
        for i in range(n_point_3d):
            data = f"{opt_points_3d[i,0]:.3f} {opt_points_3d[i,1]:.3f} {opt_points_3d[i,2]:.3f}\n"
            f.write(data)

    points_rgb_name = "sfm_global(rgb).xyz"
    with open(points_rgb_name, "wt") as f:
        for i in range(n_point_3d):
            data = f"{points_rgb[i][0]:.3f} {points_rgb[i][1]:.3f} {points_rgb[i][2]:.3f}\n"
            f.write(data)

    camera_file = "sfm_global(camera).xyz"
    with open(camera_file, 'wt') as f:
        for i in range(n_cameras):
            data = f"{opt_cam_params[i, 0]:.3f} {opt_cam_params[i, 1]:.3f} {opt_cam_params[i, 2]:.3f} {opt_cam_params[i, 3]:.3f} {opt_cam_params[i, 4]:.3f} {opt_cam_params[i, 5]:.3f}\n"
            f.write(data)
    
    print("END!!!")
if __name__ == "__main__":
    main()