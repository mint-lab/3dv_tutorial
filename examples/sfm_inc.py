import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from bundle_adjustment import plot_reconstruction
from sfm_global import reprojection_error, bundle_adjustment_sparsity


def get_camera_mat(camera):
    f, cx, cy = camera[6:9]
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


def get_projection_mat(camera):
    K = get_camera_mat(camera)
    R = Rotation.from_rotvec(camera[:3]).as_matrix()
    t = camera[3:6]
    return K @ np.column_stack((R, t))


def update_camera_pose(camera, R, t):
    rvec = Rotation.from_matrix(R).as_rotvec() if R.shape == (3, 3) else R.ravel()
    camera[:3] = rvec
    camera[3:6] = t.ravel()


def is_bad_point(X, camera1, camera2, Z_limit, max_cos_parallax):
    X = np.asarray(X)
    if not np.isfinite(X).all():
        return True
    if X[2] < -Z_limit or X[2] > Z_limit:
        return True # BAD! If the point is outside the allowed Z range.
    R1 = Rotation.from_rotvec(camera1[:3]).as_matrix()
    R2 = Rotation.from_rotvec(camera2[:3]).as_matrix()
    t1, t2 = camera1[3:6], camera2[3:6]
    p1 = R1 @ X + t1 # A 3D point w.r.t. the 1st camera coordinate
    p2 = R2 @ X + t2 # A 3D point w.r.t. the 2nd camera coordinate
    if p1[2] <= 0 or p2[2] <= 0:
        return True # BAD! If the point is beyond of one of 1st and 2nd cameras.
    v2 = R1 @ R2.T @ p2 # A 3D vector 'p2' w.r.t. the 1st camera coordinate
    cos_parallax = p1 @ v2 / np.linalg.norm(p1) / np.linalg.norm(v2)
    if cos_parallax > max_cos_parallax:
        return True # BAD! If the point has small parallax angle.
    return False


def bundle_adjust(cameras, Xs, img_keypoint, xs_visited, ba_loss_width, ba_num_iter):
    # Rebuild BA from all observations after points are added (no persistent point pointers).
    camera_indices, point_indices, xs = [], [], []
    for (cam_idx, x_idx), X_idx in xs_visited.items():
        camera_indices.append(cam_idx)
        point_indices.append(X_idx)
        xs.append(img_keypoint[cam_idx][x_idx].pt)
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    xs = np.array(xs)
    params = np.concatenate((cameras[:, :6].ravel(), np.array(Xs).ravel()))
    J = bundle_adjustment_sparsity(len(cameras), len(Xs), camera_indices, point_indices)
    # SciPy limits function evaluations; Ceres limits iterations.
    result = least_squares(reprojection_error, params, jac_sparsity=J, x_scale=1.0,
                           max_nfev=ba_num_iter if ba_num_iter > 0 else None, verbose=2,
                           args=(xs, cameras, camera_indices, point_indices, ba_loss_width))
    print(result.message)
    cameras[:, :6] = result.x[:len(cameras) * 6].reshape(-1, 6)
    Xs[:] = result.x[len(cameras) * 6:].reshape(-1, 3).tolist()


def main():
    input = '../data/relief/%02d.jpg'
    img_resize, f_init, cx_init, cy_init, Z_limit = 0.25, 500, -1, -1, 100
    max_cos_parallax = np.cos(10 * np.pi / 180)
    ba_loss_width = 9 # Negative 'ba_loss_width' makes BA not to use a loss function.
    min_inlier_num, ba_num_iter = 200, 200 # Negative 'ba_num_iter' uses the default value for BA minimization.
    show_match = False

    # Load images and extract features
    video = cv.VideoCapture(input)
    if not video.isOpened():
        return -1
    fdetector = cv.BRISK_create()
    img_keypoint, img_set, img_descriptor = [], [], []
    while True:
        _, image = video.read()
        if image is None:
            break
        if img_resize != 1:
            image = cv.resize(image, None, fx=img_resize, fy=img_resize)
        keypoint, descriptor = fdetector.detectAndCompute(image, None)
        img_set.append(image)
        img_keypoint.append(keypoint)
        img_descriptor.append(descriptor)
    video.release()
    if len(img_set) < 2:
        return -1
    if cx_init < 0:
        cx_init = img_set[0].shape[1] // 2
    if cy_init < 0:
        cy_init = img_set[0].shape[0] // 2

    # Match features and find good matches
    fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
    match_pair = []   # Good matches (image pairs)
    match_inlier = [] # Good matches (inlier feature matches)
    for i in range(len(img_set)):
        for j in range(i + 1, len(img_set)):
            if img_descriptor[i] is None or img_descriptor[j] is None:
                continue
            # Match features of two image pair (i, j) and find their inliers
            match = fmatcher.match(img_descriptor[i], img_descriptor[j])
            src = np.array([img_keypoint[i][m.queryIdx].pt for m in match])
            dst = np.array([img_keypoint[j][m.trainIdx].pt for m in match])
            if len(match) < 8:
                continue
            F, inlier_mask = cv.findFundamentalMat(src, dst, cv.FM_RANSAC)
            if F is None or inlier_mask is None:
                continue
            inlier = [m for m, keep in zip(match, inlier_mask.ravel()) if keep]
            print(f'3DV Tutorial: Image {i} - {j} are matched ({len(inlier)} / {len(match)}).')

            # Determine whether the image pair is good or not
            if len(inlier) < min_inlier_num:
                continue
            print(f'3DV Tutorial: Image {i} - {j} are selected.')
            match_pair.append((i, j))
            match_inlier.append(inlier)
            if show_match:
                match_image = cv.drawMatches(img_set[i], img_keypoint[i], img_set[j], img_keypoint[j], match, None, matchesMask=inlier_mask.ravel().tolist())
                cv.imshow('3DV Tutorial: Structure-from-Motion', match_image)
                cv.waitKey()
    if len(match_pair) < 1:
        return -1

    # Initialize cameras (rotation, translation, intrinsic parameters)
    cameras = np.tile([0., 0, 0, 0, 0, 0, f_init, cx_init, cy_init], (len(img_set), 1))
    best_pair = 0
    best_score = [len(inlier) for inlier in match_inlier]
    while True:
        # 1) Select the best pair
        best_pair = int(np.argmax(best_score))
        if best_score[best_pair] == 0:
            print("3DV Tutorial: There is no good match. Try again after reducing 'max_cos_parallax'.")
            return -1
        best_cam0, best_cam1 = match_pair[best_pair]

        # 2) Estimate relative pose from the best two views (epipolar geometry)
        src = np.array([img_keypoint[best_cam0][m.queryIdx].pt for m in match_inlier[best_pair]])
        dst = np.array([img_keypoint[best_cam1][m.trainIdx].pt for m in match_inlier[best_pair]])
        K = get_camera_mat(cameras[best_cam0])
        E, inlier_mask = cv.findEssentialMat(src, dst, K, cv.RANSAC, 0.999, 1)
        if E is None:
            best_score[best_pair] = 0
            continue
        inlier_num, R, t, inlier_mask = cv.recoverPose(E, src, dst, K, mask=inlier_mask)
        if inlier_num < 8:
            best_score[best_pair] = 0
            continue
        # Remove additionally detected outliers
        keep = inlier_mask.ravel() != 0
        src, dst = src[keep], dst[keep]
        match_inlier[best_pair] = [m for m, good in zip(match_inlier[best_pair], keep) if good]
        # Reset poses left over from a previously rejected pair.
        cameras[:, :6] = 0
        update_camera_pose(cameras[best_cam1], R, t)

        # 3) Reconstruct 3D points of the best two views (triangulation)
        P0, P1 = get_projection_mat(cameras[best_cam0]), get_projection_mat(cameras[best_cam1])
        best_Xs = cv.triangulatePoints(P0, P1, src.T, dst.T)
        best_Xs = (best_Xs[:3] / best_Xs[3]).T
        best_score[best_pair] = sum(not is_bad_point(p, cameras[best_cam0], cameras[best_cam1], Z_limit, max_cos_parallax) for p in best_Xs)
        print(f'3DV Tutorial: Image {best_cam0} - {best_cam1} were checked as the best match (# of inliers = {len(match_inlier[best_pair])}, # of good points = {best_score[best_pair]}).')
        if best_score[best_pair] > 100:
            break
        best_score[best_pair] = 0

    # Prepare the initial 3D points
    Xs, Xs_rgb = [], []
    xs_visited = {}
    for i, p in enumerate(best_Xs):
        if is_bad_point(p, cameras[best_cam0], cameras[best_cam1], Z_limit, max_cos_parallax):
            continue # Do not add if it is bad
        X_idx = len(Xs)
        x0_idx, x1_idx = match_inlier[best_pair][i].queryIdx, match_inlier[best_pair][i].trainIdx
        key0, key1 = (best_cam0, x0_idx), (best_cam1, x1_idx)
        if key0 in xs_visited or key1 in xs_visited:
            continue
        Xs.append(p.tolist())
        x, y = np.rint(img_keypoint[best_cam0][x0_idx].pt).astype(int)
        Xs_rgb.append(img_set[best_cam0][y, x])
        xs_visited[key0] = X_idx
        xs_visited[key1] = X_idx
    img_added, img_ignored = {best_cam0, best_cam1}, set()
    print(f'3DV Tutorial: Image {best_cam0} - {best_cam1} are complete (# of 3D points = {len(Xs)}).')

    # Optimize the initial reconstruction.
    bundle_adjust(cameras, Xs, img_keypoint, xs_visited, ba_loss_width, ba_num_iter)
    while True:
        # 4) Select the next image to add
        img_score = [0] * len(img_set)
        match_table = [[] for _ in img_set]
        for img in range(len(img_set)):
            if img in img_added or img in img_ignored:
                continue # When the image is already registered or ignored
            for i, (cam0, cam1) in enumerate(match_pair):
                if cam0 == img and cam1 in img_added:
                    img_score[img] += sum((cam1, m.trainIdx) in xs_visited for m in match_inlier[i])
                    match_table[img].append(i)
                elif cam1 == img and cam0 in img_added:
                    img_score[img] += sum((cam0, m.queryIdx) in xs_visited for m in match_inlier[i])
                    match_table[img].append(i)
        next_cam = int(np.argmax(img_score))
        next_match = match_table[next_cam]
        if img_score[next_cam] == 0 or not next_match:
            break

        # Separate points into known (pts_*) and unknown (new_*) for PnP (known) and triangulation (unknown)
        pts_3d, pts_2d, pts_key, pts_idx, new_pair_cam = [], [], [], [], []
        pts_seen, Xs_seen = set(), set()
        new_next_2d = [[] for _ in next_match]
        new_pair_2d = [[] for _ in next_match]
        new_next_key = [[] for _ in next_match]
        new_pair_key = [[] for _ in next_match]
        for i, m in enumerate(next_match):
            cam0, cam1 = match_pair[m]
            next_is_first = cam0 == next_cam
            pair_cam = cam1 if next_is_first else cam0
            new_pair_cam.append(pair_cam)
            for match in match_inlier[m]:
                next_idx, pair_idx = (match.queryIdx, match.trainIdx) if next_is_first else (match.trainIdx, match.queryIdx)
                key = (next_cam, next_idx)
                found = xs_visited.get((pair_cam, pair_idx))
                if found is not None: # When the matched point is already known (--> PnP)
                    # Use each image observation and each 3D point only once in PnP.
                    if key in pts_seen or found in Xs_seen:
                        continue
                    pts_seen.add(key)
                    Xs_seen.add(found)
                    pts_3d.append(Xs[found])
                    pts_2d.append(img_keypoint[next_cam][next_idx].pt)
                    pts_key.append(key)
                    pts_idx.append(found)
                else: # When the matched point is newly observed (--> triangulation)
                    new_next_2d[i].append(img_keypoint[next_cam][next_idx].pt)
                    new_pair_2d[i].append(img_keypoint[pair_cam][pair_idx].pt)
                    new_next_key[i].append(key)
                    new_pair_key[i].append((pair_cam, pair_idx))

        # 5) Estimate relative pose of the next view (PnP)
        if len(pts_3d) < 10:
            print(f'3DV Tutorial: Image {next_cam} is ignored (due to the small number of points).')
            img_ignored.add(next_cam)
            continue
        K = get_camera_mat(cameras[next_cam])
        success, rvec, t, inlier_idx = cv.solvePnPRansac(np.array(pts_3d), np.array(pts_2d), K, None, iterationsCount=100, reprojectionError=4.0, confidence=0.999)
        if not success or inlier_idx is None or len(inlier_idx) < 10:
            print(f'3DV Tutorial: Image {next_cam} is ignored (PnP failed).')
            img_ignored.add(next_cam)
            continue
        update_camera_pose(cameras[next_cam], rvec, t)
        for i in inlier_idx.ravel():
            xs_visited[pts_key[i]] = pts_idx[i]

        # 6) Reconstruct newly observed 3D points (triangulation)
        P0 = get_projection_mat(cameras[next_cam])
        for i, pair_cam in enumerate(new_pair_cam):
            if not new_next_2d[i]:
                continue
            P1 = get_projection_mat(cameras[pair_cam])
            new_Xs = cv.triangulatePoints(P0, P1, np.array(new_next_2d[i]).T, np.array(new_pair_2d[i]).T)
            new_Xs = (new_Xs[:3] / new_Xs[3]).T
            for j, p in enumerate(new_Xs):
                if is_bad_point(p, cameras[next_cam], cameras[pair_cam], Z_limit, max_cos_parallax):
                    continue # Do not add if it is bad
                # Never overwrite observations that already belong to a 3D point.
                if new_next_key[i][j] in xs_visited or new_pair_key[i][j] in xs_visited:
                    continue
                X_idx = len(Xs)
                Xs.append(p.tolist())
                x, y = np.rint(new_next_2d[i][j]).astype(int)
                Xs_rgb.append(img_set[next_cam][y, x])
                xs_visited[new_next_key[i][j]] = X_idx
                xs_visited[new_pair_key[i][j]] = X_idx
        print(f'3DV Tutorial: Image {next_cam} is complete (# of 3D points = {len(Xs)}).')

        # 7) Optimize camera pose and 3D points together (bundle adjustment)
        bundle_adjust(cameras, Xs, img_keypoint, xs_visited, ba_loss_width, ba_num_iter)
        img_added.add(next_cam)
    for j in sorted(img_added):
        camera = cameras[j]
        print(f"3DV Tutorial: Camera {j}'s (f, cx, cy) = ({camera[6]:.3f}, {camera[7]:.1f}, {camera[8]:.1f})")

    # Store the 3D points to an XYZ file
    Xs, Xs_rgb = np.array(Xs), np.array(Xs_rgb)
    keep = (Xs[:, 2] > -Z_limit) & (Xs[:, 2] < Z_limit)
    np.savetxt('sfm_inc(point).xyz', np.column_stack((Xs[keep], Xs_rgb[keep, ::-1])),
               fmt='%f %f %f %d %d %d') # Format: x, y, z, R, G, B

    # Store the camera poses to an XYZ file
    poses = []
    for j in sorted(img_added):
        rvec, t = cameras[j, :3], cameras[j, 3:6]
        R = Rotation.from_rotvec(rvec).as_matrix()
        p = -R.T @ t
        poses.append(np.concatenate((p, R.T[:, 2])))
    np.savetxt('sfm_inc(camera).xyz', poses, fmt='%f') # Format: x, y, z, n_x, n_y, n_z

    # Visualize the reconstructed 3D points and camera frustums
    K = get_camera_mat(cameras[best_cam0])
    plot_reconstruction(Xs[keep], cameras[sorted(img_added)], K, img_set[0].shape[1::-1], Xs_rgb[keep, ::-1], point_size=1)
    return 0


if __name__ == '__main__':
    main()
