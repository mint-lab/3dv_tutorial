import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from bundle_adjustment import plot_reconstruction


# Reprojection error for bundle adjustment with fixed intrinsic parameters.
def reprojection_error(params, xs, cameras, camera_indices, point_indices, ba_loss_width=-1):
    cameras = cameras.copy()
    cameras[:, :6] = params[:len(cameras) * 6].reshape(-1, 6)
    Xs = params[len(cameras) * 6:].reshape(-1, 3)
    camera = cameras[camera_indices]

    # X' = R*X + t
    R = Rotation.from_rotvec(camera[:, :3])
    X = R.apply(Xs[point_indices]) + camera[:, 3:6]

    # x' = K*X'
    x_p = camera[:, 6:7] * X[:, :2] / X[:, 2:] + camera[:, 7:9]

    # residual = x - x'
    residuals = xs - x_p
    if ba_loss_width > 0:
        # Match Ceres CauchyLoss on each 2D residual block, rather than each coordinate.
        s = np.sum(residuals**2, axis=1) / ba_loss_width**2
        weight = np.divide(np.log1p(s), s, out=np.ones_like(s), where=s > 0)
        residuals *= np.sqrt(weight[:, None])
    return residuals.ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    # Each observation depends on one camera (6 parameters) and one point (3 parameters).
    i = np.arange(len(camera_indices))
    J = lil_matrix((2 * len(i), 6 * n_cameras + 3 * n_points), dtype=int)
    for s in range(6):
        J[2 * i, 6 * camera_indices + s] = 1
        J[2 * i + 1, 6 * camera_indices + s] = 1
    for s in range(3):
        J[2 * i, 6 * n_cameras + 3 * point_indices + s] = 1
        J[2 * i + 1, 6 * n_cameras + 3 * point_indices + s] = 1
    return J.tocsr()


# Return a boolean mask identifying points behind a camera or with a large reprojection error.
def find_outlier_points(Xs, xs, cameras, visibility, reproj_error):
    is_noisy = np.zeros(len(Xs), dtype=bool)
    for (cam_idx, x_idx), X_idx in visibility.items():
        X = Xs[X_idx]
        x = np.array(xs[cam_idx][x_idx].pt)
        camera = cameras[cam_idx]

        # X' = R*X + t
        rvec, t = camera[:3], camera[3:6]
        R = Rotation.from_rotvec(rvec).as_matrix()
        X_p = R @ X + t
        if not np.isfinite(X_p).all() or X_p[2] <= 0:
            is_noisy[X_idx] = True
            continue

        # x' = K*X'
        f, cx, cy = camera[6:9]
        x_p = f * X_p[:2] / X_p[2] + [cx, cy]

        # Reject the point if any observation exceeds the threshold (pixels).
        d = x - x_p
        if reproj_error > 0 and d @ d > reproj_error**2:
            is_noisy[X_idx] = True
    return is_noisy


def main():
    input = '../data/relief/%02d.jpg'
    img_resize, f_init, cx_init, cy_init, Z_init, Z_limit = 0.25, 500, -1, -1, 2, 100
    ba_loss_width = 9 # Negative 'ba_loss_width' makes BA not to use a loss function.
    reproj_error = 3 # Maximum reprojection error [pixels]
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

    # 1) Initialize cameras (rotation, translation, intrinsic parameters)
    cameras = np.tile([0., 0, 0, 0, 0, 0, f_init, cx_init, cy_init], (len(img_set), 1))

    # 2) Initialize 3D points and build a visibility graph
    Xs, Xs_rgb = [], []
    xs_visited = {}
    Xs_cameras = []
    for m in range(len(match_pair)):
        for inlier in match_inlier[m]:
            cam1_idx, cam2_idx = match_pair[m]
            x1_idx, x2_idx = inlier.queryIdx, inlier.trainIdx
            key1, key2 = (cam1_idx, x1_idx), (cam2_idx, x2_idx)
            visit1, visit2 = xs_visited.get(key1), xs_visited.get(key2)
            if visit1 is not None and visit2 is not None:
                # Keep existing tracks and skip conflicting matches.
                continue # Skip if two observations are already visited

            if visit1 is not None:
                X_idx = visit1
            elif visit2 is not None:
                X_idx = visit2
            else:
                # Add a new point if two observations are not visited
                X_idx = len(Xs)
                Xs.append([0, 0, Z_init])
                Xs_cameras.append(set())
                x, y = np.rint(img_keypoint[cam1_idx][x1_idx].pt).astype(int)
                Xs_rgb.append(img_set[cam1_idx][y, x])
            # A track can contain only one observation from each camera.
            if visit1 is None and cam1_idx in Xs_cameras[X_idx]:
                continue
            if visit2 is None and cam2_idx in Xs_cameras[X_idx]:
                continue
            Xs_cameras[X_idx].update((cam1_idx, cam2_idx))
            xs_visited[key1] = X_idx
            xs_visited[key2] = X_idx
    print(f'3DV Tutorial: # of 3D points: {len(Xs)}')
    Xs = np.array(Xs, dtype=float)
    Xs_rgb = np.array(Xs_rgb)

    # 3) Optimize camera pose and 3D points together (bundle adjustment)
    # Intrinsic parameters remain fixed (6 DOF cameras).
    camera_indices, point_indices, xs = [], [], []
    for (cam_idx, x_idx), X_idx in xs_visited.items():
        camera_indices.append(cam_idx)
        point_indices.append(X_idx)
        xs.append(img_keypoint[cam_idx][x_idx].pt)
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    xs = np.array(xs)
    params = np.concatenate((cameras[:, :6].ravel(), Xs.ravel()))
    J = bundle_adjustment_sparsity(len(cameras), len(Xs), camera_indices, point_indices)
    # SciPy limits function evaluations; Ceres limits iterations.
    result = least_squares(reprojection_error, params, jac_sparsity=J, x_scale=1.0,
                           max_nfev=ba_num_iter if ba_num_iter > 0 else None, verbose=2,
                           args=(xs, cameras, camera_indices, point_indices, ba_loss_width))
    print(result.message)
    cameras[:, :6] = result.x[:len(cameras) * 6].reshape(-1, 6)
    Xs = result.x[len(cameras) * 6:].reshape(-1, 3)

    # Find outlier points to exclude from the output
    is_noisy = find_outlier_points(Xs, img_keypoint, cameras, xs_visited, reproj_error)
    print(f'3DV Tutorial: # of 3D points: {len(Xs)} (Rejected: {np.count_nonzero(is_noisy)})')
    for j, camera in enumerate(cameras):
        print(f"3DV Tutorial: Camera {j}'s (f, cx, cy) = ({camera[6]:.3f}, {camera[7]:.1f}, {camera[8]:.1f})")

    # Store the 3D points to an XYZ file
    keep = (Xs[:, 2] > -Z_limit) & (Xs[:, 2] < Z_limit) & ~is_noisy
    np.savetxt('sfm_global(point).xyz', np.column_stack((Xs[keep], Xs_rgb[keep, ::-1])), fmt='%f %f %f %d %d %d') # Format: x, y, z, R, G, B

    # Store the camera poses to an XYZ file
    poses = []
    for camera in cameras:
        rvec, t = camera[:3], camera[3:6]
        R = Rotation.from_rotvec(rvec).as_matrix()
        p = -R.T @ t
        poses.append(np.concatenate((p, R.T[:, 2])))
    np.savetxt('sfm_global(camera).xyz', poses, fmt='%f') # Format: x, y, z, n_x, n_y, n_z
    # Visualize the reconstructed 3D points and camera frustums
    K = np.array([[f_init, 0, cx_init], [0, f_init, cy_init], [0, 0, 1]])
    plot_reconstruction(Xs[keep], cameras, K, img_set[0].shape[1::-1], Xs_rgb[keep, ::-1], point_size=1)
    return 0


if __name__ == '__main__':
    main()
