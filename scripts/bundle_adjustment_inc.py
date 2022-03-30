from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import open3d as o3d

input_num = 5
f, cx, cy = 1000, 320, 240

class BA():
    def __init__(self, f, cx, cy, n_cameras, n_points):
        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        self.n_cameras = n_cameras
        self.n_points = n_points

    def sparsity_jacobian_inc(self, camera_indices, point_indices, n_scene):
        m = camera_indices.size * 2 # m = 800 x 2
        n = 6 * (n_scene + 1) + self.n_points * 3 # n = 6 * cameras = 5 + 3 * points = 160 --> 510
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1

        for s in range(3):
            A[2 * i, n_scene * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_scene * 6 + point_indices * 3 + s] = 1

        return A

    def project(self, camera_params, points_3d):
        rvec, tvec = camera_params[0][:, :3], camera_params[0][:, 3:]  # tvec = 1x3 -> 160개로 늘려줘야 함.
        proj_points = Rotation.from_rotvec(rvec).apply(points_3d) # 160 x 3
        proj_points += tvec                                       
        proj_points = proj_points @ self.K.T
        proj_points /= proj_points[2, np.newaxis] # 애매하네....
        return proj_points[:, :2]

    @staticmethod
    def cost_func(params, points_2d, n_points, camera_indices, point_indices, n_scene, cls):
        """
        params = (camera_params + points_3d).ravel(). 
        n_cameras = cameras number, which is also index of scene
        cls is ba itself
        """
        points_2d = points_2d.reshape((n_points * (n_scene + 1) , 2))
        camera_params = params[: 6 * (n_scene + 1)].reshape((n_scene + 1, 6))
        points_3d = params[6 * (n_scene + 1): ].reshape((n_points, 3))
        proj_points = cls.project(camera_params[camera_indices], points_3d[point_indices])
        return (points_2d - proj_points).ravel()

def load_2d_points():
    global input_num
    """Load 2D points observed from multiple views"""
    xs = []
    for i in range(input_num):
        input = f"../bin/data/image_formation{i}.xyz"
        x = np.genfromtxt(input, delimiter=" ")
        x = x[:, :2]
        xs.append(x)

    return np.array(xs)


def main():
    global f, cx, cy
    parser = argparse.ArgumentParser(description="This is BA for incremental")
    VIEW_TEST = False
    VIEW_OPT = True

    points_2d = load_2d_points()
    n_camera = points_2d.shape[0]
    n_points = points_2d.shape[1]
    # Assumption
    # - All cameras have the same and known camera matrix
    # - All points are visible on all camera views

    # 1. Select best pair(skipped because all points are visible on all images)

    # 2. Estimate relative pose of the initial two views (epipolar geometry) 두 개에 대해서만 하네...
    F, _ = cv2.findFundamentalMat(points_2d[0], points_2d[1], cv2.FM_8POINT)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    E = K.T @ F @ K

    _, R, t, _ = cv2.recoverPose(E, points_2d[0], points_2d[1])
    rvec = Rotation.from_matrix(R).as_rotvec()
    cameras = np.zeros((n_camera, 6)) # rotation and translation
    cameras[1, :] = np.array([[rvec[0], rvec[1], rvec[2], t[0], t[1], t[2]]], dtype=object)

    # 3. Reconstruct 3D points of the initial two views (triangulation)
    P0 = K @ np.eye(3, 4, dtype=np.float32)
    Rt = np.hstack((R, t))
    P1 = K @ Rt
    X = cv2.triangulatePoints(P0, P1, points_2d[0].T, points_2d[1].T)
    X /= X[3]
    X = X.T

    points_3d = np.zeros((n_points, 3))
    for i in range(points_3d.shape[0]):
        points_3d[i, :] = X[i, :3]

    if VIEW_TEST:
        print("test")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Push constraints of two views
    ba = BA(f, cx, cy, n_camera, n_points)

    # 2개는 해줘야함.
    cam_indices = np.array([[]])
    for i in range(2):
        cam_indices = np.hstack((cam_indices, np.full((1, 160), i))) # 0x160, 1x160, ...

    point_indices = np.array([])
    for i in range(2):
        point_indices = np.hstack((point_indices, np.linspace(0, 159, 160, dtype=int))) # (0 ~ 159) * 5


    opts_points_3d = None
    # Incrementally add more views
    for j in range(2, n_camera):
        # 4. Select the next image to add (skipped because all points are visible on all images)
        
        # 5. Estimate relative pose of the next view (PnP)
        _, rvec, tvec = cv2.solvePnP(points_3d, points_2d[j], K, None)
        cameras[j, :, np.newaxis] = np.array([[rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2]]], dtype=object)
        # 6. Reconstruct newly observed 3D points (triangulation; skipped because all points are visible on all images)
        
        # 7. Optimize camera pose and 3D points together (BA)
        params = np.hstack((cameras[:j+1].ravel(), points_3d.ravel())) # j번째 이전까지 

        cam_indices = np.hstack((cam_indices, np.full((1, 160), j, dtype=int)))  # 
        point_indices = np.hstack((point_indices, np.linspace(0, 159, 160, dtype=int)))
        cam_indices = cam_indices.astype(int)
        point_indices = point_indices.astype(int)

        jac = ba.sparsity_jacobian_inc(camera_indices=cam_indices, point_indices=point_indices, n_scene=j)
        
        t = time.time()
        
        opt = least_squares(ba.cost_func, params, jac_sparsity=jac, method='trf', args=(points_2d[: j+1], n_points, cam_indices, point_indices, j, ba))
        
        print("calc time:", time.time() - t)
        
    # Store the 3D points to an XYZ file
    opt_cameras = opt.x[: 6 * n_camera].reshape((n_camera,6))
    opts_points_3d = opt.x[6 * n_camera :].reshape((n_points, 3))
    if VIEW_OPT:
        print("opt")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(opts_points_3d)

    # Store the Camera poses to to an XYZ file
    # Store the 3D points to an XYZ file
    point_file = "../bin/data/bundle_adjustment_inc(point)_by_cjh.xyz"
    with open(point_file, 'wt') as f:
        for i in range(n_points):
            data = f"{opts_points_3d[i, 0]} {opts_points_3d[i, 1]} {opts_points_3d[i, 2]}\n"
            f.write(data)

    camera_file = "../bin/data/bundle_adjustment_inc(camera)_by_cjh.xyz"
    with open(camera_file, 'wt') as f:
        for i in range(n_camera):
            data = f"{opt_cameras[i, 0]} {opt_cameras[i, 1]} {opt_cameras[i, 2]} {opt_cameras[i, 3]} {opt_cameras[i, 4]} {opt_cameras[i, 5]}\n"
            f.write(data)

    o3d.visualization.draw_geometries([pcd])
    

if __name__ == "__main__":
    main()