import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation

# The given camera configuration: Focal length, principal point, image resolution, position, and orientation
f, cx, cy, noise_std = 1000, 320, 240, 1
img_res = (640, 480)
cam_pos = [[0, 0, 0], [-2, -2, 0], [2, 2, 0], [-2, 2, 0], [2, -2, 0]]          # Unit: [m]
cam_ori = [[0, 0, 0], [-15 , 15, 0], [15, -15, 0], [15, 15, 0], [-15, -15, 0]] # Unit: [deg]

# Load a point cloud in the homogeneous coordinate
X = np.loadtxt('../data/box.xyz') # Size: N x 3

# Generate images for each camera pose
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
for i, (pos, ori) in enumerate(zip(cam_pos, cam_ori)):
    # Derive 'R' and 't'
    Rc = Rotation.from_euler('zyx', ori[::-1], degrees=True).as_matrix()
    R = Rc.T
    t = -Rc.T @ pos

    # Project the points (Alternative: `cv.projectPoints()`)
    x = K @ (R @ X.T + t.reshape(-1, 1)) # Size: 3 x N
    x /= x[-1]

    # Add Gaussian noise
    noise = np.random.normal(scale=noise_std, size=(2, len(X)))
    x[0:2,:] += noise

    # Show and save the points
    img = np.zeros(img_res[::-1], dtype=np.uint8)
    for c in range(x.shape[1]):
        cv.circle(img, x[0:2,c].astype(np.int32), 2, 255, -1)
    cv.imshow(f'Image Formation {i}', img)
    np.savetxt(f'image_formation{i}.xyz', x.T) # Size: N x 2

cv.waitKey()
cv.destroyAllWindows()
