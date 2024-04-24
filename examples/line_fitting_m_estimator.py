import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def geometric_error(line, pts):
    a, b, c = line
    err = [(a*x + b*y + c) / np.sqrt(a*a + b*b) for (x, y) in pts]
    return err

if __name__ == '__main__':
    true_line = np.array([2, 3, -14]) / np.sqrt(2*2 + 3*3) # The line model: a*x + b*y + c = 0 (a^2 + b^2 = 1)
    data_range = np.array([-4, 12])
    data_num = 100
    noise_std = 0.2
    outlier_ratio = 0.7

    # Generate noisy points with outliers
    line2y = lambda line, x: (line[0] * x + line[2]) / -line[1] # ax + by + c = 0 -> y = (ax + c) / -b
    y_range = sorted(line2y(true_line, data_range))
    data = []
    for _ in range(data_num):
        x = np.random.uniform(*data_range)
        if np.random.rand() < outlier_ratio:
            y = np.random.uniform(*y_range)
        else:
            y = line2y(true_line, x)
            x += np.random.normal(scale=noise_std)
            y += np.random.normal(scale=noise_std)
        data.append((x, y))
    data = np.array(data)

    # Estimate a line using least squares with a robust kernel
    init_line = [1, 1, 0]
    result = least_squares(geometric_error, init_line, args=(data,), loss='huber', f_scale=0.3)
    mest_line = result['x'] / np.linalg.norm(result['x'][:2])

    # Estimate a line using OpenCV (for reference)
    # Note) OpenCV line model: n_y * (x - x_0) = n_x * (y - y_0)
    nnxy = cv.fitLine(data, cv.DIST_L2, 0, 0.01, 0.01).flatten()
    lsqr_line = np.array([nnxy[1], -nnxy[0], -nnxy[1]*nnxy[2] + nnxy[0]*nnxy[3]])
    nnxy = cv.fitLine(data, cv.DIST_HUBER, 0, 0.01, 0.01).flatten()
    huber_line = np.array([nnxy[1], -nnxy[0], -nnxy[1]*nnxy[2] + nnxy[0]*nnxy[3]])

    # Plot the data and result
    plt.plot(data_range, line2y(true_line, data_range), 'r-', label='The true line')
    plt.plot(data[:,0], data[:,1], 'b.', label='Noisy data')
    plt.plot(data_range, line2y(mest_line, data_range), 'g-', label='M-estimator (Huber loss)')
    plt.plot(data_range, line2y(lsqr_line, data_range), 'm-', label='OpenCV (L2 loss)')
    plt.plot(data_range, line2y(huber_line, data_range), 'm:', label='OpenCV (Huber loss)')
    plt.legend()
    plt.xlim(data_range)
    plt.show()