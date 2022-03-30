import numpy as np
from scipy.optimize import minimize
from math import sqrt
import matplotlib.pyplot as plt
import cv2

def convert_line(line):
    return np.array([[line[0], -line[1], -line[0]*line[2]+line[1]*line[3]]], dtype=np.float32)

class GeometricError():
    def __init__(self):
        pass

    @staticmethod
    def func(datas, xn, yn):
        geo_error = np.sum((datas[0] * xn + datas[1] * yn + datas[2])**2 / (datas[0] ** 2 + datas[1] ** 2))
        return geo_error

def main():
    # Initial values
    truth = np.array([1./sqrt(2.), 1./sqrt(2.), -240.]) # The line model: a*x + b*y + c = 0 (a^2 + b^2 = 1)
    true_line = lambda x : -x + 240 * sqrt(2.)
    data_noise_std = 1
    data_num = 1000
    data_inlier_ratio = 1
    data_range = np.array([0, 640])

    # Generate Data
    data = []
    for i in range(data_num):
        if np.random.rand(1) < data_inlier_ratio:
            x = np.random.randint(0, 640)
            y = (truth[0] * x + truth[2]) / (-truth[1])
            x += np.random.normal(scale=data_noise_std)
            y += np.random.normal(scale=data_noise_std)
            data.append((x,y))
        else:
            data.append((np.random.randint(0, 640), np.random.randint(0, 480))) # outlier
    data = np.array(data)
    xn = data[:, 0].ravel()
    yn = data[:, 1].ravel()

    # Estimate line using scipy
    initial = np.array([1., 1., 0]) # 초기값 영향을 굉장히 많이 받음. 그래도 실패를 많이 함.
    geo_dist = GeometricError()

    opt_line = minimize(geo_dist.func, initial, args=(xn, yn))

    # Estimate a line using least squares method (for reference)
    nnxy = cv2.fitLine(data, cv2.DIST_L2, 0, 0.01, 0.01)
    lsm_line = convert_line(nnxy)

    # Display estimates
    lsm_line = lsm_line.tolist()
    print(f"* The Truth: {truth[0]:.3f} {truth[1]:.3f} {truth[2]:.3f}")
    print(f"* Estimate (SCIPY): {opt_line.x[0]:.3f} {opt_line.x[1]:.3f} {opt_line.x[2]:.3f}")
    print(f"* Estimate (LSM): {lsm_line[0][0][0]:.3f} {lsm_line[0][1][0]:.3f} {lsm_line[0][2][0]:.3f}")

    plt.plot(xn, yn, 'g.')
    plt.show()

if __name__ == "__main__":
    main()