import numpy as np
import cv2 as cv
import random
import matplotlib.pyplot as plt

def generate_line(pts):
    # Line model: y = ax + b
    a = (pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0])
    b = pts[0][1] - a * pts[0][0]

    # Line model: ax + by + c = 0 (a^2 + b^2 = 1)
    line = np.array([a, -1, b])
    return line / np.linalg.norm(line[:2])

def evaluate_line(line, p):
    a, b, c = line
    x, y = p
    return np.fabs(a*x + b*y + c)

def fit_line_ransac(data, n_sample, ransac_trial, ransac_threshold):
    best_score = -1
    best_model = None
    for _ in range(ransac_trial):
        # Step 1: Hypothesis generation
        sample = random.choices(data, k=n_sample)
        model = generate_line(sample)

        # Step 2: Hypothesis evaluation
        score = 0
        for p in data:
            error = evaluate_line(model, p)
            if error < ransac_threshold:
                score += 1
        if score > best_score:
            best_score = score
            best_model = model

    return best_model, best_score

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
            y = np.random.uniform(*y_range)        # Generate an outlier
        else:
            y = line2y(true_line, x)               # Generate an inlier
            x += np.random.normal(scale=noise_std) # Add Gaussian noise
            y += np.random.normal(scale=noise_std)
        data.append((x, y))
    data = np.array(data)

    # Estimate a line using RANSAC
    best_line, best_score = fit_line_ransac(data, 2, 100, 0.3) # log(1 - 0.999) / log(1 - 0.3^2) = 73

    # Estimate a line using OpenCV (for reference)
    # Note) OpenCV line model: n_y * (x - x_0) = n_x * (y - y_0)
    nnxy = cv.fitLine(data, cv.DIST_L2, 0, 0.01, 0.01).flatten()
    lsqr_line = np.array([nnxy[1], -nnxy[0], -nnxy[1]*nnxy[2] + nnxy[0]*nnxy[3]])

    # Plot the data and result
    plt.plot(data_range, line2y(true_line, data_range), 'r-', label='The true line')
    plt.plot(data[:,0], data[:,1], 'b.', label='Noisy data')
    plt.plot(data_range, line2y(best_line, data_range), 'g-', label=f'RASAC (score={best_score})')
    plt.plot(data_range, line2y(lsqr_line, data_range), 'm-', label='Least square method')
    plt.legend()
    plt.xlim(data_range)
    plt.show()
