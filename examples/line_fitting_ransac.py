import numpy as np
import cv2
from math import fabs, sqrt

def convert_line(line):
    return np.array([[line[0], -line[1], -line[0]*line[2]+line[1]*line[3]]], dtype=np.float32)

def main():
    truth = np.array([1./sqrt(2.), 1./sqrt(2.), -240.]) # The line model: a*x + b*y + c = 0 (a^2 + b^2 = 1)
    ransac_trial, ransac_n_sample, ransac_thresh = 50, 2, 3.
    data_num, data_inlier_ratio, data_mean, data_inlier_noise = 1000, 0.5, 0, 1.0

    # Generate Data
    data = []
    for i in range(data_num):
        if np.random.randn(1) < data_inlier_ratio:
            x = np.random.randint(0, 480)
            y = (truth[0] * x + truth[2]) / (-truth[1])
            x += np.random.normal(data_mean, data_inlier_noise, 1)
            y += np.random.normal(data_mean, data_inlier_noise, 1)
            data.append((x,y))
        else:
            data.append((np.random.randint(0, 640), np.random.randint(0, 480)))

    data = np.array(data, dtype=np.float32)

    # Estimate a line using RANSAC
    best_score = -1
    best_line = []
    for i in range(ransac_trial):
        # Step 1: Hypothesis generation
        sample = []
        for j in range(ransac_n_sample):
            index = np.random.randint(0, int(data_num))
            sample.append(data[index])
        sample = np.array(sample, dtype=np.float32)

        nnxy = cv2.fitLine(sample, cv2.DIST_L2, 0, 0.01, 0.01)
        line = convert_line(nnxy)

        # Step 2: Hypothesis evailation
        score = 0
        for j in range(len(data)):
            error = fabs(line[0][0] * data[j][0] * line[0][1] * data[j][1] + line[0][2])
            if error<ransac_thresh: score += 1
        
        if score > best_score:
            best_score = score
            best_line = line
    
    # Estimate a line using squares method (for reference)
    nnxy = cv2.fitLine(data, cv2.DIST_L2, 0, 0.01, 0.01)
    lsm_line = convert_line(nnxy)

    # Display estimates
    best_line = best_line.tolist()
    lsm_line = lsm_line.tolist()
    print(f"* The Truth: {truth[0]:.3f} {truth[1]:.3f} {truth[2]:.3f}")
    print(f"* Estimate (RANSAC): {best_line[0][0][0]:.3f} {best_line[0][1][0]:.3f} {best_line[0][2][0]:.3f} (Score: {best_score})")
    print(f"* Estimate (LSM): {lsm_line[0][0][0]:.3f} {lsm_line[0][1][0]:.3f} {lsm_line[0][2][0]:.3f}")

if __name__ == "__main__":
    main()