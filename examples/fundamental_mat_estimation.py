import numpy as np
import cv2 as cv

# Load two images
img1 = cv.imread('../data/KITTI07/image_0/000000.png')
img2 = cv.imread('../data/KITTI07/image_0/000023.png')
assert (img1 is not None) and (img2 is not None), 'Cannot read the given images'
f, cx, cy = 707.0912, 601.8873, 183.1104 # From the KITTI dataset
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

# Retrieve matching points
fdetector = cv.BRISK_create()
keypoints1, descriptors1 = fdetector.detectAndCompute(img1, None)
keypoints2, descriptors2 = fdetector.detectAndCompute(img2, None)

fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
match = fmatcher.match(descriptors1, descriptors2)

# Calculate the fundamental matrix
pts1, pts2 = [], []
for i in range(len(match)):
    pts1.append(keypoints1[match[i].queryIdx].pt)
    pts2.append(keypoints2[match[i].trainIdx].pt)
pts1 = np.array(pts1, dtype=np.float32)
pts2 = np.array(pts2, dtype=np.float32)
F, inlier_mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 0.5, 0.999)
print(f'* F = {F}')
print(f'* The number of inliers = {sum(inlier_mask.ravel())}')

# Extract relative camera pose between two images
E = K.T @ F @ K
positive_num, R, t, positive_mask = cv.recoverPose(E, pts1, pts2, K, mask=inlier_mask)
print(f'* R = {R}')
print(f'* t = {t}')
print(f'* The position of Image #2 = {-R.T @ t}') # [-0.57, 0.09, 0.82]
print(f'* The number of positive-depth inliers = {sum(positive_mask.ravel())}')

# Show the matched images
img_matched = cv.drawMatches(img1, keypoints1, img2, keypoints2, match, None, None, None,
                             matchesMask=inlier_mask.ravel().tolist()) # Remove `matchesMask` if you want to show all putative matches
cv.namedWindow('Fundamental Matrix Estimation', cv.WINDOW_NORMAL)
cv.imshow('Fundamental Matrix Estimation', img_matched)
cv.waitKey(0)
cv.destroyAllWindows()
