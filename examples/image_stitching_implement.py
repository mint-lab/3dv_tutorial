import numpy as np
import cv2 as cv
import random
from homography_estimation_implement import getPerspectiveTransform
from image_warping_implement import warpPerspective2

def evaluate_homography(H, p, q):
    p2q = H @ np.array([[p[0]], [p[1]], [1]])
    p2q /= p2q[-1]
    return np.linalg.norm(p2q[:2].flatten() - q)

def findHomography(src, dst, n_sample, ransac_trial, ransac_threshold):
    best_score = -1
    best_model = None
    for _ in range(ransac_trial):
        # Step 1: Hypothesis generation
        sample_idx = random.choices(range(len(src)), k=n_sample)
        model = getPerspectiveTransform(src[sample_idx], dst[sample_idx])

        # Step 2: Hypothesis evaluation
        score = 0
        for (p, q) in zip(src, dst):
            error = evaluate_homography(model, p, q)
            if error < ransac_threshold:
                score += 1
        if score > best_score:
            best_score = score
            best_model = model

    # Generate the best inlier mask
    best_inlier_mask = np.zeros(len(src), dtype=np.uint8)
    for idx, (p, q) in enumerate(zip(src, dst)):
        error = evaluate_homography(best_model, p, q)
        if error < ransac_threshold:
            best_inlier_mask[idx] = 1

    return best_model, best_inlier_mask

if __name__ == '__main__':
    # Load two images
    img1 = cv.imread('../data/hill01.jpg')
    img2 = cv.imread('../data/hill02.jpg')
    assert (img1 is not None) and (img2 is not None), 'Cannot read the given images'

    # Retrieve matching points
    fdetector = cv.BRISK_create()
    keypoints1, descriptors1 = fdetector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = fdetector.detectAndCompute(img2, None)

    fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
    match = fmatcher.match(descriptors1, descriptors2)

    # Calculate planar homography and merge them
    pts1, pts2 = [], []
    for i in range(len(match)):
        pts1.append(keypoints1[match[i].queryIdx].pt)
        pts2.append(keypoints2[match[i].trainIdx].pt)
    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)

    H, inlier_mask = findHomography(pts2, pts1, 4, 1000, 2) # log(1 - 0.999) / log(1 - 0.3^4) = 849
    img_merged = warpPerspective2(img2, H, (img1.shape[1]*2, img1.shape[0]))
    img_merged[:,:img1.shape[1]] = img1 # Copy

    # Show the merged image
    img_matched = cv.drawMatches(img1, keypoints1, img2, keypoints2, match, None, None, None,
                                 matchesMask=inlier_mask) # Remove `matchesMask` if you want to show all putative matches
    merge = np.vstack((np.hstack((img1, img2)), img_matched, img_merged))
    cv.imshow(f'Planar Image Stitching with My RANSAC (score={sum(inlier_mask)})', merge)
    cv.waitKey(0)
    cv.destroyAllWindows()
