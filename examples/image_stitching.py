import numpy as np
import cv2

def main():
    # Load two images
    input1, input2 = "../bin/data/hill01.jpg", "../bin/data/hill02.jpg"
    image1 = cv2.imread(input1)
    image2 = cv2.imread(input2)

    # Retrieve matching points
    brisk = cv2.BRISK_create()
    keypoints1, descriptors1 = brisk.detectAndCompute(image1, None)
    keypoints2, descriptors2 = brisk.detectAndCompute(image2, None)

    fmatcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
    match = fmatcher.match(descriptors1, descriptors2)

    # Calculate planar homography and merge them
    points1, points2 = [], []
    for i in range(len(match)):
        points1.append(keypoints1[match[i].queryIdx].pt)
        points2.append(keypoints2[match[i].trainIdx].pt)
    
    points1 = np.array(points1, dtype=np.float32)
    points2 = np.array(points2, dtype=np.float32)

    H, inlier_mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    merged = cv2.warpPerspective(image2, H, (image1.shape[1]*2, image1.shape[0]))
    merged[:,:image1.shape[1]] = image1
    merged[:,0:image1.shape[1]] = image1 # copy
    cv2.imshow("3DV Tutorial: Image Stitching", merged)

    # show the merged image
    matched = cv2.drawMatches(img1=image1, 
                                keypoints1=keypoints1, 
                                img2=image2, 
                                keypoints2=keypoints2, 
                                matches1to2=match[:15], 
                                outImg=None)
    
    original = np.hstack((image1, image2))
    matched = np.vstack((original, matched))
    merged = np.vstack((matched, merged))
    
    cv2.imshow("3DV Tutorial: Image Stitching", merged)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()