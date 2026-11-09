#include "opencv2/opencv.hpp"

int main()
{
    // Load two images
    cv::Mat img1 = cv::imread("../data/hill01.jpg");
    cv::Mat img2 = cv::imread("../data/hill02.jpg");
    if (img1.empty() || img2.empty()) return -1;

    // Retrieve matching points
    cv::Ptr<cv::FeatureDetector> fdetector = cv::BRISK::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    fdetector->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
    fdetector->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);
    cv::Ptr<cv::DescriptorMatcher> fmatcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<cv::DMatch> match;
    fmatcher->match(descriptors1, descriptors2, match);

    // Calculate planar homography and merge two images
    std::vector<cv::Point2f> pts1, pts2;
    for (size_t i = 0; i < match.size(); i++)
    {
        pts1.push_back(keypoints1.at(match.at(i).queryIdx).pt);
        pts2.push_back(keypoints2.at(match.at(i).trainIdx).pt);
    }
    cv::Mat inlier_mask;
    cv::Mat H = cv::findHomography(pts2, pts1, inlier_mask, cv::RANSAC);

    cv::Mat img_merged;
    cv::warpPerspective(img2, img_merged, H, cv::Size(img1.cols * 2, img1.rows));
    img_merged.colRange(0, img1.cols) = img1 * 1; // Copy

    // Show the merged image
    cv::Mat original, img_matched;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, match, img_matched, cv::Scalar::all(-1), cv::Scalar::all(-1), inlier_mask); // Remove 'inlier_mask' if you want to show all putative matches
    cv::hconcat(img1, img2, original);
    cv::vconcat(original, img_matched, img_matched);
    cv::vconcat(img_matched, img_merged, img_merged);
    cv::imshow("Planar Image Stitching", img_merged);
    cv::waitKey(0);
    return 0;
}
