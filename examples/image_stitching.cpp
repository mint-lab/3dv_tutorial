#include "opencv2/opencv.hpp"

int main()
{
    // Load two images
    cv::Mat image1 = cv::imread("../data/hill01.jpg");
    cv::Mat image2 = cv::imread("../data/hill02.jpg");
    if (image1.empty() || image2.empty()) return -1;

    // Retrieve matching points
    cv::Ptr<cv::FeatureDetector> fdetector = cv::BRISK::create();
    std::vector<cv::KeyPoint> keypoint1, keypoint2;
    cv::Mat descriptor1, descriptor2;
    fdetector->detectAndCompute(image1, cv::Mat(), keypoint1, descriptor1);
    fdetector->detectAndCompute(image2, cv::Mat(), keypoint2, descriptor2);
    cv::Ptr<cv::DescriptorMatcher> fmatcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<cv::DMatch> match;
    fmatcher->match(descriptor1, descriptor2, match);

    // Calculate planar homography and merge them
    std::vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < match.size(); i++)
    {
        points1.push_back(keypoint1.at(match.at(i).queryIdx).pt);
        points2.push_back(keypoint2.at(match.at(i).trainIdx).pt);
    }
    cv::Mat inlier_mask;
    cv::Mat H = cv::findHomography(points2, points1, inlier_mask, cv::RANSAC);

    cv::Mat merged;
    cv::warpPerspective(image2, merged, H, cv::Size(image1.cols * 2, image1.rows));
    merged.colRange(0, image1.cols) = image1 * 1; // Copy

    // Show the merged image
    cv::Mat original, matched;
    cv::drawMatches(image1, keypoint1, image2, keypoint2, match, matched, cv::Scalar::all(-1), cv::Scalar::all(-1), inlier_mask); // Remove 'inlier_mask' if you want to show all putative matches
    cv::hconcat(image1, image2, original);
    cv::vconcat(original, matched, matched);
    cv::vconcat(matched, merged, merged);
    cv::imshow("Planar Image Stitching", merged);
    cv::waitKey(0);
    return 0;
}
