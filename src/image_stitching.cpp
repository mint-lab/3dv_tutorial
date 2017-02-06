#include "opencv_all.hpp"

int main(void)
{
    // Load two images (c.f. Assume that two images have the same size and type)
    cv::Mat image1 = cv::imread("data/hill01.jpg");
    cv::Mat image2 = cv::imread("data/hill02.jpg");
    if (image1.empty() || image2.empty()) return -1;

    // Retrieve matching points
    cv::Mat gray1, gray2;
    if (image1.channels() > 1)
    {
        cv::cvtColor(image1, gray1, CV_RGB2GRAY);
        cv::cvtColor(image2, gray2, CV_RGB2GRAY);
    }
    else
    {
        gray1 = image1.clone();
        gray2 = image2.clone();
    }
    cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SURF::create(); // SURF features
    std::vector<cv::KeyPoint> keypoint1, keypoint2;
    detector->detect(gray1, keypoint1);
    detector->detect(gray2, keypoint2);
    cv::Ptr<cv::FeatureDetector> extractor = cv::xfeatures2d::SURF::create(); // SURF descriptors
    cv::Mat descriptor1, descriptor2;
    extractor->compute(gray1, keypoint1, descriptor1);
    extractor->compute(gray2, keypoint2, descriptor2);
    cv::FlannBasedMatcher matcher; // Approximate Nearest Neighbors (ANN) matcher
    std::vector<cv::DMatch> match;
    matcher.match(descriptor1, descriptor2, match);

    // Calculate planar homography and merge them
    std::vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < match.size(); i++)
    {
        points1.push_back(keypoint1.at(match.at(i).queryIdx).pt);
        points2.push_back(keypoint2.at(match.at(i).trainIdx).pt);
    }
    cv::Mat H = cv::findHomography(points2, points1, cv::RANSAC);
    cv::Mat merge;
    cv::warpPerspective(image2, merge, H, cv::Size(image1.cols * 2, image1.rows));
    merge.colRange(0, image1.cols) = image1 * 1; // Copy

    // Show the merged image
    cv::imshow("3DV Tutorial: Image Stitching", merge);
    printf("Press any key to terminate tihs program!\n");
    cv::waitKey(0);
    return 0;
}