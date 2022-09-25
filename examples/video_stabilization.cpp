#include "opencv2/opencv.hpp"

int main()
{
    // Open a video and get the reference image and feature points
    cv::VideoCapture video;
    if (!video.open("data/traffic.avi")) return -1;

    cv::Mat gray_ref;
    video >> gray_ref;
    if (gray_ref.empty())
    {
        video.release();
        return -1;
    }
    if (gray_ref.channels() > 1) cv::cvtColor(gray_ref, gray_ref, cv::COLOR_RGB2GRAY);

    std::vector<cv::Point2f> point_ref;
    cv::goodFeaturesToTrack(gray_ref, point_ref, 2000, 0.01, 10);
    if (point_ref.size() < 4)
    {
        video.release();
        return -1;
    }

    // Run and show video stabilization
    while (true)
    {
        // Grab an image from the video
        cv::Mat image, gray;
        video >> image;
        if (image.empty()) break;
        if (image.channels() > 1) cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
        else                      gray = image.clone();

        // Extract optical flow and calculate planar homography
        std::vector<cv::Point2f> point;
        std::vector<uchar> status;
        cv::Mat err, inlier_mask;
        cv::calcOpticalFlowPyrLK(gray_ref, gray, point_ref, point, status, err);
        cv::Mat H = cv::findHomography(point, point_ref, inlier_mask, cv::RANSAC);

        // Synthesize a stabilized image
        cv::Mat warp;
        cv::warpPerspective(image, warp, H, cv::Size(image.cols, image.rows));

        // Show the original and rectified images together
        for (int i = 0; i < point_ref.size(); i++)
        {
            if (inlier_mask.at<uchar>(i) > 0) cv::line(image, point_ref[i], point[i], cv::Vec3b(0, 0, 255));
            else cv::line(image, point_ref[i], point[i], cv::Vec3b(0, 127, 0));
        }
        cv::hconcat(image, warp, image);
        cv::imshow("3DV Tutorial: Video Stabilization", image);
        if (cv::waitKey(1) == 27) break; // 'ESC' key: Exit
    }

    video.release();
    return 0;
}
