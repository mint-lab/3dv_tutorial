#include "opencv2/opencv.hpp"

int main()
{
    // Open a video and get the reference image and feature points
    cv::VideoCapture video;
    if (!video.open("../data/traffic.avi")) return -1;

    cv::Mat gray_ref;
    video >> gray_ref;
    if (gray_ref.empty())
    {
        video.release();
        return -1;
    }
    if (gray_ref.channels() > 1) cv::cvtColor(gray_ref, gray_ref, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> pts_ref;
    cv::goodFeaturesToTrack(gray_ref, pts_ref, 2000, 0.01, 10);
    if (pts_ref.size() < 4)
    {
        video.release();
        return -1;
    }

    // Run video stabilization
    while (true)
    {
        // Read an image from `video`
        cv::Mat img, gray;
        video >> img;
        if (img.empty()) break;
        if (img.channels() > 1) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        else                      gray = img.clone();

        // Extract optical flow and calculate planar homography
        std::vector<cv::Point2f> pts;
        std::vector<uchar> status;
        cv::Mat error, inlier_mask;
        cv::calcOpticalFlowPyrLK(gray_ref, gray, pts_ref, pts, status, error);
        std::vector<cv::Point2f> ref, tracked_pts;
        for (size_t i = 0; i < status.size(); i++)
            if (status[i]) { tracked_pts.push_back(pts[i]); ref.push_back(pts_ref[i]); }
        pts = tracked_pts;
        cv::Mat H = cv::findHomography(pts, ref, inlier_mask, cv::RANSAC);

        // Synthesize a stabilized image
        cv::Mat warp;
        cv::warpPerspective(img, warp, H, cv::Size(img.cols, img.rows));

        // Show the original and stabilized images together
        for (int i = 0; i < pts.size(); i++)
        {
            cv::Vec3b color = cv::Vec3b(0, 127, 0);
            if (inlier_mask.at<uchar>(i) > 0) color = cv::Vec3b(0, 0, 255);
            cv::line(img, ref[i], pts[i], color);
        }
        cv::hconcat(img, warp, img);
        cv::imshow("2D Video Stabilization", img);
        int key = cv::waitKey(1);
        if (key == 32) key = cv::waitKey(); // Space
        if (key == 27) break;               // ESC
    }

    video.release();
    return 0;
}
