#include "opencv2/opencv.hpp"

int main()
{
    const char* video_file = "../data/KITTI07/image_0/%06d.png";
    double f = 707.0912;
    cv::Point2d c(601.8873, 183.1104);
    bool use_5pt = true;
    int min_inlier_num = 100;
    double min_inlier_ratio = 0.2;
    const char* traj_file = "vo_epipolar.xyz";

    // Open a video and get an initial image
    cv::VideoCapture video;
    if (!video.open(video_file)) return -1;

    cv::Mat gray_prev;
    video >> gray_prev;
    if (gray_prev.empty())
    {
        video.release();
        return -1;
    }
    if (gray_prev.channels() > 1) cv::cvtColor(gray_prev, gray_prev, cv::COLOR_BGR2GRAY);

    // Run the monocular visual odometry
    cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, c.x, 0, f, c.y, 0, 0, 1);
    cv::Mat camera_pose = cv::Mat::eye(4, 4, CV_64F);
    FILE* camera_traj = fopen(traj_file, "wt");
    if (camera_traj == NULL) return -1;
    fprintf(camera_traj, "0.000000 0.000000 0.000000\n");
    while (true)
    {
        // Grab an image from the video
        cv::Mat img, gray;
        video >> img;
        if (img.empty()) break;
        if (img.channels() > 1) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        else                      gray = img.clone();

        // Extract optical flow
        std::vector<cv::Point2f> pts_prev, pts;
        cv::goodFeaturesToTrack(gray_prev, pts_prev, 2000, 0.01, 10);
        std::vector<uchar> status;
        cv::Mat error;
        cv::calcOpticalFlowPyrLK(gray_prev, gray, pts_prev, pts, status, error);
        size_t n = 0;
        for (size_t i = 0; i < status.size(); i++)
            if (status[i]) { pts_prev[n] = pts_prev[i]; pts[n++] = pts[i]; }
        pts_prev.resize(n);
        pts.resize(n);
        gray_prev = gray;

        // Calculate relative pose
        cv::Mat E, inlier_mask;
        if (use_5pt)
        {
            E = cv::findEssentialMat(pts_prev, pts, f, c, cv::RANSAC, 0.99, 1, inlier_mask);
        }
        else
        {
            cv::Mat F = cv::findFundamentalMat(pts_prev, pts, cv::FM_RANSAC, 1, 0.99, inlier_mask);
            E = K.t() * F * K;
        }
        cv::Mat R, t;
        int inlier_num = cv::recoverPose(E, pts_prev, pts, R, t, f, c, inlier_mask);
        double inlier_ratio = static_cast<double>(inlier_num) / pts.size();

        // Accumulate relative pose if result is reliable
        cv::Vec3b info_color(0, 255, 0);
        if ((inlier_num > min_inlier_num) && (inlier_ratio > min_inlier_ratio))
        {
            cv::Mat T = cv::Mat::eye(4, 4, R.type());
            T(cv::Rect(0, 0, 3, 3)) = R * 1.0;
            T.col(3).rowRange(0, 3) = t * 1.0;
            camera_pose = camera_pose * T.inv();
            info_color = cv::Vec3b(0, 0, 255);
        }

        // Show the image and write camera pose 
        if (img.channels() < 3) cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
        for (int i = 0; i < pts_prev.size(); i++)
        {
            if (inlier_mask.at<uchar>(i) > 0) cv::line(img, pts_prev[i], pts[i], cv::Vec3b(0, 0, 255));
            else cv::line(img, pts_prev[i], pts[i], cv::Vec3b(0, 127, 0));
        }
        double x = camera_pose.at<double>(0, 3), y = camera_pose.at<double>(1, 3), z = camera_pose.at<double>(2, 3);
        cv::String info = cv::format("Inliers: %d (%.0f%%), XYZ: [%.3f %.3f %.3f]", inlier_num, inlier_ratio*100, x, y, z);
        cv::putText(img, info, cv::Point(5, 15), cv::FONT_HERSHEY_PLAIN, 1, info_color);
        cv::imshow("Monocular Visual Odometry (Epipolar)", img);
        fprintf(camera_traj, "%.6f %.6f %.6f\n", x, y, z);
        int key = cv::waitKey(1);
        if (key == 32) key = cv::waitKey(); // Space
        if (key == 27) break;               // ESC
    }

    fclose(camera_traj);
    video.release();
    return 0;
}
