#include "opencv_all.hpp"

int main(void)
{
    bool use_5pt = false;
    double camera_focal = 718.8560;
    cv::Point2d camera_center(607.1928, 185.2157);

    // Open a file to write camera trajectory
    FILE* camera_traj = fopen("visual_odometry_epipolar.xyz", "wt");
    if (camera_traj == NULL) return -1;

    // Open an video and get the initial image
    cv::VideoCapture video;
    if (!video.open("data/KITTI_00_L/%06d.png")) return -1;

    cv::Mat gray_prev;
    video >> gray_prev;
    if (gray_prev.empty())
    {
        video.release();
        return -1;
    }
    if (gray_prev.channels() > 1) cv::cvtColor(gray_prev, gray_prev, CV_RGB2GRAY);

    // Run and record monocular visual odometry
    cv::Mat camera_pose = cv::Mat::eye(4, 4, CV_64F);
    while (true)
    {
        // Grab an image from the video
        cv::Mat image, gray;
        video >> image;
        if (image.empty()) break;
        if (image.channels() > 1) cv::cvtColor(image, gray, CV_RGB2GRAY);
        else                      gray = image.clone();

        // Extract optical flow
        std::vector<cv::Point2f> point_prev, point;
        cv::goodFeaturesToTrack(gray_prev, point_prev, 2000, 0.01, 10);
        std::vector<uchar> m_status;
        cv::Mat err;
        cv::calcOpticalFlowPyrLK(gray_prev, gray, point_prev, point, m_status, err);
        gray_prev = gray;

        // Calculate relative pose
        cv::Mat E, inlier_mask;
        if (use_5pt)
        {
            E = cv::findEssentialMat(point_prev, point, camera_focal, camera_center, cv::RANSAC, 0.99, 1, inlier_mask);
        }
        else
        {
            cv::Mat F = cv::findFundamentalMat(point_prev, point, cv::FM_RANSAC, 1, 0.99, inlier_mask);
            cv::Mat K = (cv::Mat_<double>(3, 3) << camera_focal, 0, camera_center.x, 0, camera_focal, camera_center.y, 0, 0, 1);
            E = K.t() * F * K;
        }
        cv::Mat R, t;
        int inlier_num = cv::recoverPose(E, point_prev, point, R, t, camera_focal, camera_center, inlier_mask);

        // Accumulate pose
        cv::Mat T = cv::Mat::eye(4, 4, R.type());
        T(cv::Range(0, 3), cv::Range(0, 3)) = R * 1.0;
        T(cv::Range(0, 3), cv::Range(3, 4)) = t * 1.0;
        camera_pose = camera_pose * T.inv();

        // Show the image and write camera pose 
        if (image.channels() < 3) cv::cvtColor(image, image, CV_GRAY2RGB);
        for (size_t i = 0; i < point_prev.size(); i++)
        {
            if (inlier_mask.at<uchar>(i) > 0) cv::line(image, point_prev[i], point[i], cv::Scalar(0, 0, 255));
            else cv::line(image, point_prev[i], point[i], cv::Scalar(0, 255, 0));
        }
        cv::imshow("3DV Tutorial: Visual Odometry (Epipolar)", image);
        fprintf(camera_traj, "%.6f %.6f %.6f\n", camera_pose.at<double>(0, 3), camera_pose.at<double>(1, 3), camera_pose.at<double>(2, 3));
        if (cv::waitKey(1) == 27) break; // 'ESC' key: Exit
    }

    video.release();
    fclose(camera_traj);
    return 0;
}
