#include "opencv2/opencv.hpp"

int main()
{
    const char* input = "data/07/image_0/%06d.png";
    double f = 707.0912;
    cv::Point2d c(601.8873, 183.1104);
    bool use_5pt = true;
    int min_inlier_num = 100;

    // Open a file to write camera trajectory
    FILE* camera_traj = fopen("vo_epipolar.xyz", "wt");
    if (camera_traj == NULL) return -1;

    // Open a video and get the initial image
    cv::VideoCapture video;
    if (!video.open(input)) return -1;

    cv::Mat gray_prev;
    video >> gray_prev;
    if (gray_prev.empty())
    {
        video.release();
        return -1;
    }
    if (gray_prev.channels() > 1) cv::cvtColor(gray_prev, gray_prev, cv::COLOR_RGB2GRAY);

    // Run and record monocular visual odometry
    cv::Mat camera_pose = cv::Mat::eye(4, 4, CV_64F);
    while (true)
    {
        // Grab an image from the video
        cv::Mat image, gray;
        video >> image;
        if (image.empty()) break;
        if (image.channels() > 1) cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
        else                      gray = image.clone();

        // Extract optical flow
        std::vector<cv::Point2f> point_prev, point;
        cv::goodFeaturesToTrack(gray_prev, point_prev, 2000, 0.01, 10);
        std::vector<uchar> status;
        cv::Mat err;
        cv::calcOpticalFlowPyrLK(gray_prev, gray, point_prev, point, status, err);
        gray_prev = gray;

        // Calculate relative pose
        cv::Mat E, inlier_mask;
        if (use_5pt)
        {
            E = cv::findEssentialMat(point_prev, point, f, c, cv::RANSAC, 0.99, 1, inlier_mask);
        }
        else
        {
            cv::Mat F = cv::findFundamentalMat(point_prev, point, cv::FM_RANSAC, 1, 0.99, inlier_mask);
            cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, c.x, 0, f, c.y, 0, 0, 1);
            E = K.t() * F * K;
        }
        cv::Mat R, t;
        int inlier_num = cv::recoverPose(E, point_prev, point, R, t, f, c, inlier_mask);

        // Accumulate relative pose if result is reliable
        if (inlier_num > min_inlier_num)
        {
            cv::Mat T = cv::Mat::eye(4, 4, R.type());
            T(cv::Rect(0, 0, 3, 3)) = R * 1.0;
            T.col(3).rowRange(0, 3) = t * 1.0;
            camera_pose = camera_pose * T.inv();
        }

        // Show the image and write camera pose 
        if (image.channels() < 3) cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
        for (int i = 0; i < point_prev.size(); i++)
        {
            if (inlier_mask.at<uchar>(i) > 0) cv::line(image, point_prev[i], point[i], cv::Vec3b(0, 0, 255));
            else cv::line(image, point_prev[i], point[i], cv::Vec3b(0, 127, 0));
        }
        cv::String info = cv::format("Inliers: %d (%d%%),  XYZ: [%.3f, %.3f, %.3f]", inlier_num, 100 * inlier_num / point.size(), camera_pose.at<double>(0, 3), camera_pose.at<double>(1, 3), camera_pose.at<double>(2, 3));
        cv::putText(image, info, cv::Point(5, 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Vec3b(0, 255, 0));
        cv::imshow("3DV Tutorial: Visual Odometry (Epipolar)", image);
        fprintf(camera_traj, "%.6f %.6f %.6f\n", camera_pose.at<double>(0, 3), camera_pose.at<double>(1, 3), camera_pose.at<double>(2, 3));
        if (cv::waitKey(1) == 27) break; // 'ESC' key: Exit
    }

    video.release();
    fclose(camera_traj);
    return 0;
}
