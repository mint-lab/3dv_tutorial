#include "opencv_all.hpp"

int main(void)
{
    cv::Mat K = (cv::Mat_<double>(3, 3) << 432.7390364738057, 0, 476.0614994349778, 0, 431.2395555913084, 288.7602152621297, 0, 0, 1);
    cv::Mat dist_coeff = (cv::Mat_<double>(5, 1) << -0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194);
    cv::Size board_pattern(10, 7);
    double board_cellsize = 0.025;

    // Open an video
    cv::VideoCapture video;
    if (!video.open("data/chessboard.avi")) return -1;

    // Prepare four sticks for simple AR
    std::vector<cv::Point3f> stick_points;
    stick_points.push_back(cv::Point3f(0, 0, 0));
    stick_points.push_back(cv::Point3f(0, 0, -board_cellsize * 2));
    stick_points.push_back(cv::Point3f(board_cellsize * (board_pattern.width - 1), 0, 0));
    stick_points.push_back(cv::Point3f(board_cellsize * (board_pattern.width - 1), 0, -board_cellsize * 2));
    stick_points.push_back(cv::Point3f(0, board_cellsize * (board_pattern.height - 1), 0));
    stick_points.push_back(cv::Point3f(0, board_cellsize * (board_pattern.height - 1), -board_cellsize * 2));
    stick_points.push_back(cv::Point3f(board_cellsize * (board_pattern.width - 1), board_cellsize * (board_pattern.height - 1), 0));
    stick_points.push_back(cv::Point3f(board_cellsize * (board_pattern.width - 1), board_cellsize * (board_pattern.height - 1), -board_cellsize * 2));

    // Run pose estimation
    std::vector<cv::Point3f> object_points;
    for (int r = 0; r < board_pattern.height; r++)
        for (int c = 0; c < board_pattern.width; c++)
            object_points.push_back(cv::Point3f(board_cellsize * c, board_cellsize * r, 0));
     while (true)
    {
        // Grab an image from the video
        cv::Mat image;
        video >> image;
        if (image.empty()) break;

        // Esimate camera pose
        std::vector<cv::Point2f> image_points;
        bool complete = cv::findChessboardCorners(image, board_pattern, image_points, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
        if (complete)
        {
            cv::Mat rvec, tvec;
            cv::solvePnP(object_points, image_points, K, dist_coeff, rvec, tvec);

            // Draw four sticks
            std::vector<cv::Point2f> line_points;
            cv::projectPoints(stick_points, rvec, tvec, K, dist_coeff, line_points);
            for (int i = 0; i < 8; i += 2)
                cv::line(image, line_points[i], line_points[i + 1], cv::Scalar(0, 0, 255), 4);

            // Print camera position
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            cv::Mat p = -R.t() * tvec;
            cv::String info = cv::format("XYZ: [%.3f, %.3f, %.3f]", p.at<double>(0), p.at<double>(1), p.at<double>(2));
            cv::putText(image, info, cv::Point(5, 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
        }

        // Show the image
        cv::imshow("3DV Tutorial: Pose Estimation", image);
        int key = cv::waitKey(1);
        if (key == 27) break;                                   // 'ESC' key: Exit
        else if (key == 32)                                     // 'Space' key: Pause
        {
            key = cv::waitKey();
            if (key == 27) break;                               // 'ESC' key: Exit
        }
    }

    video.release();
    return 0;
}
