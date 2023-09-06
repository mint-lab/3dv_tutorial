#include "opencv2/opencv.hpp"

int main()
{
    // The given video and calibration data
    const char* video_file = "../data/chessboard.avi";
    cv::Matx33d K(432.7390364738057, 0, 476.0614994349778,
                  0, 431.2395555913084, 288.7602152621297,
                  0, 0, 1);
    std::vector<double> dist_coeff = { -0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194 };
    cv::Size board_pattern(10, 7);
    double board_cellsize = 0.025;

    // Open a video
    cv::VideoCapture video;
    if (!video.open(video_file)) return -1;

    // Prepare a 3D box for simple AR
    std::vector<cv::Point3d> box_lower = { cv::Point3d(4 * board_cellsize, 2 * board_cellsize, 0), cv::Point3d(5 * board_cellsize, 2 * board_cellsize, 0), cv::Point3d(5 * board_cellsize, 4 * board_cellsize, 0), cv::Point3d(4 * board_cellsize, 4 * board_cellsize, 0) };
    std::vector<cv::Point3d> box_upper = { cv::Point3d(4 * board_cellsize, 2 * board_cellsize, -board_cellsize), cv::Point3d(5 * board_cellsize, 2 * board_cellsize, -board_cellsize), cv::Point3d(5 * board_cellsize, 4 * board_cellsize, -board_cellsize), cv::Point3d(4 * board_cellsize, 4 * board_cellsize, -board_cellsize) };

    // Prepare 3D points on a chessboard
    std::vector<cv::Point3d> obj_points;
    for (int r = 0; r < board_pattern.height; r++)
        for (int c = 0; c < board_pattern.width; c++)
            obj_points.push_back(cv::Point3d(board_cellsize * c, board_cellsize * r, 0));

    // Run pose estimation
    while (true)
    {
        // Grab an image from the video
        cv::Mat img;
        video >> img;
        if (img.empty()) break;

        // Estimate the camera pose
        std::vector<cv::Point2d> img_points;
        bool success = cv::findChessboardCorners(img, board_pattern, img_points, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
        if (success)
        {
            cv::Mat rvec, tvec;
            cv::solvePnP(obj_points, img_points, K, dist_coeff, rvec, tvec);

            // Draw the box on the image
            cv::Mat line_lower, line_upper;
            cv::projectPoints(box_lower, rvec, tvec, K, dist_coeff, line_lower);
            cv::projectPoints(box_upper, rvec, tvec, K, dist_coeff, line_upper);
            line_lower.reshape(1).convertTo(line_lower, CV_32S); // Change 4 x 1 matrix (CV_64FC2) to 4 x 2 matrix (CV_32SC1)
            line_upper.reshape(1).convertTo(line_upper, CV_32S); // Because 'cv::polylines()' only accepts 'CV_32S' depth.
            cv::polylines(img, line_lower, true, cv::Vec3b(255, 0, 0), 2);
            for (int i = 0; i < line_lower.rows; i++)
                cv::line(img, cv::Point(line_lower.row(i)), cv::Point(line_upper.row(i)), cv::Vec3b(0, 255, 0), 2);
            cv::polylines(img, line_upper, true, cv::Vec3b(0, 0, 255), 2);

            // Print the camera position
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            cv::Mat p = -R.t() * tvec;
            cv::String info = cv::format("XYZ: [%.3f, %.3f, %.3f]", cv::Point3d(p));
            cv::putText(img, info, cv::Point(5, 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Vec3b(0, 255, 0));
        }

        // Show the image and process the key event
        cv::imshow("Pose Estimation (Chessboard)", img);
        int key = cv::waitKey(1);
        if (key == 32) key = cv::waitKey(); // Space
        if (key == 27) break;               // ESC
    }

    video.release();
    return 0;
}
