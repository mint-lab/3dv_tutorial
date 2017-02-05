#include "opencv_all.hpp"

int main(void)
{
    cv::Size board_pattern(10, 7);
    double board_cellsize = 0.025;

    // Open an video
    cv::VideoCapture video;
    if (!video.open("data/chessboard.avi")) return -1;

    // Select images and find 2D corner points from them
    cv::Size image_size;
    std::vector<std::vector<cv::Point2f> > image_points;
    while (true)
    {
        // Grab an image from the video
        cv::Mat image;
        video >> image;
        if (image.empty()) break;
        image_size = image.size();

        // Show the image and keep it if necessary
        cv::imshow("3DV Tutorial: Camera Calibration", image);
        int key = cv::waitKey(1);
        if (key == 27) break;                   // 'ESC' key
        else if (key == 32)                     // 'Space' key
        {
            std::vector<cv::Point2f> pts;
            bool complete = cv::findChessboardCorners(image, board_pattern, pts);
            cv::drawChessboardCorners(image, board_pattern, pts, complete);
            cv::imshow("3DV Tutorial: Camera Calibration", image);
            key = cv::waitKey();
            if (key == 27) break;               // 'ESC' key
            else if (complete && key == 13)     // 'Enter' key
            {
                image_points.push_back(pts);
                std::cout << image_points.size() << " images are selected for camera calibration." << std::endl;
            }
        }
    }
    video.release();
    if (image_points.empty()) return -1;

    // Prepare 3D points from the chess board
    std::vector<std::vector<cv::Point3f> > object_points(1);
    for (int r = 0; r < board_pattern.height; r++)
        for (int c = 0; c < board_pattern.width; c++)
            object_points[0].push_back(cv::Point3f(board_cellsize * c, board_cellsize * r, 0));
    object_points.resize(image_points.size(), object_points[0]); // Copy

    // Calibrate the camera
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat dist_coeff = cv::Mat::zeros(4, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(object_points, image_points, image_size, K, dist_coeff, rvecs, tvecs);

    // Report calibration results
    std::ofstream report("camera_calibration.txt");
    if (!report.is_open()) return -1;
    report << "## Camera Calbration Results" << std::endl;
    report << "* The number of applied images = " << image_points.size() << std::endl;
    report << "* RMS error = " << rms << std::endl;
    report << "* Camera matrix (K) = " << std::endl << "  " << K.row(0) << K.row(1) << K.row(2) << std::endl;
    report << "* Distortion coefficient (k1, k2, p1, p2, k3, ...) = " << std::endl << "  " << dist_coeff.t() << std::endl;
    report.close();
    return 0;
}
