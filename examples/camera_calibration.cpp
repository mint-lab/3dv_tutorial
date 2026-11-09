#include "opencv2/opencv.hpp"
#include <iostream>

std::vector<cv::Mat> select_img_from_video(const cv::String& video_file, const cv::Size& board_pattern,
    bool select_all = false, int wait_msec = 10, const cv::String& wnd_name = "Camera Calibration")
{
    // Open a video
    cv::VideoCapture video(video_file);
    CV_Assert(video.isOpened());

    // Select images
    std::vector<cv::Mat> img_select;
    while (true)
    {
        // Read an image from the video
        cv::Mat img;
        if (!video.read(img)) break;

        if (select_all) img_select.push_back(img.clone());
        else
        {
            // Show the image
            cv::Mat display = img.clone();
            cv::putText(display, cv::format("NSelect: %zu", img_select.size()), cv::Point(10, 25), cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(0, 255, 0));
            cv::imshow(wnd_name, display);

            // Process the key event
            int key = cv::waitKey(wait_msec);
            if (key == ' ') // Space: Pause and show corners
            {
                std::vector<cv::Point2f> pts;
                bool complete = cv::findChessboardCorners(img, board_pattern, pts);
                cv::drawChessboardCorners(display, board_pattern, pts, complete);
                cv::imshow(wnd_name, display);
                key = cv::waitKey();
                if (key == '\r') img_select.push_back(img.clone()); // Enter: Select the image
            }
            if (key == 27) break; // ESC: Exit (Complete image selection)
        }
    }
    video.release();
    cv::destroyAllWindows();
    return img_select;
}

double calib_camera_from_chessboard(const std::vector<cv::Mat>& images, const cv::Size& board_pattern,
    float board_cellsize, cv::Mat& K, cv::Mat& dist_coeff, std::vector<cv::Mat>& rvecs,
    std::vector<cv::Mat>& tvecs, int calib_flags = 0)
{
    // Find 2D corner points from given images
    std::vector<std::vector<cv::Point2f>> img_points;
    for (const auto& img : images)
    {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> pts;
        if (cv::findChessboardCorners(gray, board_pattern, pts)) img_points.push_back(pts);
    }
    CV_Assert(!img_points.empty());

    // Prepare 3D points of the chess board
    std::vector<cv::Point3f> obj_pts;
    for (int r = 0; r < board_pattern.height; r++)
        for (int c = 0; c < board_pattern.width; c++)
            obj_pts.emplace_back(c * board_cellsize, r * board_cellsize, 0.f);
    std::vector<std::vector<cv::Point3f>> obj_points(img_points.size(), obj_pts);

    // Calibrate the camera
    return cv::calibrateCamera(obj_points, img_points, images.front().size(), K, dist_coeff, rvecs, tvecs, calib_flags);
}

int main()
{
    const char* video_file = "../data/chessboard.avi";
    cv::Size board_pattern(10, 7);
    float board_cellsize = 0.025f;

    auto img_select = select_img_from_video(video_file, board_pattern);
    CV_Assert(!img_select.empty());
    cv::Mat K, dist_coeff;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize, K, dist_coeff, rvecs, tvecs);

    // Print calibration results
    std::cout << "## Camera Calibration Results\n";
    std::cout << "* The number of selected images = " << img_select.size() << "\n";
    std::cout << "* RMS error = " << rms << "\n";
    std::cout << "* Camera matrix (K) = \n" << K << "\n";
    std::cout << "* Distortion coefficient (k1, k2, p1, p2, k3, ...) = " << dist_coeff.t() << std::endl;
    return 0;
}
