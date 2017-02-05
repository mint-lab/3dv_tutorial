#include "opencv_all.hpp"

int main(void)
{
    cv::Mat K = (cv::Mat_<double>(3, 3) << 434.3588729143197, 0, 476.1230925877231, 0, 432.7138830286992, 289.2709802508451, 0, 0, 1);
    cv::Mat dist_coeff = (cv::Mat_<double>(5, 1) << -0.2918143346191932, 0.1095347774113121, -0.000105133686343854, 4.350475599617356e-005, -0.02083205595737927);

    // Open an video
    cv::VideoCapture video;
    if (!video.open("data/chessboard.avi")) return -1;

    // Run distortion correction
    bool show_rectify = true;
    cv::Mat map1, map2;
    while (true)
    {
        // Grab an image from the video
        cv::Mat image;
        video >> image;
        if (image.empty()) break;

        // Rectify geometric distortion (c.f. 'cv::undistort()' can be applied for one-time remapping.)
        if (show_rectify)
        {
            if (map1.empty() || map2.empty())
                cv::initUndistortRectifyMap(K, dist_coeff, cv::Mat(), cv::Mat(), image.size(), CV_32FC1, map1, map2);
            cv::remap(image, image, map1, map2, cv::InterpolationFlags::INTER_LINEAR);
        }

        // Show the image
        cv::imshow("3DV Tutorial: Distortion Correction", image);
        int key = cv::waitKey(1);
        if (key == 27) break;                                   // 'ESC' key
        else if (key == 9) show_rectify = !show_rectify;        // 'Tab' key
        else if (key == 32)                                     // 'Space' key
        {
            key = cv::waitKey();
            if (key == 27) break;                               // 'ESC' key
            else if (key == 9) show_rectify = !show_rectify;    // 'Tab' key
        }
    }

    video.release();
    return 0;
}