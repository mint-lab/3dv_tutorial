#include "opencv_all.hpp"

int main(void)
{
    cv::Mat K = (cv::Mat_<double>(3, 3) << 864.43, 0, 952.16, 0, 861.90, 581.92, 0, 0, 1);
    cv::Mat dist_coeff = (cv::Mat_<double>(5, 1) << -0.2862454796798899, 0.104008092811024, -0.000707339764553983, 6.557555700709726e-005, -0.01922363820734248);

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
        if (key == 27) break;                                   // "ESC" key
        else if (key == 9) show_rectify = !show_rectify;        // "Tab" key
        else if (key == 32)                                     // "Space" key
        {
            key = cv::waitKey();
            if (key == 27) break;                               // "ESC" key
            else if (key == 9) show_rectify = !show_rectify;    // "Tab" key
        }
    }

    video.release();
    return 0;
}