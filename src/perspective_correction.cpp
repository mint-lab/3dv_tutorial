#include "opencv_all.hpp"

void MouseEventHandler(int event, int x, int y, int flags, void* param)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        // Add the point to the given vector
        std::vector<cv::Point> *points = (std::vector<cv::Point> *)param;
        points->push_back(cv::Point(x, y));
        printf("A point (index: %d) is selectd at (%d, %d).\n", points->size() - 1, x, y);
    }
}

int main(void)
{
    cv::Size plate_size(450, 250);

    // Prepare the rectified points
    std::vector<cv::Point> points_dst;
    points_dst.push_back(cv::Point(0, 0));
    points_dst.push_back(cv::Point(plate_size.width, 0));
    points_dst.push_back(cv::Point(0, plate_size.height));
    points_dst.push_back(cv::Point(plate_size.width, plate_size.height));

    // Load an image
    cv::Mat original = cv::imread("data/sunglok.jpg");
    if (original.empty()) return -1;

    // Get the matched points from a user's mouse
    std::vector<cv::Point> points_src;
    cv::namedWindow("3DV Tutorial: Perspective Correction");
    cv::setMouseCallback("3DV Tutorial: Perspective Correction", MouseEventHandler, &points_src);
    while (points_src.size() < 4)
    {
        cv::Mat display = original.clone();
        cv::rectangle(display, cv::Rect(cv::Point(10, 10), plate_size), cv::Scalar(0, 0, 255), 2);
        int idx = cv::min(points_src.size(), points_dst.size() - 1);
        cv::circle(display, points_dst[idx] + cv::Point(10, 10), 5, cv::Scalar(0, 255, 0), -1);
        cv::imshow("3DV Tutorial: Perspective Correction", display);
        if (cv::waitKey(1) == 27) break; // 'ESC' key
    }
    if (points_src.size() < 4) return -1;

    // Calculate planar homography and rectify perspective distortion
    cv::Mat H = cv::findHomography(points_src, points_dst);
    cv::Mat rectify;
    cv::warpPerspective(original, rectify, H, plate_size);

    // Show the rectified image
    cv::imshow("3DV Tutorial: Perspective Correction", rectify);
    printf("Press any key to terminate tihs program!\n");
    cv::waitKey(0);
    return 0;
}