#include "opencv2/opencv.hpp"

void MouseEventHandler(int event, int x, int y, int flags, void* param)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        // Add the point to the given vector
        std::vector<cv::Point> *points_src = (std::vector<cv::Point> *)param;
        points_src->push_back(cv::Point(x, y));
        printf("A point (index: %zd) is selected at (%d, %d).\n", points_src->size() - 1, x, y);
    }
}

int main()
{
    const char* input = "../data/sunglok_card.jpg";
    cv::Size card_size(450, 250);

    // Prepare the rectified points
    std::vector<cv::Point> points_dst;
    points_dst.push_back(cv::Point(0, 0));
    points_dst.push_back(cv::Point(card_size.width, 0));
    points_dst.push_back(cv::Point(0, card_size.height));
    points_dst.push_back(cv::Point(card_size.width, card_size.height));

    // Load an image
    cv::Mat original = cv::imread(input);
    if (original.empty()) return -1;

    // Get the matched points from a user's mouse
    std::vector<cv::Point> points_src;
    cv::namedWindow("3DV Tutorial: Perspective Correction");
    cv::setMouseCallback("3DV Tutorial: Perspective Correction", MouseEventHandler, &points_src);
    while (points_src.size() < 4)
    {
        cv::Mat display = original.clone();
        cv::rectangle(display, cv::Rect(cv::Point(10, 10), card_size), cv::Vec3b(0, 0, 255), 2);
        size_t idx = cv::min(points_src.size(), points_dst.size() - 1);
        cv::circle(display, points_dst[idx] + cv::Point(10, 10), 5, cv::Vec3b(0, 255, 0), -1);
        cv::imshow("3DV Tutorial: Perspective Correction", display);
        if (cv::waitKey(1) == 27) break; // 'ESC' key: Exit
    }
    if (points_src.size() < 4) return -1;

    // Calculate planar homography and rectify perspective distortion
    cv::Mat H = cv::findHomography(points_src, points_dst);
    cv::Mat rectify;
    cv::warpPerspective(original, rectify, H, card_size);

    // Show the rectified image
    cv::imshow("3DV Tutorial: Perspective Correction", rectify);
    cv::waitKey(0);
    return 0;
}
