#include "opencv2/opencv.hpp"
#include <iostream>

void MouseEventHandler(int event, int x, int y, int flags, void* param)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        std::vector<cv::Point> *pts_src = (std::vector<cv::Point> *)param;
        if (pts_src->size() < 4) pts_src->push_back(cv::Point(x, y));
    }
}

int main()
{
    const char* img_file = "../data/sunglok_card.jpg";
    cv::Size card_size(450, 250);
    int offset = 10;

    // Prepare the rectified points
    std::vector<cv::Point> pts_dst;
    pts_dst.push_back(cv::Point(0, 0));
    pts_dst.push_back(cv::Point(card_size.width, 0));
    pts_dst.push_back(cv::Point(0, card_size.height));
    pts_dst.push_back(cv::Point(card_size.width, card_size.height));

    // Load an image
    cv::Mat img = cv::imread(img_file);
    if (img.empty()) return -1;

    // Get the matched points from mouse clicks
    std::vector<cv::Point> pts_src;
    cv::namedWindow("Perspective Correction: Point Selection");
    cv::setMouseCallback("Perspective Correction: Point Selection", MouseEventHandler, &pts_src);
    while (pts_src.size() < pts_dst.size())
    {
        cv::Mat img_display = img.clone();
        cv::rectangle(img_display, cv::Rect(cv::Point(offset, offset), card_size), cv::Vec3b(0, 0, 255), 2);
        size_t idx = cv::min(pts_src.size(), pts_dst.size() - 1);
        cv::circle(img_display, pts_dst[idx] + cv::Point(offset, offset), 5, cv::Vec3b(0, 255, 0), -1);
        cv::imshow("Perspective Correction: Point Selection", img_display);
        if (cv::waitKey(10) == 27) break; // ESC
    }

    if (pts_src.size() >= pts_dst.size())
    {
        // Calculate planar homography and rectify perspective distortion
        cv::Mat H = cv::findHomography(pts_src, pts_dst);
        std::cout << "pts_src = " << cv::Mat(pts_src) << std::endl;
        std::cout << "pts_dst = " << cv::Mat(pts_dst) << std::endl;
        std::cout << "H = " << H << std::endl;
        cv::Mat img_rectify;
        cv::warpPerspective(img, img_rectify, H, card_size);

        // Show the rectified image
        cv::imshow("Perspective Distortion Correction: Rectified Image", img_rectify);
        cv::waitKey(0);
    }

    cv::destroyAllWindows();
    return 0;
}
