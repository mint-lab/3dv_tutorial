#include "opencv_all.hpp"

int main(void)
{
    bool use_height_filter = true;
    double f = 810.5, cx = 480, cy = 270, theta = 18.7 * CV_PI / 180, L = 3.31;
    int backsub_history = 1000, area_thresh = 100;
    double backsub_thresh = 32, height_thresh = 1;
    cv::Mat morph_kernel_small = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat morph_kernel_big = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 10));

    // Open a video and create MOG2 background subtractor
    cv::VideoCapture video;
    cv::Ptr<cv::BackgroundSubtractorMOG2> backsub = cv::createBackgroundSubtractorMOG2(backsub_history, backsub_thresh);
    if (!video.open("data/daejeon.avi") || backsub.empty()) return -1;

    // Run foreground extraction and object detection
    while (true)
    {
        // Grab an image from the video
        cv::Mat image, fgmask;
        video >> image;
        if (image.empty()) break;

        // Extract the foreground mask
        backsub->apply(image, fgmask);
        cv::threshold(fgmask, fgmask, 128, 255, cv::THRESH_BINARY);
        cv::erode(fgmask, fgmask, morph_kernel_small, cv::Point(-1, -1));
        cv::dilate(fgmask, fgmask, morph_kernel_small, cv::Point(-1, -1));
        cv::dilate(fgmask, fgmask, morph_kernel_big, cv::Point(-1, -1), 2);
        cv::erode(fgmask, fgmask, morph_kernel_big, cv::Point(-1, -1), 2);

        // Extract blobs
        std::vector<cv::Rect> blobs;
        cv::Mat visit = cv::Mat::zeros(fgmask.rows + 2, fgmask.cols + 2, CV_8UC1);
        for (int v = 0; v < fgmask.rows; v++)
        {
            const uchar* fgmask_row = fgmask.ptr<uchar>(v);
            uchar* visit_row = visit.ptr<uchar>(v + 1);
            for (int u = 0; u < fgmask.cols; u++)
            {
                if (fgmask_row[u] == 255 && visit_row[u + 1] == 0)
                {
                    cv::Rect blob;
                    int area = cv::floodFill(fgmask, visit, cv::Point(u, v), 255, &blob);
                    if (area > area_thresh) blobs.push_back(blob);
                }
            }
        }

        // Show the image and foreground mask together
        if (image.channels() < 3) cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
        cv::cvtColor(fgmask, fgmask, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < blobs.size(); i++)
        {
            double phi = atan2(blobs[i].y + blobs[i].height - cy, f), phi_head = atan2(blobs[i].y - cy, f);
            double Z = L / tan(phi + theta), Z_head = L / tan(phi_head + theta);
            double H = (Z_head - Z) / Z_head * L;
            cv::Scalar color = cv::Scalar(0, 255, 0);
            if (use_height_filter && H < height_thresh) color = cv::Scalar(127, 127, 255);
            cv::rectangle(image, blobs[i], color, 2);
            cv::rectangle(fgmask, blobs[i], color, 2);
            cv::putText(image, cv::format("(D%.2f,H%.2f)", Z, H), blobs[i].tl() + cv::Point(-5, -15), cv::FONT_HERSHEY_DUPLEX, 0.5, color);
        }
        cv::hconcat(image, fgmask, image);
        cv::imshow("3DV Tutorial: Simple Object Filtering", image);
        if (cv::waitKey(1) == 27) break; // 'ESC' key: Exit
    }

    video.release();
    return 0;
}
