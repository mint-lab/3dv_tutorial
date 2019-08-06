#include "opencv_all.hpp"

int main(void)
{
    bool use_height_filter = true;
    double f = 810.5, cx = 480, cy = 270, theta = 18.7 * CV_PI / 180, L = 3.31;
    int backsub_history = 1000, area_thresh = 100;
    double backsub_thresh = 32, height_thresh = 1;

    // Open a video and create MOG2 background subtractor
    cv::VideoCapture video;
    cv::Ptr<cv::BackgroundSubtractorMOG2> backsub = cv::createBackgroundSubtractorMOG2(backsub_history, backsub_thresh);
    if (!video.open("data/daejeon.avi") || backsub.empty()) return -1;

    // Run foreground extraction and object detection
    cv::Matx33d R_t(1, 0, 0, 0, cos(-theta), -sin(-theta), 0, sin(-theta), cos(-theta));
    cv::Mat morph_kernel_small = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat morph_kernel_big = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 10));
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
            cv::Vec3d p_foot = R_t * cv::Vec3d(blobs[i].x + blobs[i].width / 2 - cx, blobs[i].y + blobs[i].height - cy, f);
            cv::Vec3d p_head = R_t * cv::Vec3d(blobs[i].x + blobs[i].width / 2 - cx, blobs[i].y - cy, f);
            if (p_foot[1] <= 0 || p_head[1] <= 0) continue;
            double Z_foot = p_foot[2] / p_foot[1] * L, x_foot = p_foot[0] / p_foot[1] * L;
            double Z_head = p_head[2] / p_head[1] * L;
            double H = (Z_head - Z_foot) / Z_head * L;
            cv::Scalar color = cv::Scalar(0, 255, 0);
            if (use_height_filter && H < height_thresh) color = cv::Scalar(127, 127, 255);
            cv::rectangle(image, blobs[i], color, 2);
            cv::rectangle(fgmask, blobs[i], color, 2);
            cv::putText(image, cv::format("(X:%.2f,Z:%.2f,H:%.2f)", x_foot, Z_foot, H), blobs[i].tl() + cv::Point(-5, -15), cv::FONT_HERSHEY_DUPLEX, 0.5, color);
        }
        cv::hconcat(image, fgmask, image);
        cv::imshow("3DV Tutorial: Simple Object Filtering", image);
        int key = cv::waitKey(1);
        if (key == 27) break;       // 'ESC' key: Exit
        else if (key == 32)         // 'Space' key: Pause
        {
            key = cv::waitKey(0);
            if (key == 27) break;   // 'ESC' key: Exit
        }
    }

    video.release();
    return 0;
}
