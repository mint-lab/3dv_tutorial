#include "opencv_all.hpp"

void MouseEventHandler(int event, int x, int y, int flags, void* param)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        cv::Point2i* point = (cv::Point2i*)param;
        point->x = x;
        point->y = y;
    }
}

int main(void)
{
    double f = 810.5, cx = 480, cy = 270, theta = 18.7 * CV_PI / 180, L = 3.31;
    double H = 1.9;

    // Load an image
    cv::Mat image = cv::imread("data/daejeon.jpg");
    if (image.empty()) return -1;

    // Configure mouse callback
    cv::Point2i contact;
    cv::namedWindow("3DV Tutorial: Simple Object Proposal");
    cv::setMouseCallback("3DV Tutorial: Simple Object Proposal", MouseEventHandler, &contact);

    // Show object proposals if clicked
    cv::Point2i contact_prev;
    while (true)
    {
        if (contact != contact_prev)
        {
            // Calculate a head point from the given contact point
            double phi = atan2(contact.y - cy, f);
            double Z = L / tan(phi + theta);
            double head = cy + f * tan(atan2(L - H, Z) - theta);

            // Draw the object proposal
            double h = contact.y - head;
            double w = h / 3;
            cv::rectangle(image, cv::Rect(contact.x - w / 2, head, w, h), cv::Scalar(0, 255, 0), 2);
            contact_prev = contact;
        }

        // Show the image
        cv::imshow("3DV Tutorial: Simple Object Proposal", image);
        int key = cv::waitKey(1);
        if (key == 27) break; // 'ESC' key: Exit
    }

    return 0;
}