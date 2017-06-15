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
    bool use_box = false;
    double f = 810.5, cx = 480, cy = 270, theta = 18.7 * CV_PI / 180, L = 3.31;
    double H = 1.9;

    // Load background and object images
    cv::Mat image = cv::imread("data/daejeon.jpg");
    cv::Mat object = cv::imread("data/daejeon_giraffe.png");
    if (image.empty() || object.empty()) return -1;

    // Prepare an object mask
    std::vector<cv::Mat> object_ch(object.channels());
    cv::split(object, object_ch);
    if (object_ch.size() < 3) return -1;
    cv::Mat object_mask = (object_ch[0] != 0) | (object_ch[1] != 255) | (object_ch[2] != 0); // Background is green, (0, 255, 0).

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
            cv::Rect obj_box;
            obj_box.x = contact.x;
            obj_box.y = head;
            obj_box.height = contact.y - head;
            obj_box.width = object.cols * (contact.y - head) / object.rows;
            if (use_box) cv::rectangle(image, obj_box, cv::Scalar(0, 255, 0), 2);
            else if (obj_box.tl().x >= 0 && obj_box.tl().y >= 0 && obj_box.br().x < image.cols && obj_box.br().y < image.rows)
            {
                cv::Mat obj, obj_mask;
                cv::resize(object, obj, obj_box.size(), 0, 0, 0);
                cv::resize(object_mask, obj_mask, obj_box.size(), 0, 0, 0);
                obj.copyTo(image(obj_box), obj_mask);
            }
            contact_prev = contact;
        }

        // Show the image
        cv::imshow("3DV Tutorial: Simple Object Proposal", image);
        int key = cv::waitKey(1);
        if (key == 27) break; // 'ESC' key: Exit
    }

    return 0;
}