#include "opencv2/opencv.hpp"

#define DEG2RAD(v)  (v * CV_PI / 180)
#define Rx(rx)      (cv::Matx33d(1, 0, 0, 0, cos(rx), -sin(rx), 0, sin(rx), cos(rx)))
#define Ry(ry)      (cv::Matx33d(cos(ry), 0, sin(ry), 0, 1, 0, -sin(ry), 0, cos(ry)))
#define Rz(rz)      (cv::Matx33d(cos(rz), -sin(rz), 0, sin(rz), cos(rz), 0, 0, 0, 1))

class MouseDrag
{
public:
    MouseDrag() : dragged(false) { }
    bool dragged;
    cv::Point start, end;
};

void MouseEventHandler(int event, int x, int y, int flags, void* param)
{
    if (param == NULL) return;
    MouseDrag* drag = (MouseDrag*)param;
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        drag->dragged = true;
        drag->start = cv::Point(x, y);
        drag->end = cv::Point(0, 0);
    }
    else if (event == cv::EVENT_MOUSEMOVE)
    {
        if (drag->dragged) drag->end = cv::Point(x, y);
    }
    else if (event == cv::EVENT_LBUTTONUP)
    {
        if (drag->dragged)
        {
            drag->dragged = false;
            drag->end = cv::Point(x, y);
        }
    }
}

int main()
{
    const char* input = "data/daejeon_station.png";
    double f = 810.5, cx = 480, cy = 270, L = 3.31;
    cv::Point3d cam_ori(DEG2RAD(-18.7), DEG2RAD(-8.2), DEG2RAD(2.0));
    cv::Range grid_x(-2, 3), grid_z(5, 35);

    // Load an images
    cv::Mat image = cv::imread(input);
    if (image.empty()) return -1;

    // Configure mouse callback
    MouseDrag drag;
    cv::namedWindow("3DV Tutorial: Object Localization and Measurement");
    cv::setMouseCallback("3DV Tutorial: Object Localization and Measurement", MouseEventHandler, &drag);

    // Draw grids on the ground
    cv::Matx33d K(f, 0, cx, 0, f, cy, 0, 0, 1);
    cv::Matx33d Rc = Rz(cam_ori.z) * Ry(cam_ori.y) * Rx(cam_ori.x), R = Rc.t();
    cv::Point3d tc = cv::Point3d(0, -L, 0), t = -Rc.t() * tc;
    for (int z = grid_z.start; z <= grid_z.end; z++)
    {
        cv::Point3d p = K * (R * cv::Point3d(grid_x.start, 0, z) + t);
        cv::Point3d q = K * (R * cv::Point3d(grid_x.end, 0, z) + t);
        cv::line(image, cv::Point2d(p.x / p.z, p.y / p.z), cv::Point2d(q.x / q.z, q.y / q.z), cv::Vec3b(64, 128, 64), 1);
    }
    for (int x = grid_x.start; x <= grid_x.end; x++)
    {
        cv::Point3d p = K * (R * cv::Point3d(x, 0, grid_z.start) + t);
        cv::Point3d q = K * (R * cv::Point3d(x, 0, grid_z.end) + t);
        cv::line(image, cv::Point2d(p.x / p.z, p.y / p.z), cv::Point2d(q.x / q.z, q.y / q.z), cv::Vec3b(64, 128, 64), 1);
    }

    while (true)
    {
        cv::Mat image_copy = image.clone();
        if (drag.end.x > 0 && drag.end.y > 0)
        {
            // Calculate object location and height
            cv::Point3d c = R.t() * cv::Point3d(drag.start.x - cx, drag.start.y - cy, f);
            if (c.y < DBL_EPSILON) continue; // Skip the degenerate case (beyond the horizon)
            cv::Point3d h = R.t() * cv::Point3d(drag.end.x - cx, drag.end.y - cy, f);
            double Z = c.z / c.y * L, X = c.x / c.y * L, H = (c.y / c.z - h.y / h.z) * Z;

            // Draw head/contact points and location/height
            cv::line(image_copy, drag.start, drag.end, cv::Vec3b(0, 0, 255), 2);
            cv::circle(image_copy, drag.end, 4, cv::Vec3b(255, 0, 0), -1);
            cv::circle(image_copy, drag.start, 4, cv::Vec3b(0, 255, 0), -1);
            cv::putText(image_copy, cv::format("X:%.2f, Z:%.2f, H:%.2f", X, Z, H), drag.start + cv::Point(-20, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Vec3b(0, 255, 0));
        }

        // Show the image
        cv::imshow("3DV Tutorial: Object Localization and Measurement", image_copy);
        int key = cv::waitKey(1);
        if (key == 27) break; // 'ESC' key: Exit
    }
    return 0;
}
