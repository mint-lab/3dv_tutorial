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
    cv::Point xy_s, xy_e;
};

void MouseEventHandler(int event, int x, int y, int flags, void* param)
{
    // Change 'mouse_state' (given as 'param') according to the mouse 'event'
    if (param == NULL) return;
    MouseDrag* drag = (MouseDrag*)param;
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        drag->dragged = true;
        drag->xy_s = cv::Point(x, y);
        drag->xy_e = cv::Point(0, 0);
    }
    else if (event == cv::EVENT_MOUSEMOVE)
    {
        if (drag->dragged) drag->xy_e = cv::Point(x, y);
    }
    else if (event == cv::EVENT_LBUTTONUP)
    {
        if (drag->dragged)
        {
            drag->dragged = false;
            drag->xy_e = cv::Point(x, y);
        }
    }
}

int main()
{
    // The given image and its calibration data
    const char* img_file = "../data/daejeon_station.png";
    double f = 810.5, cx = 480, cy = 270, L = 3.31;                   // Unit: [px], [px], [px], [m]
    cv::Point3d cam_ori(DEG2RAD(-18.7), DEG2RAD(-8.2), DEG2RAD(2.0)); // Unit: [deg]
    cv::Range grid_x(-2, 3), grid_z(5, 36);                           // Unit: [m]

    // Load an images
    cv::Mat img = cv::imread(img_file);
    if (img.empty()) return -1;

    // Register the mouse callback function
    MouseDrag drag;
    cv::namedWindow("Object Localization and Measurement");
    cv::setMouseCallback("Object Localization and Measurement", MouseEventHandler, &drag);

    // Prepare the camera projection
    cv::Matx33d K(f, 0, cx, 0, f, cy, 0, 0, 1);
    cv::Matx33d Rc = Rz(cam_ori.z) * Ry(cam_ori.y) * Rx(cam_ori.x);
    cv::Point3d tc = cv::Point3d(0, -L, 0);
    cv::Matx33d R = Rc.t();
    cv::Point3d t = -Rc.t() * tc;

    // Draw X- and Z-grids on the ground
    for (int z = grid_z.start; z <= grid_z.end; z++)
    {
        cv::Point3d p = K * (R * cv::Point3d(grid_x.start, 0, z) + t);
        cv::Point3d q = K * (R * cv::Point3d(grid_x.end, 0, z) + t);
        cv::line(img, cv::Point2d(p.x / p.z, p.y / p.z), cv::Point2d(q.x / q.z, q.y / q.z), cv::Vec3b(64, 128, 64), 1);
    }
    for (int x = grid_x.start; x <= grid_x.end; x++)
    {
        cv::Point3d p = K * (R * cv::Point3d(x, 0, grid_z.start) + t);
        cv::Point3d q = K * (R * cv::Point3d(x, 0, grid_z.end) + t);
        cv::line(img, cv::Point2d(p.x / p.z, p.y / p.z), cv::Point2d(q.x / q.z, q.y / q.z), cv::Vec3b(64, 128, 64), 1);
    }

    while (true)
    {
        cv::Mat img_copy = img.clone();
        if (drag.xy_e.x > 0 && drag.xy_e.y > 0)
        {
            // Calculate object location and height
            cv::Point3d c = R.t() * cv::Point3d(drag.xy_s.x - cx, drag.xy_s.y - cy, f);
            if (c.y < DBL_EPSILON) continue; // Skip the degenerate case (beyond the horizon)
            cv::Point3d h = R.t() * cv::Point3d(drag.xy_e.x - cx, drag.xy_e.y - cy, f);
            double Z = c.z / c.y * L, X = c.x / c.y * L, H = (c.y / c.z - h.y / h.z) * Z;

            // Draw head/contact points and location/height
            cv::line(img_copy, drag.xy_s, drag.xy_e, cv::Vec3b(0, 0, 255), 2);
            cv::circle(img_copy, drag.xy_e, 4, cv::Vec3b(255, 0, 0), -1);
            cv::circle(img_copy, drag.xy_s, 4, cv::Vec3b(0, 255, 0), -1);
            cv::putText(img_copy, cv::format("X:%.2f, Z:%.2f, H:%.2f", X, Z, H), drag.xy_s + cv::Point(-20, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Vec3b(0, 255, 0));
        }

        // Show the image
        cv::imshow("Object Localization and Measurement", img_copy);
        int key = cv::waitKey(1);
        if (key == 27) break; // ESC
    }
    return 0;
}
