#include "opencv_all.hpp"

#define Rx(rx)  (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, cos(rx), -sin(rx), 0, sin(rx), cos(rx))
#define Ry(ry)  (cv::Mat_<double>(3, 3) << cos(ry), 0, sin(ry), 0, 1, 0, -sin(ry), 0, cos(ry))
#define Rz(rz)  (cv::Mat_<double>(3, 3) << cos(rz), -sin(rz), 0, sin(rz), cos(rz), 0, 0, 0, 1)

int main(void)
{
    // The given camera configuration: focal length, principal point, image resolution, position, and orientation
    double camera_focal = 1000;
    cv::Point2d camera_center(320, 240);
    cv::Size camera_res(640, 480);
    cv::Point3d camera_pos[] = { cv::Point3d(0, 0, 0), cv::Point3d(-2, -2, 0), cv::Point3d(2, 2, 0) };
    cv::Point3d camera_ori[] = { cv::Point3d(0, 0, 0), cv::Point3d(-CV_PI / 12, CV_PI / 12, 0), cv::Point3d(CV_PI / 12, -CV_PI / 12, 0) };

    // Load a point cloud in the homogeneous coordinate
    FILE* fin = fopen("data/box.xyz", "rt");
    if (fin == NULL) return -1;
    cv::Mat X;
    while (!feof(fin))
    {
        double x, y, z;
        if (fscanf(fin, "%lf %lf %lf", &x, &y, &z) == 3)
        {
            X.push_back<double>(x);
            X.push_back<double>(y);
            X.push_back<double>(z);
            X.push_back<double>(1);
        }
    }
    fclose(fin);
    X = X.reshape(1, X.rows / 4).t(); // Convert to a 4 x N matrix

    // Generate images for each camera pose
    cv::Mat K = (cv::Mat_<double>(3, 3) << camera_focal, 0, camera_center.x, 0, camera_focal, camera_center.y, 0, 0, 1);
    for (int i = 0; i < sizeof(camera_pos) / sizeof(cv::Point3d); i++)
    {
        // Derive a projection matrix
        cv::Mat Rc = Rz(camera_ori[i].z) * Ry(camera_ori[i].y) * Rx(camera_ori[i].x);
        cv::Mat tc = (cv::Mat_<double>(3, 1) << camera_pos[i].x, camera_pos[i].y, camera_pos[i].z);
        cv::Mat Rt;
        cv::hconcat(Rc.t(), -Rc.t() * tc, Rt);
        cv::Mat P = K * Rt;

        // Project the points (c.f. OpenCV provide 'cv::projectPoints' with consideration of distortion.)
        cv::Mat x = P * X;
        x.row(0) = x.row(0) / x.row(2);
        x.row(1) = x.row(1) / x.row(2);
        x.row(2) = 1;

        // Show and store the points
        cv::Mat image = cv::Mat::zeros(camera_res, CV_8UC1);
        for (int c = 0; c < x.cols; c++)
        {
            cv::Point p(x.at<double>(0, c), x.at<double>(1, c));
            if (p.x >= 0 && p.x < camera_res.width && p.y >= 0 && p.y < camera_res.height)
                cv::circle(image, p, 2, 255, -1);
        }
        cv::imshow(cv::format("3DV_Tutorial: Image Generation %d", i), image);
        cv::waitKey(0);

        std::ofstream fout(cv::format("image_generation%d.csv", i));
        if (!fout.is_open()) break;
        fout << cv::format(x.t(), cv::Formatter::FMT_CSV);
        fout.close();
    }

    return 0;
}