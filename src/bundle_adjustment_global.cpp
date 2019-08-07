#include "opencv2/opencv.hpp"
#include "cvsba.h"

int main()
{
    double camera_focal = 1000;
    cv::Point2d camera_center(320, 240);
    int n_views = 5;

    // Load multiple views of 'box.xyz'
    // c.f. You need to run 'image_formation.cpp' to generate point observation.
    //      You can apply Gaussian noise by change value of 'camera_noise' if necessay.
    std::vector<std::vector<cv::Point2d> > xs;
    for (int i = 0; i < n_views; i++)
    {
        FILE* fin = fopen(cv::format("image_formation%d.xyz", i).c_str(), "rt");
        if (fin == NULL) return -1;
        std::vector<cv::Point2d> pts;
        while (!feof(fin))
        {
            double x, y, w;
            if (fscanf(fin, "%lf %lf %lf", &x, &y, &w) == 3)
                pts.push_back(cv::Point2d(x, y));
        }
        fclose(fin);
        xs.push_back(pts);
        if (xs.front().size() != xs.back().size()) return -1;
    }

    // Assume that all cameras have the same and known camera matrix
    // Assume that all feature points are visible on all views
    cv::Mat K = (cv::Mat_<double>(3, 3) << camera_focal, 0, camera_center.x, 0, camera_focal, camera_center.y, 0, 0, 1);
    cv::Mat dist_coeff = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<int> visible_all(xs.front().size(), 1);

    // Initialize each camera projection matrix
    std::vector<cv::Mat> Ks, dist_coeffs, Rs, ts;
    std::vector<std::vector<int> > visibility;
    for (int i = 0; i < n_views; i++)
    {
        visibility.push_back(visible_all);
        Ks.push_back(K.clone());                                // K for all cameras
        dist_coeffs.push_back(cv::Mat::zeros(5, 1, CV_64F));    // dist_coeff for all cameras
        Rs.push_back(cv::Mat::eye(3, 3, CV_64F));               // R for all cameras
        ts.push_back(cv::Mat::zeros(3, 1, CV_64F));             // t for all cameras
    }

    // Initialize 3D points
    std::vector<cv::Point3d> Xs;
    Xs.resize(xs.front().size(), cv::Point3d(0, 0, 5.5));

    // Optimize camera pose and 3D points
    try
    {
        cvsba::Sba sba;
        cvsba::Sba::Params param;
        param.type = cvsba::Sba::MOTIONSTRUCTURE;
        param.fixedIntrinsics = 5;
        param.fixedDistortion = 5;
        param.verbose = true;
        sba.setParams(param);
        double error = sba.run(Xs, xs, visibility, Ks, Rs, ts, dist_coeffs);
    }
    catch (cv::Exception) { }

    // Store the 3D points to an XYZ file
    FILE* fpts = fopen("bundle_adjustment_global(point).xyz", "wt");
    if (fpts == NULL) return -1;
    for (size_t i = 0; i < Xs.size(); i++)
        fprintf(fpts, "%f %f %f\n", Xs[i].x, Xs[i].y, Xs[i].z);
    fclose(fpts);

    // Store the camera poses to an XYZ file 
    FILE* fcam = fopen("bundle_adjustment_global(camera).xyz", "wt");
    if (fcam == NULL) return -1;
    for (size_t i = 0; i < Rs.size(); i++)
    {
        cv::Mat p = -Rs[i].t() * ts[i];
        fprintf(fcam, "%f %f %f\n", p.at<double>(0), p.at<double>(1), p.at<double>(2));
    }
    fclose(fcam);
    return 0;
}
