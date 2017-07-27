#include "opencv_all.hpp"
#include "cvsba.h"

int main(void)
{
    double camera_focal = 1000;
    cv::Point2d camera_center(320, 240);
    int n_views = 5;

    // Load multiple views of 'box.xyz'
    // c.f. You need to run 'image_formation.cpp' to generate point observation.
    //      You can apply Gaussian noise by change value of 'camera_noise' if necessay.
    std::vector<std::vector<cv::Point2d> > data;
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
        data.push_back(pts);
        if (data.front().size() != data.back().size()) return -1;
    }

    // Assume that all cameras have the same and known camera matrix
    // Assume that all feature points are visible on all views
    cv::Mat K = (cv::Mat_<double>(3, 3) << camera_focal, 0, camera_center.x, 0, camera_focal, camera_center.y, 0, 0, 1);
    cv::Mat dist_coeff = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<int> visible_all(data.front().size(), 1);

    // Prepare each camera projection matrix
    std::vector<cv::Mat> Ks, dist_coeffs, Rs, ts;
    std::vector<std::vector<int> > visibility;
    std::vector<std::vector<cv::Point2d> > xs;

    xs.push_back(data[0]);
    visibility.push_back(visible_all);
    Ks.push_back(K.clone());                                    // K for the first camera (index: 0)
    dist_coeffs.push_back(dist_coeff.clone());                  // dist_coeff for the first camera (index: 0)
    Rs.push_back(cv::Mat::eye(3, 3, CV_64F));                   // R for the first camera (index: 0)
    ts.push_back(cv::Mat::zeros(3, 1, CV_64F));                 // t for the first camera (index: 0)

    // 1) Select the best pair (skipped because all points are visible on all images)

    // 2) Estimate relative pose of the inital two views (epipolar geometry)
    cv::Mat F = cv::findFundamentalMat(data[0], data[1], cv::FM_8POINT);
    cv::Mat E = K.t() * F * K;
    cv::Mat R, t;
    cv::recoverPose(E, data[0], data[1], K, R, t);

    xs.push_back(data[1]);
    visibility.push_back(visible_all);
    Ks.push_back(K.clone());                                    // K for the second camera
    dist_coeffs.push_back(dist_coeff.clone());                  // dist_coeff for the second camera
    Rs.push_back(R.clone());                                    // R for the second camera
    ts.push_back(t.clone());                                    // t for the second camera

    // 3) Reconstruct 3D points of the initial two views (triangulation)
    cv::Mat Rt;
    cv::hconcat(R, t, Rt);
    cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat P1 = K * Rt, X;
    cv::triangulatePoints(P0, P1, xs[0], xs[1], X);

    std::vector<cv::Point3d> Xs;
    X.row(0) = X.row(0) / X.row(3);
    X.row(1) = X.row(1) / X.row(3);
    X.row(2) = X.row(2) / X.row(3);
    X.row(3) = 1;
    for (int c = 0; c < X.cols; c++)
        Xs.push_back(cv::Point3d(X.col(c).rowRange(0, 3)));

    // Incrementally add more views
    cvsba::Sba sba;
    cvsba::Sba::Params param;
    param.type = cvsba::Sba::MOTIONSTRUCTURE;
    param.fixedIntrinsics = 5;
    param.fixedDistortion = 5;
    param.verbose = true;
    sba.setParams(param);
    for (int i = 2; i < n_views; i++)
    {
        // 4) Select the next image to add (skipped because all points are visible on all images)

        // 5) Estimate relative pose of the next view (PnP)
        cv::Mat rvec;
        cv::solvePnP(Xs, data[i], K, dist_coeff, rvec, t);
        cv::Rodrigues(rvec, R);

        xs.push_back(data[i]);
        visibility.push_back(visible_all);
        Ks.push_back(K.clone());                                // K for the third and other cameras
        dist_coeffs.push_back(dist_coeff.clone());              // dist_coeff for the third and other cameras
        Rs.push_back(R.clone());                                // R for the third and other cameras
        ts.push_back(t.clone());                                // t for the third and other cameras

        // 6) Reconstruct newly observed 3D points (triangulation, skipped because all points are visible on all images)

        // 7) Optimize camera pose and 3D points (bundle adjustment)
        try { double error = sba.run(Xs, xs, visibility, Ks, Rs, ts, dist_coeffs); }
        catch (cv::Exception) { }
    }

    // Store the 3D points to an XYZ file
    FILE* fpts = fopen("bundle_adjustment_inc(point).xyz", "wt");
    if (fpts == NULL) return -1;
    for (size_t i = 0; i < Xs.size(); i++)
        fprintf(fpts, "%f %f %f\n", Xs[i].x, Xs[i].y, Xs[i].z);
    fclose(fpts);

    // Store the camera poses to an XYZ file 
    FILE* fcam = fopen("bundle_adjustment_inc(camera).xyz", "wt");
    if (fcam == NULL) return -1;
    for (size_t i = 0; i < Rs.size(); i++)
    {
        cv::Mat p = -Rs[i].t() * ts[i];
        fprintf(fcam, "%f %f %f\n", p.at<double>(0), p.at<double>(1), p.at<double>(2));
    }
    fclose(fcam);
    return 0;
}
