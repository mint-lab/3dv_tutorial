#include "opencv_all.hpp"
#include "cvsba.h"

int main(void)
{
    double camera_focal = 1000;
    cv::Point2d camera_center(320, 240);
    int n_views = 5;

    // Load multiple views of 'box.xyz'
    // c.f. You need to run 'image_generation%02d.cpp' to generate point observation.
    //      You can apply Gaussian noise by change value of 'camera_noise' if necessay.
    std::vector<std::vector<cv::Point2d> > xs;
    for (int i = 0; i < n_views; i++)
    {
        FILE* fin = fopen(cv::format("image_generation%d.xyz", i).c_str(), "rt");
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
    std::vector<int> visible_all(xs.front().size(), 1);
    std::vector<std::vector<int> > visibility(n_views, visible_all);

    // Prepare each camera projection matrix
    std::vector<cv::Mat> Ks, dist_coeffs, Rs, ts;
    cv::Mat K = (cv::Mat_<double>(3, 3) << camera_focal, 0, camera_center.x, 0, camera_focal, camera_center.y, 0, 0, 1);
    Ks.resize(n_views, K);                                      // K for all cameras
    dist_coeffs.resize(n_views, cv::Mat::zeros(5, 1, CV_64F));  // dist_coeff for all cameras
    Rs.push_back(cv::Mat::eye(3, 3, CV_64F));                   // R for the first camera (index: 0)
    ts.push_back(cv::Mat::zeros(3, 1, CV_64F));                 // t for the first camera (index: 0)

    // Esitmate relative pose of the inital two views
    cv::Mat F = cv::findFundamentalMat(xs[0], xs[1], cv::FM_8POINT);
    cv::Mat E = K.t() * F * K;
    cv::Mat R, t;
    cv::recoverPose(E, xs[0], xs[1], K, R, t);
    Rs.push_back(R);                                            // R for the second camera
    ts.push_back(t);                                            // t for the second camera

    // Reconstruct the initial 3D points of 'box.xyz' (triangulation)
    cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat Rt, X;
    cv::hconcat(R, t, Rt);
    cv::Mat P1 = K * Rt;
    cv::triangulatePoints(P0, P1, xs[0], xs[1], X);
    std::vector<cv::Point3d> Xs;
    X.row(0) = X.row(0) / X.row(3);
    X.row(1) = X.row(1) / X.row(3);
    X.row(2) = X.row(2) / X.row(3);
    X.row(3) = 1;
    for (int c = 0; c < X.cols; c++)
        Xs.push_back(cv::Point3d(X.at<double>(0, c), X.at<double>(1, c), X.at<double>(2, c)));

    // Estimate the initial relative pose of other views (PnP)
    for (int i = 2; i < n_views; i++)
    {
        cv::Mat rvec;
        cv::solvePnP(Xs, xs[i], Ks[i], dist_coeffs[i], rvec, t);
        cv::Rodrigues(rvec, R);
        Rs.push_back(R);                                        // R for the third and other cameras
        ts.push_back(t);                                        // t for the third and other cameras
    }

    // Optimize camera pose and 3D points (bundle adjustment)
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

    // Store the 3D points
    FILE* fout = fopen("bundle_adjustment.xyz", "wt");
    if (fout == NULL) return -1;
    for (size_t i = 0; i < Xs.size(); i++)
        fprintf(fout, "%f %f %f\n", Xs[i].x, Xs[i].y, Xs[i].z);
    fclose(fout);
    return 0;
}