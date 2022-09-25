#include "opencv2/opencv.hpp"

int main()
{
    // c.f. You need to run 'image_formation.cpp' to generate point observation.
    const char *input0 = "image_formation0.xyz", *input1 = "image_formation1.xyz";
    double f = 1000, cx = 320, cy = 240;

    // Load 2D points observed from two views
    std::vector<cv::Point2d> points0, points1;
    FILE* fin0 = fopen(input0, "rt");
    FILE* fin1 = fopen(input1, "rt");
    if (fin0 == NULL || fin1 == NULL) return -1;
    while (!feof(fin0) || !feof(fin1))
    {
        double x, y, w;
        if (!feof(fin0) && fscanf(fin0, "%lf %lf %lf", &x, &y, &w) == 3)
            points0.push_back(cv::Point2d(x, y));
        if (!feof(fin1) && fscanf(fin1, "%lf %lf %lf", &x, &y, &w) == 3)
            points1.push_back(cv::Point2d(x, y));
    }
    fclose(fin0);
    fclose(fin1);
    if (points0.size() != points1.size()) return -1;

    // Estimate relative pose of two views
    cv::Mat F = cv::findFundamentalMat(points0, points1, cv::FM_8POINT);
    cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
    cv::Mat E = K.t() * F * K;
    cv::Mat R, t;
    cv::recoverPose(E, points0, points1, K, R, t);

    // Reconstruct 3D points (triangulation)
    cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat Rt, X;
    cv::hconcat(R, t, Rt);
    cv::Mat P1 = K * Rt;
    cv::triangulatePoints(P0, P1, points0, points1, X);
    X.row(0) = X.row(0) / X.row(3);
    X.row(1) = X.row(1) / X.row(3);
    X.row(2) = X.row(2) / X.row(3);
    X.row(3) = 1;

    // Store the 3D points
    FILE* fout = fopen("triangulation.xyz", "wt");
    if (fout == NULL) return -1;
    for (int c = 0; c < X.cols; c++)
        fprintf(fout, "%f %f %f\n", X.at<double>(0, c), X.at<double>(1, c), X.at<double>(2, c));
    fclose(fout);
    return 0;
}
