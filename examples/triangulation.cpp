#include "opencv2/opencv.hpp"

int main()
{
    double f = 1000, cx = 320, cy = 240;
    const char *pts0_file = "../data/image_formation0.xyz", *pts1_file = "../data/image_formation1.xyz";
    const char *output_file = "triangulation.xyz";

    // Load 2D points observed from two views
    std::vector<cv::Point2d> pts0, pts1;
    FILE* fin0 = fopen(pts0_file, "rt");
    FILE* fin1 = fopen(pts1_file, "rt");
    if (fin0 == NULL || fin1 == NULL) return -1;
    while (!feof(fin0) || !feof(fin1))
    {
        double x, y, w;
        if (!feof(fin0) && fscanf(fin0, "%lf %lf %lf", &x, &y, &w) == 3)
            pts0.push_back(cv::Point2d(x, y));
        if (!feof(fin1) && fscanf(fin1, "%lf %lf %lf", &x, &y, &w) == 3)
            pts1.push_back(cv::Point2d(x, y));
    }
    fclose(fin0);
    fclose(fin1);
    if (pts0.size() != pts1.size()) return -1;

    // Estimate relative pose of two views
    cv::Mat F = cv::findFundamentalMat(pts0, pts1, cv::FM_8POINT);
    cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
    cv::Mat E = K.t() * F * K;
    cv::Mat R, t;
    cv::recoverPose(E, pts0, pts1, K, R, t);

    // Reconstruct 3D points (triangulation)
    cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat Rt, X;
    cv::hconcat(R, t, Rt);
    cv::Mat P1 = K * Rt;
    cv::triangulatePoints(P0, P1, pts0, pts1, X);
    X.row(0) = X.row(0) / X.row(3);
    X.row(1) = X.row(1) / X.row(3);
    X.row(2) = X.row(2) / X.row(3);
    X.row(3) = 1;

    // Store the 3D points
    FILE* fout = fopen(output_file, "wt");
    if (fout == NULL) return -1;
    for (int c = 0; c < X.cols; c++)
        fprintf(fout, "%f %f %f\n", X.at<double>(0, c), X.at<double>(1, c), X.at<double>(2, c));
    fclose(fout);
    return 0;
}
