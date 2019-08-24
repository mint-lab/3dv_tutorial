#ifndef __BUNDLE_ADJUSTMENT__
#define __BUNDLE_ADJUSTMENT__

#include "opencv2/opencv.hpp"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

// Reprojection error for bundle adjustment
struct ReprojectionError
{
    ReprojectionError(const cv::Point2d& _x, double _f, double _cx, double _cy) : x(_x), f(_f), cx(_cx), cy(_cy) { }

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const
    {
        // X' = R*X + t
        T X[3];
        ceres::AngleAxisRotatePoint(camera, point, X);
        X[0] += camera[3];
        X[1] += camera[4];
        X[2] += camera[5];

        // x' = K*X'
        T x_p = f * X[0] / X[2] + cx;
        T y_p = f * X[1] / X[2] + cy;

        // residual = x - x'
        residuals[0] = T(x.x) - x_p;
        residuals[1] = T(x.y) - y_p;
        return true;
    }

    static ceres::CostFunction* create(const cv::Point2d& _x, double _f, double _cx, double _cy)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(new ReprojectionError(_x, _f, _cx, _cy)));
    }

private:
    const cv::Point2d x;
    const double f;
    const double cx;
    const double cy;
};

#endif // End of '__BUNDLE_ADJUSTMENT__'
