#ifndef __SFM__
#define __SFM__

#include "opencv2/opencv.hpp"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

// Reprojection error for bundle adjustment
// - Ref. http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment
struct ReprojectionErrorSnavely
{
    ReprojectionErrorSnavely(const cv::Point2d& _x) : x(_x) { }

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const
    {
        // X' = R*X + t
        T X[3];
        ceres::AngleAxisRotatePoint(camera, point, X);
        X[0] += camera[3];
        X[1] += camera[4];
        X[2] += camera[5];

        // x' = K*X' with radial distortion
        const T& f = camera[6];
        const T& cx = camera[7];
        const T& cy = camera[8];
        const T& k1 = camera[9];
        const T& k2 = camera[10];
        T x_n = X[0] / X[2];
        T y_n = X[1] / X[2];
        T r2 = x_n * x_n + y_n * y_n;
        T radial_distort = T(1.0) + r2 * (k1 + k2 * r2);
        T x_p = f * radial_distort * x_n + cx;
        T y_p = f * radial_distort * y_n + cy;

        // residual = x - x'
        residuals[0] = T(x.x) - x_p;
        residuals[1] = T(x.y) - y_p;
        return true;
    }

    static ceres::CostFunction* create(const cv::Point2d& _x)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorSnavely, 2, 11, 3>(new ReprojectionErrorSnavely(_x)));
    }

private:
    const cv::Point2d x;
};

#endif // End of '__SFM__'
