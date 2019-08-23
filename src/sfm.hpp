#ifndef __SFM__
#define __SFM__

#include "opencv2/opencv.hpp"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <unordered_map>

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

class SFM
{
public:

    typedef cv::Vec<double, 11> Vec11d;

    typedef std::unordered_map<uint, uint> VisibilityGraph;

    static inline uint genKey(uint img_idx, uint pt_idx) { return ((img_idx << 16) + pt_idx); }

    static inline uint getImIdx(uint key) { return ((key >> 16) & 0xFFFF); }

    static inline uint getPtIdx(uint key) { return (key & 0xFFFF); }

    static bool addCostFunc(ceres::Problem& problem, const std::vector<cv::Point3d>& Xs, const std::vector<std::vector<cv::KeyPoint>>& xs, const std::vector<Vec11d>& views, const std::unordered_map<uint, uint>& visibility, double loss_width = 4)
    {
        for (auto visible = visibility.begin(); visible != visibility.end(); visible++)
        {
            int img_idx = getImIdx(visible->first), pt_idx = getPtIdx(visible->first);
            const cv::Point2d& x = xs[img_idx][pt_idx].pt;
            ceres::CostFunction* cost_func = ReprojectionErrorSnavely::create(x);
            ceres::LossFunction* loss_func = NULL;
            if (loss_width > 0) loss_func = new ceres::CauchyLoss(loss_width);
            double* view = (double*)(&(views[img_idx]));
            double* X = (double*)(&(Xs[visible->second]));
            problem.AddResidualBlock(cost_func, loss_func, view, X);
        }
        return true;
    }

    static int markNoisyPoints(std::vector<cv::Point3d>& Xs, const std::vector<std::vector<cv::KeyPoint>>& xs, const std::vector<Vec11d>& views, const std::unordered_map<uint, uint>& visibility, double reproj_error2 = 4)
    {
        if (reproj_error2 <= 0) return -1;

        int n_mark = 0;
        for (auto visible = visibility.begin(); visible != visibility.end(); visible++)
        {
            cv::Point3d& X = Xs[visible->second];
            if (X.z < 0) continue;
            int img_idx = getImIdx(visible->first), pt_idx = getPtIdx(visible->first);
            const cv::Point2d& x = xs[img_idx][pt_idx].pt;
            const Vec11d& view = views[img_idx];

            // Project the given 'X'
            cv::Vec3d rvec(view[0], view[1], view[2]);
            cv::Matx33d R;
            cv::Rodrigues(rvec, R);
            cv::Point3d X_p = R * X + cv::Point3d(view[3], view[4], view[5]);
            const double &f = view[6], &k1 = view[9], &k2 = view[10];
            cv::Point2d x_n(X_p.x / X_p.z, X_p.y / X_p.z), c(view[7], view[8]);
            double r2 = x_n.x * x_n.x + x_n.y * x_n.y;
            double radial_distort = 1 + r2 * (k1 + k2 * r2);
            cv::Point2d x_p = f * radial_distort * x_n + c;

            // Calculate distance between 'x' and 'x_p'
            cv::Point2d d = x - x_p;
            if (d.x * d.x + d.y * d.y > reproj_error2)
            {
                X.z *= -1;
                n_mark++;
            }
        }
        return n_mark;
    }
};

#endif // End of '__SFM__'
