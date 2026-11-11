#ifndef __SFM__
#define __SFM__

#include "opencv2/opencv.hpp"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "bundle_adjustment.hpp"
#include <unordered_map>
#include <cmath>

// Reprojection error for bundle adjustment with 7 DOF cameras
// - 7 DOF = 3 DOF rotation + 3 DOF translation + 1 DOF focal length
struct ReprojectionError7DOF
{
    ReprojectionError7DOF(const cv::Point2d& _x, const cv::Point2d& _c) : x(_x), c(_c) { }

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
        const T& f = camera[6];
        T x_p = f * X[0] / X[2] + c.x;
        T y_p = f * X[1] / X[2] + c.y;

        // residual = x - x'
        residuals[0] = T(x.x) - x_p;
        residuals[1] = T(x.y) - y_p;
        return true;
    }

    static ceres::CostFunction* create(const cv::Point2d& _x, const cv::Point2d& _c)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError7DOF, 2, 7, 3>(new ReprojectionError7DOF(_x, _c)));
    }

private:
    const cv::Point2d x;
    const cv::Point2d c;
};

class SFM
{
public:

    typedef cv::Vec<double, 9> Vec9d;

    typedef std::unordered_map<uint, uint> VisibilityGraph;

    static inline uint gen_key(uint cam_idx, uint obs_idx) { return ((cam_idx << 16) + obs_idx); }

    static inline uint get_cam_idx(uint key) { return ((key >> 16) & 0xFFFF); }

    static inline uint get_obs_idx(uint key) { return (key & 0xFFFF); }

    // Return a boolean mask identifying points behind a camera or with a large reprojection error.
    static std::vector<bool> find_outlier_points(const std::vector<cv::Point3d>& Xs, const std::vector<std::vector<cv::KeyPoint>>& xs, const std::vector<Vec9d>& cameras, const VisibilityGraph& visibility, double reproj_error)
    {
        std::vector<bool> is_noisy(Xs.size(), false);
        for (auto visible = visibility.begin(); visible != visibility.end(); visible++)
        {
            const cv::Point3d& X = Xs[visible->second];
            int cam_idx = get_cam_idx(visible->first), x_idx = get_obs_idx(visible->first);
            const cv::Point2d& x = xs[cam_idx][x_idx].pt;
            const Vec9d& camera = cameras[cam_idx];

            // X' = R*X + t
            cv::Vec3d rvec(camera[0], camera[1], camera[2]);
            cv::Matx33d R;
            cv::Rodrigues(rvec, R);
            cv::Point3d X_p = R * X + cv::Point3d(camera[3], camera[4], camera[5]);
            if (!std::isfinite(X_p.x) || !std::isfinite(X_p.y) || !std::isfinite(X_p.z) || X_p.z <= 0)
            {
                is_noisy[visible->second] = true;
                continue;
            }

            // x' = K*X'
            const double &f = camera[6], &cx = camera[7], &cy = camera[8];
            cv::Point2d x_p(f * X_p.x / X_p.z + cx, f * X_p.y / X_p.z + cy);

            // Reject the point if any observation exceeds the threshold (pixels).
            cv::Point2d d = x - x_p;
            if (reproj_error > 0 && d.dot(d) > reproj_error * reproj_error) is_noisy[visible->second] = true;
        }
        return is_noisy;
    }

    static bool add_cost_func_7dof(ceres::Problem& problem, const cv::Point3d& X, const cv::Point2d& x, const Vec9d& camera, double loss_width = -1)
    {
        double* _X = (double*)(&(X.x));
        double* _camera = (double*)(&(camera[0]));
        ceres::CostFunction* cost_func = ReprojectionError7DOF::create(x, cv::Point2d(camera[7], camera[8]));
        ceres::LossFunction* loss_func = NULL;
        if (loss_width > 0) loss_func = new ceres::CauchyLoss(loss_width);
        problem.AddResidualBlock(cost_func, loss_func, _camera, _X);
        return true;
    }

    static bool add_cost_func_6dof(ceres::Problem& problem, const cv::Point3d& X, const cv::Point2d& x, const Vec9d& camera, double loss_width = -1)
    {
        double* _X = (double*)(&(X.x));
        double* _camera = (double*)(&(camera[0]));
        ceres::CostFunction* cost_func = ReprojectionError::create(x, camera[6], cv::Point2d(camera[7], camera[8]));
        ceres::LossFunction* loss_func = NULL;
        if (loss_width > 0) loss_func = new ceres::CauchyLoss(loss_width);
        problem.AddResidualBlock(cost_func, loss_func, _camera, _X);
        return true;
    }
};

#endif // End of '__SFM__'
