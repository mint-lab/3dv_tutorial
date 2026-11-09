#include "opencv2/opencv.hpp"
#include "ceres/ceres.h"
#include <algorithm>

// Convert a line format, [n_x, n_y, x_0, y_0] to [a, b, c]
// Note) A line model in OpenCV: n_y * (x - x_0) = n_x * (y - y_0)
#define CONVERT_LINE(line) (cv::Vec3d(line[1], -line[0], -line[1] * line[2] + line[0] * line[3]))

struct GeometricError
{
    GeometricError(const cv::Point2d& pt) : datum(pt) { }

    template<typename T>
    bool operator()(const T* const line, T* residual) const
    {
        residual[0] = (line[0] * T(datum.x) + line[1] * T(datum.y) + line[2]) / sqrt(line[0] * line[0] + line[1] * line[1]);
        return true;
    }

private:
    const cv::Point2d datum;
};

int main()
{
    cv::Vec3d true_line = cv::Vec3d(2, 3, -14) / sqrt(13.0); // The line model: a*x + b*y + c = 0 (a^2 + b^2 = 1)
    cv::Vec2d data_range(-4, 12);
    int data_num = 100;
    double noise_std = 0.2, outlier_ratio = 0.7;

    double loss_width = 0.3;

    // Generate noisy points with outliers
    auto line2y = [](const cv::Vec3d& line, double x) { return (line[0] * x + line[2]) / -line[1]; }; // ax + by + c = 0 -> y = (ax + c) / -b
    double y0 = line2y(true_line, data_range[0]), y1 = line2y(true_line, data_range[1]);
    std::vector<cv::Point2d> data;
    cv::RNG rng;
    for (int i = 0; i < data_num; i++)
    {
        double x = rng.uniform(data_range[0], data_range[1]), y;
        if (rng.uniform(0.0, 1.0) < outlier_ratio) y = rng.uniform(std::min(y0, y1), std::max(y0, y1)); // Generate an outlier
        else
        {
            y = line2y(true_line, x); // Generate an inlier
            x += rng.gaussian(noise_std); // Add Gaussian noise
            y += rng.gaussian(noise_std);
        }
        data.emplace_back(x, y);
    }

    // Estimate a line using M-estimator
    cv::Vec3d mest_line(1, 1, 0);
    ceres::Problem problem;
    for (size_t i = 0; i < data.size(); i++)
    {
        ceres::CostFunction* cost_func = new ceres::AutoDiffCostFunction<GeometricError, 1, 3>(new GeometricError(data[i]));
        ceres::LossFunction* loss_func = NULL;
        if (loss_width > 0) loss_func = new ceres::HuberLoss(loss_width);
        problem.AddResidualBlock(cost_func, loss_func, mest_line.val);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.num_threads = 8;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    mest_line /= sqrt(mest_line[0] * mest_line[0] + mest_line[1] * mest_line[1]); // Normalize

    // Estimate a line using least squares method (for reference)
    cv::Vec4d nnxy;
    cv::fitLine(data, nnxy, cv::DIST_L2, 0, 0.01, 0.01);
    cv::Vec3d lsqr_line = CONVERT_LINE(nnxy);

    // Display estimates
    printf("* The Truth: %.3f, %.3f, %.3f\n", true_line[0], true_line[1], true_line[2]);
    printf("* Estimate (M-estimator): %.3f, %.3f, %.3f (Cost: %.3f)\n", mest_line[0], mest_line[1], mest_line[2], summary.final_cost / data.size());
    printf("* Estimate (LSM): %.3f, %.3f, %.3f\n", lsqr_line[0], lsqr_line[1], lsqr_line[2]);
    return 0;
}
