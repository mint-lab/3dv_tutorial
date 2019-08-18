#include "opencv2/opencv.hpp"
#include "ceres/ceres.h"

// Convert a line format, [n_x, n_y, x_0, y_0] to [a, b, c]
// c.f. A line model in OpenCV: n_x * (x - x_0) = n_y * (y - y_0)
#define CONVERT_LINE(line) (cv::Vec3d(line[0], -line[1], -line[0] * line[2] + line[1] * line[3]))

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
    cv::Vec3d truth(1.0 / sqrt(2.0), 1.0 / sqrt(2.0), -240.0); // The line model: a*x + b*y + c = 0 (a^2 + b^2 = 1)
    double loss_width = 3.0; // 3 x 'data_inlier_noise'; if this value is less than equal to 0, M-estimator is disabled.
    int data_num = 1000;
    double data_inlier_ratio = 0.5, data_inlier_noise = 1.0;

    // Generate data
    std::vector<cv::Point2d> data;
    cv::RNG rng;
    for (int i = 0; i < data_num; i++)
    {
        if (rng.uniform(0.0, 1.0) < data_inlier_ratio)
        {
            double x = rng.uniform(0.0, 480.0);
            double y = (truth[0] * x + truth[2]) / -truth[1];
            x += rng.gaussian(data_inlier_noise);
            y += rng.gaussian(data_inlier_noise);
            data.push_back(cv::Point2d(x, y)); // Inlier
        }
        else data.push_back(cv::Point2d(rng.uniform(0.0, 640.0), rng.uniform(0.0, 480.0))); // Outlier
    }

    // Estimate a line using M-estimator
    cv::Vec3d opt_line(1, 0, 0);
    ceres::Problem problem;
    for (size_t i = 0; i < data.size(); i++)
    {
        ceres::CostFunction* cost_func = new ceres::AutoDiffCostFunction<GeometricError, 1, 3>(new GeometricError(data[i]));
        ceres::LossFunction* loss_func = NULL;
        if (loss_width > 0) loss_func = new ceres::CauchyLoss(loss_width);
        problem.AddResidualBlock(cost_func, loss_func, opt_line.val);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.num_threads = 8;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    opt_line /= sqrt(opt_line[0] * opt_line[0] + opt_line[1] * opt_line[1]); // Normalize

    // Estimate a line using least squares method (for reference)
    cv::Vec4d nnxy;
    cv::fitLine(data, nnxy, cv::DIST_L2, 0, 0.01, 0.01);
    cv::Vec3d lsm_line = CONVERT_LINE(nnxy);

    // Display estimates
    printf("* The Truth: %.3f, %.3f, %.3f\n", truth[0], truth[1], truth[2]);
    printf("* Estimate (M-estimator): %.3f, %.3f, %.3f (Cost: %.3f)\n", opt_line[0], opt_line[1], opt_line[2], summary.final_cost / data.size());
    printf("* Estimate (LSM): %.3f, %.3f, %.3f\n", lsm_line[0], lsm_line[1], lsm_line[2]);
    return 0;
}
