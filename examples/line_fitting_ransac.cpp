#include "opencv2/opencv.hpp"
#include <algorithm>

// Convert a line format, [n_x, n_y, x_0, y_0] to [a, b, c]
// Note) A line model in OpenCV: n_y * (x - x_0) = n_x * (y - y_0)
#define CONVERT_LINE(line) (cv::Vec3d(line[1], -line[0], -line[1] * line[2] + line[0] * line[3]))

int main()
{
    cv::Vec3d true_line = cv::Vec3d(2, 3, -14) / sqrt(13.0); // The line model: a*x + b*y + c = 0 (a^2 + b^2 = 1)
    cv::Vec2d data_range(-4, 12);
    int data_num = 100;
    double noise_std = 0.2, outlier_ratio = 0.7;

    int ransac_trial = 100, n_sample = 2;
    double ransac_threshold = 0.3;

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

    // Estimate a line using RANSAC
    int best_score = -1;
    cv::Vec3d best_line;
    for (int i = 0; i < ransac_trial; i++)
    {
        // Step 1: Hypothesis generation
        std::vector<cv::Point2d> sample;
        for (int j = 0; j < n_sample; j++)
        {
            int index = rng.uniform(0, int(data.size()));
            while (std::find(sample.begin(), sample.end(), data[index]) != sample.end())
                index = rng.uniform(0, int(data.size()));
            sample.push_back(data[index]);
        }
        cv::Vec4d nnxy;
        cv::fitLine(sample, nnxy, cv::DIST_L2, 0, 0.01, 0.01);
        cv::Vec3d line = CONVERT_LINE(nnxy);

        // Step 2: Hypothesis evaluation
        int score = 0;
        for (size_t j = 0; j < data.size(); j++)
        {
            double error = fabs(line(0) * data[j].x + line(1) * data[j].y + line(2));
            if (error < ransac_threshold) score++;
        }

        if (score > best_score)
        {
            best_score = score;
            best_line = line;
        }
    }

    // Estimate a line using least squares method (for reference)
    cv::Vec4d nnxy;
    cv::fitLine(data, nnxy, cv::DIST_L2, 0, 0.01, 0.01);
    cv::Vec3d lsqr_line = CONVERT_LINE(nnxy);

    // Display estimates
    printf("* The Truth: %.3f, %.3f, %.3f\n", true_line[0], true_line[1], true_line[2]);
    printf("* Estimate (RANSAC): %.3f, %.3f, %.3f (Score: %d)\n", best_line[0], best_line[1], best_line[2], best_score);
    printf("* Estimate (LSM): %.3f, %.3f, %.3f\n", lsqr_line[0], lsqr_line[1], lsqr_line[2]);
    return 0;
}
