#include "opencv_all.hpp"

// Convert a line format, [n_x, n_y, x, y] to [a, b, c]
#define CONVERT_LINE(line) (cv::Vec3d(line(0), -line(1), -line(0) * line(2) + line(1) * line(3)))

int main(void)
{
    int ransac_trial = 50;
    double ransac_thresh = 3.0;
    int ransac_n_sample = 2;
    int sim_n_data = 1000;
    double sim_inlier_ratio = 0.5, sim_inlier_noise = 1.0;
    cv::Vec3d truth(1.0 / sqrt(2.0), 1.0 / sqrt(2.0), -240.0); // The line model: a*x + b*y + c = 0

    // Generate data
    std::vector<cv::Point2d> data;
    cv::RNG rng;
    for (int i = 0; i < sim_n_data; i++)
    {
        if (rng.uniform(0.0, 1.0) < sim_inlier_ratio)
        {
            double x = rng.uniform(0.0, 480.0);
            double y = (truth(0) * x + truth(2)) / -truth(1);
            x += rng.gaussian(sim_inlier_noise);
            y += rng.gaussian(sim_inlier_noise);
            data.push_back(cv::Point2d(x, y)); // Inlier
        }
        else data.push_back(cv::Point2d(rng.uniform(0.0, 640.0), rng.uniform(0.0, 480.0))); // Outlier
    }

    // Estimate a line using RANSAC
    int best_score = -1;
    cv::Vec3d best_line;
    for (int i = 0; i < ransac_trial; i++)
    {
        // Step 1: Hypothesis generation
        std::vector<cv::Point2d> sample;
        for (int j = 1; j < ransac_n_sample; j++)
        {
            int index = rng.uniform(0, data.size());
            sample.push_back(data[index]);
        }
        cv::Vec4d vvxy;
        cv::fitLine(sample, vvxy, cv::DIST_L2, 0, 0.01, 0.01);
        cv::Vec3d line = CONVERT_LINE(vvxy);

        // Step 2: Hypothesis evaluation
        int score = 0;
        for (size_t j = 0; j < data.size(); j++)
        {
            double error = fabs(line(0) * data[j].x + line(1) * data[j].y + line(2));
            if (error < ransac_thresh) score++;
        }

        if (score > best_score)
        {
            best_score = score;
            best_line = line;
        }
    }

    // Estimate a line using least-squares method (for reference)
    cv::Vec4d vvxy;
    cv::fitLine(data, vvxy, cv::DIST_L2, 0, 0.01, 0.01);
    cv::Vec3d lsm_line = CONVERT_LINE(vvxy);

    // Display estimates
    printf("* The Truth: %.3f, %.3f, %.3f\n", truth(0), truth(1), truth(2));
    printf("* Estimate (RANSAC): %.3f, %.3f, %.3f (Score: %d)\n", best_line(0), best_line(1), best_line(2), best_score);
    printf("* Estimate (LSM): %.3f, %.3f, %.3f\n", lsm_line(0), lsm_line(1), lsm_line(2));
    return 0;
}
