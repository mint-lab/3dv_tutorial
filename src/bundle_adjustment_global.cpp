#include "bundle_adjustment.hpp"

int main()
{
    // c.f. You need to run 'image_formation.cpp' to generate point observation.
    const char* input = "image_formation%d.xyz";
    int input_num = 5;
    double f = 1000, cx = 320, cy= 240;

    // Load 2D points observed from multiple views
    std::vector<std::vector<cv::Point2d> > xs;
    for (int i = 0; i < input_num; i++)
    {
        FILE* fin = fopen(cv::format(input, i).c_str(), "rt");
        if (fin == NULL) return -1;
        std::vector<cv::Point2d> pts;
        while (!feof(fin))
        {
            double x, y, w;
            if (fscanf(fin, "%lf %lf %lf", &x, &y, &w) == 3)
                pts.push_back(cv::Point2d(x, y));
        }
        fclose(fin);
        xs.push_back(pts);
        if (xs.front().size() != xs.back().size()) return -1;
    }

    // Assumption
    // - All cameras have the same and known camera matrix.
    // - All points are visible on all camera views.

    // Initialize camera views and 3D points
    std::vector<cv::Vec6d> views(xs.size());
    std::vector<cv::Point3d> Xs(xs.front().size(), cv::Point3d(0, 0, 5.5));

    // Run bundle adjustment
    ceres::Problem ba;
    for (size_t j = 0; j < xs.size(); j++)
    {
        for (size_t i = 0; i < xs[j].size(); i++)
        {
            ceres::CostFunction* cost_func = ReprojectionError::create(xs[j][i], f, cx, cy);
            double* view = (double*)(&(views[j]));
            double* X = (double*)(&(Xs[i]));
            ba.AddResidualBlock(cost_func, NULL, view, X);
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.num_threads = 8;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &ba, &summary);

    // Store the 3D points to an XYZ file
    FILE* fpts = fopen("bundle_adjustment_global(point).xyz", "wt");
    if (fpts == NULL) return -1;
    for (size_t i = 0; i < Xs.size(); i++)
        fprintf(fpts, "%f %f %f\n", Xs[i].x, Xs[i].y, Xs[i].z);
    fclose(fpts);

    // Store the camera poses to an XYZ file 
    FILE* fcam = fopen("bundle_adjustment_global(camera).xyz", "wt");
    if (fcam == NULL) return -1;
    for (size_t j = 0; j < views.size(); j++)
    {
        cv::Vec3d rvec(views[j][0], views[j][1], views[j][2]), t(views[j][3], views[j][4], views[j][5]);
        cv::Matx33d R;
        cv::Rodrigues(rvec, R);
        cv::Vec3d p = -R.t() * t;
        fprintf(fcam, "%f %f %f\n", p[0], p[1], p[2]);
    }
    fclose(fcam);
    return 0;
}
