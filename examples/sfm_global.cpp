#include "sfm.hpp"

std::vector<bool> maskNoisyPoints(std::vector<cv::Point3d>& Xs, const std::vector<std::vector<cv::KeyPoint>>& xs, const std::vector<SFM::Vec9d>& views, const SFM::VisibilityGraph& visibility, double reproj_error2)
{
    std::vector<bool> is_noisy(Xs.size(), false);
    if (reproj_error2 > 0)
    {
        for (auto visible = visibility.begin(); visible != visibility.end(); visible++)
        {
            cv::Point3d& X = Xs[visible->second];
            if (X.z < 0) continue;
            int img_idx = SFM::getCamIdx(visible->first), pt_idx = SFM::getObsIdx(visible->first);
            const cv::Point2d& x = xs[img_idx][pt_idx].pt;
            const SFM::Vec9d& view = views[img_idx];

            // Project the given 'X'
            cv::Vec3d rvec(view[0], view[1], view[2]);
            cv::Matx33d R;
            cv::Rodrigues(rvec, R);
            cv::Point3d X_p = R * X + cv::Point3d(view[3], view[4], view[5]);
            const double &f = view[6], &cx = view[7], &cy = view[8];
            cv::Point2d x_p(f * X_p.x / X_p.z + cx, f * X_p.y / X_p.z + cy);

            // Calculate distance between 'x' and 'x_p'
            cv::Point2d d = x - x_p;
            if (d.x * d.x + d.y * d.y > reproj_error2) is_noisy[visible->second] = true;
        }
    }
    return is_noisy;
}

int main()
{
    const char* input = "../data/relief/%02d.jpg";
    double img_resize = 0.25, f_init = 500, cx_init = -1, cy_init = -1, Z_init = 2, Z_limit = 100, ba_loss_width = 9; // Negative 'loss_width' makes BA not to use a loss function.
    int min_inlier_num = 200, ba_num_iter = 200; // Negative 'ba_num_iter' uses the default value for BA minimization
    bool show_match = false;

    // Load images and extract features
    cv::VideoCapture video;
    if (!video.open(input)) return -1;
    cv::Ptr<cv::FeatureDetector> fdetector = cv::BRISK::create();
    std::vector<std::vector<cv::KeyPoint>> img_keypoint;
    std::vector<cv::Mat> img_set, img_descriptor;
    while (true)
    {
        cv::Mat image;
        video >> image;
        if (image.empty()) break;
        if (img_resize != 1) cv::resize(image, image, cv::Size(), img_resize, img_resize);

        std::vector<cv::KeyPoint> keypoint;
        cv::Mat descriptor;
        fdetector->detectAndCompute(image, cv::Mat(), keypoint, descriptor);
        img_set.push_back(image);
        img_keypoint.push_back(keypoint);
        img_descriptor.push_back(descriptor.clone());
    }
    if (img_set.size() < 2) return -1;
    if (cx_init < 0) cx_init = img_set.front().cols / 2;
    if (cy_init < 0) cy_init = img_set.front().rows / 2;

    // Match features and find good matches
    cv::Ptr<cv::DescriptorMatcher> fmatcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<std::pair<uint, uint>> match_pair;        // Good matches (image pairs)
    std::vector<std::vector<cv::DMatch>> match_inlier;  // Good matches (inlier feature matches)
    for (size_t i = 0; i < img_set.size(); i++)
    {
        for (size_t j = i + 1; j < img_set.size(); j++)
        {
            // Match features of two image pair (i, j) and find their inliers
            std::vector<cv::DMatch> match, inlier;
            fmatcher->match(img_descriptor[i], img_descriptor[j], match);
            std::vector<cv::Point2d> src, dst;
            for (auto itr = match.begin(); itr != match.end(); itr++)
            {
                src.push_back(img_keypoint[i][itr->queryIdx].pt);
                dst.push_back(img_keypoint[j][itr->trainIdx].pt);
            }
            cv::Mat inlier_mask;
            cv::findFundamentalMat(src, dst, inlier_mask, cv::RANSAC);
            for (int k = 0; k < inlier_mask.rows; k++)
                if (inlier_mask.at<uchar>(k)) inlier.push_back(match[k]);
            printf("3DV Tutorial: Image %zd - %zd are matched (%zd / %zd).\n", i, j, inlier.size(), match.size());

            // Determine whether the image pair is good or not
            if (inlier.size() < min_inlier_num) continue;
            printf("3DV Tutorial: Image %zd - %zd are selected.\n", i, j);
            match_pair.push_back(std::make_pair(uint(i), uint(j)));
            match_inlier.push_back(inlier);
            if (show_match)
            {
                cv::Mat match_image;
                cv::drawMatches(img_set[i], img_keypoint[i], img_set[j], img_keypoint[j], match, match_image, cv::Scalar::all(-1), cv::Scalar::all(-1), inlier_mask);
                cv::imshow("3DV Tutorial: Structure-from-Motion", match_image);
                cv::waitKey();
            }
        }
    }
    if (match_pair.size() < 1) return -1;

    // 1) Initialize cameras (rotation, translation, intrinsic parameters)
    std::vector<SFM::Vec9d> cameras(img_set.size(), SFM::Vec9d(0, 0, 0, 0, 0, 0, f_init, cx_init, cy_init));

    // 2) Initialize 3D points and build a visibility graph
    std::vector<cv::Point3d> Xs;
    std::vector<cv::Vec3b> Xs_rgb;
    SFM::VisibilityGraph xs_visited;
    for (size_t m = 0; m < match_pair.size(); m++)
    {
        for (size_t in = 0; in < match_inlier[m].size(); in++)
        {
            const uint &cam1_idx = match_pair[m].first, &cam2_idx = match_pair[m].second;
            const uint &x1_idx = match_inlier[m][in].queryIdx, &x2_idx = match_inlier[m][in].trainIdx;
            const uint key1 = SFM::genKey(cam1_idx, x1_idx), key2 = SFM::genKey(cam2_idx, x2_idx);
            auto visit1 = xs_visited.find(key1), visit2 = xs_visited.find(key2);
            if (visit1 != xs_visited.end() && visit2 != xs_visited.end())
            {
                // Remove previous observations if they are not consistent
                if (visit1->second != visit2->second)
                {
                    xs_visited.erase(visit1);
                    xs_visited.erase(visit2);
                }
                continue; // Skip if two observations are already visited
            }

            uint X_idx = 0;
            if (visit1 != xs_visited.end()) X_idx = visit1->second;
            else if (visit2 != xs_visited.end()) X_idx = visit2->second;
            else
            {
                // Add a new point if two observations are not visited
                X_idx = uint(Xs.size());
                Xs.push_back(cv::Point3d(0, 0, Z_init));
                Xs_rgb.push_back(img_set[cam1_idx].at<cv::Vec3b>(img_keypoint[cam1_idx][x1_idx].pt));
            }
            if (visit1 == xs_visited.end()) xs_visited[key1] = X_idx;
            if (visit2 == xs_visited.end()) xs_visited[key2] = X_idx;
        }
    }
    printf("3DV Tutorial: # of 3D points: %zd\n", Xs.size());

    // 3) Optimize camera pose and 3D points together (bundle adjustment)
    ceres::Problem ba;
    for (auto visit = xs_visited.begin(); visit != xs_visited.end(); visit++)
    {
        int cam_idx = SFM::getCamIdx(visit->first), x_idx = SFM::getObsIdx(visit->first);
        const cv::Point2d& x = img_keypoint[cam_idx][x_idx].pt;
        SFM::addCostFunc6DOF(ba, Xs[visit->second], x, cameras[cam_idx], ba_loss_width);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    if (ba_num_iter > 0) options.max_num_iterations = ba_num_iter;
    options.num_threads = 8;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &ba, &summary);
    std::cout << summary.FullReport() << std::endl;

    // Mark erroneous points to reject them
    std::vector<bool> is_noisy = maskNoisyPoints(Xs, img_keypoint, cameras, xs_visited, ba_loss_width);
    int num_noisy = std::accumulate(is_noisy.begin(), is_noisy.end(), 0);
    printf("3DV Tutorial: # of 3D points: %zd (Rejected: %d)\n", Xs.size(), num_noisy);
    for (size_t j = 0; j < cameras.size(); j++)
        printf("3DV Tutorial: Camera %zd's (f, cx, cy) = (%.3f, %.1f, %.1f)\n", j, cameras[j][6], cameras[j][7], cameras[j][8]);

    // Store the 3D points to an XYZ file
    FILE* fpts = fopen("sfm_global(point).xyz", "wt");
    if (fpts == NULL) return -1;
    for (size_t i = 0; i < Xs.size(); i++)
    {
        if (Xs[i].z > -Z_limit && Xs[i].z < Z_limit && !is_noisy[i])
            fprintf(fpts, "%f %f %f %d %d %d\n", Xs[i].x, Xs[i].y, Xs[i].z, Xs_rgb[i][2], Xs_rgb[i][1], Xs_rgb[i][0]); // Format: x, y, z, R, G, B
    }
    fclose(fpts);

    // Store the camera poses to an XYZ file 
    FILE* fcam = fopen("sfm_global(camera).xyz", "wt");
    if (fcam == NULL) return -1;
    for (size_t j = 0; j < cameras.size(); j++)
    {
        cv::Vec3d rvec(cameras[j][0], cameras[j][1], cameras[j][2]), t(cameras[j][3], cameras[j][4], cameras[j][5]);
        cv::Matx33d R;
        cv::Rodrigues(rvec, R);
        cv::Vec3d p = -R.t() * t;
        fprintf(fcam, "%f %f %f %f %f %f\n", p[0], p[1], p[2], R.t()(0, 2), R.t()(1, 2), R.t()(2, 2)); // Format: x, y, z, n_x, n_y, n_z
    }
    fclose(fcam);
    return 0;
}
