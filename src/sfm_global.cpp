#include "sfm.hpp"
#include <unordered_map>

#define MAKE_KEY(img_idx, pt_idx)       ((uint(img_idx) << 16) + pt_idx)
#define GET_IMG_IDX(key)                ((key >> 16) & 0xFFFF)
#define GET_PT_IDX(key)                 (key & 0xFFFF)

int main()
{
    const char* input = "data/relief/%02d.jpg";
    double img_resize = 0.25, f_init = 500, Z_init = 5, Z_limit = 10;
    size_t min_inlier_num = 200;
    bool show_match = false;

    // Load images and extract features
    cv::VideoCapture video;
    if (!video.open(input)) return -1;
    cv::Ptr<cv::FeatureDetector> fdetector = cv::BRISK::create();
    std::vector<std::vector<cv::KeyPoint> > img_keypoint;
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
    cv::Point2d img_center(img_set.front().cols / 2, img_set.front().rows / 2);

    // Match features all over the image pairs and find inliers
    cv::Ptr<cv::DescriptorMatcher> fmatcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<std::pair<int, int> > match_pair;           // The viewing graph (image pairs)
    std::vector<std::vector<cv::DMatch> > match_inlier;     // The viewing graph (indices of inlier matches)
    for (size_t i = 0; i < img_set.size(); i++)
    {
        for (size_t j = i + 1; j < img_set.size(); j++)
        {
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
            printf("3DV Tutorial: Image %d - %d are matched (%d / %d).\n", i, j, inlier.size(), match.size());
            if (inlier.size() < min_inlier_num) continue; // The threshold for putting into the viewing graph
            printf("3DV Tutorial: Image %d - %d are selected.\n", i, j);
            match_pair.push_back(std::pair<int, int>(i, j));
            match_inlier.push_back(inlier);
            if (show_match)
            {
                cv::Mat match_image;
                cv::drawMatches(img_set[i], img_keypoint[i], img_set[j], img_keypoint[j], match, match_image, cv::Vec3b(0, 0, 255), cv::Vec3b(0, 127, 0), inlier_mask);
                cv::imshow("3DV Tutorial: Global Structure-from-Motion", match_image);
                cv::waitKey();
            }
        }
    }
    if (match_pair.size() < 1) return -1;

    // Initialize camera views
    typedef cv::Vec<double, 11> Vec11d;
    std::vector<Vec11d> views(img_set.size(), Vec11d(0, 0, 0, 0, 0, 0, f_init, img_center.x, img_center.y));

    // Initialize 3D points and build a visibility graph
    std::vector<cv::Point3d> Xs;
    std::unordered_map<uint, uint> xs_visited;
    for (size_t m = 0; m < match_pair.size(); m++)
    {
        for (size_t in = 0; in < match_inlier[m].size(); in++)
        {
            const int &img1 = match_pair[m].first, &img2 = match_pair[m].second, &x1_idx = match_inlier[m][in].queryIdx, &x2_idx = match_inlier[m][in].trainIdx;
            const uint key1 = MAKE_KEY(img1, x1_idx), key2 = MAKE_KEY(img2, x2_idx);
            auto visit1 = xs_visited.find(key1), visit2 = xs_visited.find(key2);
            if (visit1 != xs_visited.end() && visit2 != xs_visited.end())
            {
                if (visit1->second != visit2->second)
                {
                    xs_visited.erase(visit1);
                    xs_visited.erase(visit2);
                }
                // Skip if two observations are already visited
                continue;
            }

            uint X_idx = 0;
            if (visit1 != xs_visited.end()) X_idx = visit1->second;
            else if (visit2 != xs_visited.end()) X_idx = visit2->second;
            else
            {
                // Add a new point if two observations are not visited
                X_idx = Xs.size();
                Xs.push_back(cv::Point3d(0, 0, Z_init));
            }
            if (visit1 == xs_visited.end()) xs_visited[key1] = X_idx;
            if (visit2 == xs_visited.end()) xs_visited[key2] = X_idx;
        }
    }

    // Run bundle adjustment
    ceres::Problem ba;
    for (auto visit = xs_visited.begin(); visit != xs_visited.end(); visit++)
    {
        int img_idx = GET_IMG_IDX(visit->first), pt_idx = GET_PT_IDX(visit->first);
        const cv::Point2d& x = img_keypoint[img_idx][pt_idx].pt;
        ceres::CostFunction* cost_func = ReprojectionErrorSnavely::create(x);
        double* view = (double*)(&(views[img_idx]));
        double* X = (double*)(&(Xs[visit->second]));
        ba.AddResidualBlock(cost_func, NULL, view, X);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.max_num_iterations = 200;
    options.num_threads = 8;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &ba, &summary);
    std::cout << summary.FullReport() << std::endl;

    // Store the 3D points to an XYZ file
    FILE* fpts = fopen("sfm_global(point).xyz", "wt");
    if (fpts == NULL) return -1;
    int Xs_reject = 0;
    for (size_t i = 0; i < Xs.size(); i++)
    {
        if (Xs[i].z > 0 && Xs[i].z < Z_limit)
            fprintf(fpts, "%f %f %f\n", Xs[i].x, Xs[i].y, Xs[i].z);
        else Xs_reject++;
    }
    fclose(fpts);
    printf("3DV Tutorial: # of 3D points = %d (# of rejected = %d).\n", Xs.size(), Xs_reject);

    // Store the camera poses to an XYZ file 
    FILE* fcam = fopen("sfm_global(camera).xyz", "wt");
    if (fcam == NULL) return -1;
    for (size_t j = 0; j < views.size(); j++)
    {
        cv::Vec3d rvec(views[j][0], views[j][1], views[j][2]), t(views[j][3], views[j][4], views[j][5]);
        cv::Matx33d R;
        cv::Rodrigues(rvec, R);
        cv::Vec3d p = -R.t() * t;
        fprintf(fcam, "%f %f %f\n", p[0], p[1], p[2]);
        printf("3DV Tutorial: Image %d's (f, cx, cy) = (%.3f, %.1f, %.1f), (k1, k2) = (%.3f, %.3f)\n", j, views[j][6], views[j][7], views[j][8], views[j][9], views[j][10]);
    }
    fclose(fcam);
    return 0;
}
