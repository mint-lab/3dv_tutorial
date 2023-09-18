#include "sfm.hpp"
#include <unordered_set>

cv::Mat getCameraMat(const SFM::Vec9d& camera)
{
    const double &f = camera[6], &cx = camera[7], &cy = camera[8];
    cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
    return K;
}

cv::Mat getProjectionMat(const SFM::Vec9d& camera)
{
    cv::Mat K = getCameraMat(camera);
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << camera[0], camera[1], camera[2]), R, Rt;
    cv::Mat t = (cv::Mat_<double>(3, 1) << camera[3], camera[4], camera[5]);
    cv::Rodrigues(rvec, R);
    cv::hconcat(R, t, Rt);
    return K * Rt;
}

void updateCameraPose(SFM::Vec9d& camera, const cv::Mat& R, const cv::Mat& t)
{
    cv::Mat rvec = R.clone();
    if (rvec.cols == 3 && rvec.rows == 3) cv::Rodrigues(rvec, rvec);
    camera[0] = rvec.at<double>(0);
    camera[1] = rvec.at<double>(1);
    camera[2] = rvec.at<double>(2);
    camera[3] = t.at<double>(0);
    camera[4] = t.at<double>(1);
    camera[5] = t.at<double>(2);
}

bool isBadPoint(const cv::Point3d& X, const SFM::Vec9d& camera1, const SFM::Vec9d& camera2, double Z_limit, double max_cos_parallax)
{
    if (X.z < -Z_limit || X.z > Z_limit) return true;   // BAD! If the point is too far from the origin.
    cv::Vec3d rvec1(camera1[0], camera1[1], camera1[2]), rvec2(camera2[0], camera2[1], camera2[2]);
    cv::Matx33d R1, R2;
    cv::Rodrigues(rvec1, R1);
    cv::Rodrigues(rvec2, R2);
    cv::Point3d t1(camera1[3], camera1[4], camera1[5]), t2(camera2[3], camera2[4], camera2[5]);
    cv::Point3d p1 = R1 * X + t1;                       // A 3D point w.r.t. the 1st camera coordinate
    cv::Point3d p2 = R2 * X + t2;                       // A 3D point w.r.t. the 2nd camera coordinate
    if (p1.z <= 0 || p2.z <= 0) return true;            // BAD! If the point is beyond of one of 1st and 2nd cameras.
    cv::Point3d v2 = R1 * R2.t() * p2;                  // A 3D vector 'p2' w.r.t. the 1st camera coordinate
    double cos_parallax = p1.dot(v2) / cv::norm(p1) / cv::norm(v2);
    if (cos_parallax > max_cos_parallax) return true;   // BAD! If the point has small parallax angle.
    return false;
}

int main()
{
    const char* input = "../data/relief/%02d.jpg";
    double img_resize = 0.25, f_init = 500, cx_init = -1, cy_init = -1, Z_limit = 100, max_cos_parallax = cos(10 * CV_PI / 180), ba_loss_width = 9; // Negative 'loss_width' makes BA not to use a loss function.
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

    // Initialize cameras (rotation, translation, intrinsic parameters)
    std::vector<SFM::Vec9d> cameras(img_set.size(), SFM::Vec9d(0, 0, 0, 0, 0, 0, f_init, cx_init, cy_init));

    uint best_pair = 0;
    std::vector<uint> best_score(match_inlier.size());
    for (size_t i = 0; i < match_inlier.size(); i++)
        best_score[i] = uint(match_inlier[i].size());
    cv::Mat best_Xs;
    while (true)
    {
        // 1) Select the best pair
        for (size_t i = 0; i < best_score.size(); i++)
            if (best_score[i] > best_score[best_pair]) best_pair = uint(i);
        if (best_score[best_pair] == 0)
        {
            printf("3DV Tutorial: There is no good match. Try again after reducing 'max_cos_parallax'.\n");
            return -1;
        }
        const uint best_cam0 = match_pair[best_pair].first, best_cam1 = match_pair[best_pair].second;;

        // 2) Estimate relative pose from the best two views (epipolar geometry)
        std::vector<cv::Point2d> src, dst;
        for (auto itr = match_inlier[best_pair].begin(); itr != match_inlier[best_pair].end(); itr++)
        {
            src.push_back(img_keypoint[best_cam0][itr->queryIdx].pt);
            dst.push_back(img_keypoint[best_cam1][itr->trainIdx].pt);
        }
        cv::Mat K = getCameraMat(cameras[best_cam0]), R, t, inlier_mask;
        cv::Mat E = cv::findEssentialMat(src, dst, K, cv::RANSAC, 0.999, 1, inlier_mask);
        cv::recoverPose(E, src, dst, K, R, t, inlier_mask);
        for (int r = inlier_mask.rows - 1; r >= 0; r--)
        {
            if (!inlier_mask.at<uchar>(r))
            {
                // Remove additionally detected outliers
                src.erase(src.begin() + r);
                dst.erase(dst.begin() + r);
                match_inlier[best_pair].erase(match_inlier[best_pair].begin() + r);
            }
        }
        updateCameraPose(cameras[best_cam1], R, t);

        // 3) Reconstruct 3D points of the best two views (triangulation)
        cv::Mat P0 = getProjectionMat(cameras[best_cam0]), P1 = getProjectionMat(cameras[best_cam1]);
        cv::triangulatePoints(P0, P1, src, dst, best_Xs);
        best_Xs.row(0) = best_Xs.row(0) / best_Xs.row(3);
        best_Xs.row(1) = best_Xs.row(1) / best_Xs.row(3);
        best_Xs.row(2) = best_Xs.row(2) / best_Xs.row(3);

        best_score[best_pair] = 0;
        for (int i = 0; i < best_Xs.cols; i++)
        {
            cv::Point3d p(best_Xs.col(i).rowRange(0, 3)); // A 3D point at 'idx'
            if (isBadPoint(p, cameras[best_cam0], cameras[best_cam1], Z_limit, max_cos_parallax)) continue; // Do not add if it is bad
            best_score[best_pair]++;
        }
        printf("3DV Tutorial: Image %u - %u were checked as the best match (# of inliers = %zd, # of good points = %d).\n", best_cam0, best_cam1, match_inlier[best_pair].size(), best_score[best_pair]);
        if (best_score[best_pair] > 100) break;
        best_score[best_pair] = 0;
    } // End of the 1st 'while (true)'
    const uint best_cam0 = match_pair[best_pair].first, best_cam1 = match_pair[best_pair].second;;

    // Prepare the initial 3D points
    std::vector<cv::Point3d> Xs;
    Xs.reserve(10000); // Allocate memory in advance not to break pointer access in Ceres Solver
    std::vector<cv::Vec3b> Xs_rgb;
    SFM::VisibilityGraph xs_visited;
    for (int i = 0; i < best_Xs.cols; i++)
    {
        cv::Point3d p(best_Xs.col(i).rowRange(0, 3)); // A 3D point at 'idx'
        if (isBadPoint(p, cameras[best_cam0], cameras[best_cam1], Z_limit, max_cos_parallax)) continue; // Do not add if it is bad
        uint X_idx = uint(Xs.size()), x0_idx = match_inlier[best_pair][i].queryIdx, x1_idx = match_inlier[best_pair][i].trainIdx;
        Xs.push_back(p);
        Xs_rgb.push_back(img_set[best_cam0].at<cv::Vec3b>(img_keypoint[best_cam0][x0_idx].pt));
        xs_visited[SFM::genKey(best_cam0, x0_idx)] = X_idx;
        xs_visited[SFM::genKey(best_cam1, x1_idx)] = X_idx;
    }
    std::unordered_set<uint> img_added;
    img_added.insert(best_cam0);
    img_added.insert(best_cam1);
    printf("3DV Tutorial: Image %d - %d are complete (# of 3D points = %zd).\n", best_cam0, best_cam1, Xs.size());

    // Prepare bundle adjustment
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

    while (true)
    {
        // 4) Select the next image to add
        std::vector<uint> img_score(img_set.size(), 0);
        std::vector<std::vector<uint>> match_table(img_set.size());
        for (size_t img = 0; img < img_score.size(); img++)
        {
            if (img_added.find(uint(img)) == img_added.end())                                                           // When the image is not added to the viewing graph
            {
                for (size_t i = 0; i < match_pair.size(); i++)
                {
                    if (match_pair[i].first == img && img_added.find(match_pair[i].second) != img_added.end())          // When 'first' is the current image and 'second' is already added
                    {
                        for (auto itr = match_inlier[i].begin(); itr != match_inlier[i].end(); itr++)
                            if (xs_visited.find(SFM::genKey(match_pair[i].second, itr->trainIdx)) != xs_visited.end())  // When a matched inlier is in 'Xs', the current image gains more score
                                img_score[img]++;
                        match_table[img].push_back(uint(i));
                    }
                    else if (match_pair[i].second == img && img_added.find(match_pair[i].first) != img_added.end())     // When 'second' is the current image and 'first' is already added
                    {
                        for (auto itr = match_inlier[i].begin(); itr != match_inlier[i].end(); itr++)
                            if (xs_visited.find(SFM::genKey(match_pair[i].first, itr->queryIdx)) != xs_visited.end())   // When a matched inlier is in 'Xs', the current image gains more score
                                img_score[img]++;
                        match_table[img].push_back(uint(i));
                    }
                }
            }
        }
        const auto next_score = std::max_element(img_score.begin(), img_score.end());
        const uint next_cam = static_cast<uint>(std::distance(img_score.begin(), next_score));
        const std::vector<uint> next_match = match_table[next_cam];
        if (next_match.empty()) break;

        // Separate points into known (pts_*) and unknown (new_*) for PnP (known) and triangulation (unknown)
        std::vector<cv::Point3d> pts_3d;
        std::vector<cv::Point2d> pts_2d;
        std::vector<uint> pts_key, pts_idx, new_pair_cam;
        std::vector<std::vector<cv::Point2d>> new_next_2d(next_match.size()), new_pair_2d(next_match.size());
        std::vector<std::vector<uint>> new_next_key(next_match.size()), new_pair_key(next_match.size());
        for (size_t i = 0; i < next_match.size(); i++)
        {
            bool next_is_first = true;
            int pair_cam = match_pair[next_match[i]].second;
            if (pair_cam == next_cam)
            {
                next_is_first = false;
                pair_cam = match_pair[next_match[i]].first;
            }
            new_pair_cam.push_back(pair_cam);
            for (auto itr = match_inlier[next_match[i]].begin(); itr != match_inlier[next_match[i]].end(); itr++)
            {
                int next_idx = itr->queryIdx, pair_idx = itr->trainIdx;
                if (!next_is_first)
                {
                    next_idx = itr->trainIdx;
                    pair_idx = itr->queryIdx;
                }
                auto found = xs_visited.find(SFM::genKey(pair_cam, pair_idx));
                if (found != xs_visited.end())  // When the matched point is already known (--> PnP)
                {
                    pts_3d.push_back(Xs[found->second]);
                    pts_2d.push_back(img_keypoint[next_cam][next_idx].pt);
                    pts_key.push_back(SFM::genKey(next_cam, next_idx));
                    pts_idx.push_back(found->second);
                }
                else                            // When the matched point is newly observed (--> triangulation)
                {
                    new_next_2d[i].push_back(img_keypoint[next_cam][next_idx].pt);
                    new_pair_2d[i].push_back(img_keypoint[pair_cam][pair_idx].pt);
                    new_next_key[i].push_back(SFM::genKey(next_cam, next_idx));
                    new_pair_key[i].push_back(SFM::genKey(pair_cam, pair_idx));
                }
            }
        }

        // 5) Estimate relative pose of the next view (PnP)
        if (pts_3d.size() < 10)
        {
            printf("3DV Tutorial: Image %d is ignored (due to the small number of points).\n", next_cam);
            img_added.insert(next_cam);
            continue;
        }
        cv::Mat K = getCameraMat(cameras[next_cam]), rvec, t;
        std::vector<int> inlier_idx;
        cv::solvePnPRansac(pts_3d, pts_2d, K, cv::Mat::zeros(5, 1, CV_64F), rvec, t, false, 100, 4.0, 0.999, inlier_idx);
        updateCameraPose(cameras[next_cam], rvec, t);
        for (size_t i = 0; i < pts_key.size(); i++)
        {
            SFM::addCostFunc6DOF(ba, Xs[pts_idx[i]], pts_2d[i], cameras[next_cam], ba_loss_width);
            xs_visited[pts_key[i]] = pts_idx[i];
        }

        // 6) Reconstruct newly observed 3D points (triangulation)
        uint new_pts_total = 0;
        for (auto new_pts = new_next_2d.begin(); new_pts != new_next_2d.end(); new_pts++)
            new_pts_total += uint(new_pts->size());
        if (new_pts_total < 10)
        {
            printf("3DV Tutorial: Image %d is complete (only localization; no 3D point addition).\n", next_cam);
            img_added.insert(next_cam);
            continue;
        }
        cv::Mat P0 = getProjectionMat(cameras[next_cam]);
        for (size_t i = 0; i < new_pair_cam.size(); i++)
        {
            const int pair_cam = new_pair_cam[i];
            cv::Mat P1 = getProjectionMat(cameras[pair_cam]), new_Xs;
            cv::triangulatePoints(P0, P1, new_next_2d[i], new_pair_2d[i], new_Xs);
            new_Xs.row(0) = new_Xs.row(0) / new_Xs.row(3);
            new_Xs.row(1) = new_Xs.row(1) / new_Xs.row(3);
            new_Xs.row(2) = new_Xs.row(2) / new_Xs.row(3);

            for (int j = 0; j < new_Xs.cols; j++)
            {
                cv::Point3d p(new_Xs.col(j).rowRange(0, 3));
                if (isBadPoint(p, cameras[next_cam], cameras[pair_cam], Z_limit, max_cos_parallax)) continue; // Do not add if it is bad
                uint X_idx = uint(Xs.size());
                Xs.push_back(p);
                Xs_rgb.push_back(img_set[next_cam].at<cv::Vec3b>(new_next_2d[i][j]));
                xs_visited[new_next_key[i][j]] = X_idx;
                xs_visited[new_pair_key[i][j]] = X_idx;
            }
        }
        printf("3DV Tutorial: Image %d is complete (# of 3D points = %zd).\n", next_cam, Xs.size());

        // 7) Optimize camera pose and 3D points together (bundle adjustment)
        ceres::Solve(options, &ba, &summary);
        img_added.insert(next_cam);
    } // End of the 2nd 'while (true)'
    for (size_t j = 0; j < cameras.size(); j++)
        printf("3DV Tutorial: Camera %zd's (f, cx, cy) = (%.3f, %.1f, %.1f)\n", j, cameras[j][6], cameras[j][7], cameras[j][8]);

    // Store the 3D points to an XYZ file
    FILE* fpts = fopen("sfm_inc(point).xyz", "wt");
    if (fpts == NULL) return -1;
    for (size_t i = 0; i < Xs.size(); i++)
    {
        if (Xs[i].z > -Z_limit && Xs[i].z < Z_limit)
            fprintf(fpts, "%f %f %f %d %d %d\n", Xs[i].x, Xs[i].y, Xs[i].z, Xs_rgb[i][2], Xs_rgb[i][1], Xs_rgb[i][0]); // Format: x, y, z, R, G, B
    }
    fclose(fpts);

    // Store the camera poses to an XYZ file 
    FILE* fcam = fopen("sfm_inc(camera).xyz", "wt");
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
