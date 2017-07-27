#include "opencv_all.hpp"
#include "cvsba.h"
#include <unordered_map>
#include <unordered_set>

#define MAKE_KEY(img_idx, pt_idx)       ((uint(img_idx) << 16) + pt_idx)

int main(void)
{
    double default_camera_f = 500, image_resize = 0.25, max_cos_parallax = cos(10 * CV_PI / 180);
    size_t min_inlier_num = 500, min_add_pts_num = 10;
    bool show_match = true;

    // Load images and extract features
    cv::VideoCapture video;
    if (!video.open("data/relief/%02d.jpg")) return -1;
    cv::Ptr<cv::FeatureDetector> fdetector = cv::BRISK::create();
    std::vector<std::vector<cv::KeyPoint> > img_keypoint;
    std::vector<cv::Mat> img_set, img_descriptor;
    while (true)
    {
        cv::Mat image;
        video >> image;
        if (image.empty()) break;
        if (image_resize != 1) cv::resize(image, image, cv::Size(), image_resize, image_resize);
        std::vector<cv::KeyPoint> keypoint;
        cv::Mat descriptor;
        fdetector->detectAndCompute(image, cv::Mat(), keypoint, descriptor);
        img_set.push_back(image);
        img_keypoint.push_back(keypoint);
        img_descriptor.push_back(descriptor.clone());
    }
    if (img_set.size() < 2) return -1;

    // 0) Build a viewing graph (match features over all image pairs and find inliers)
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
            if (inlier.size() < min_inlier_num) continue;   // The threshold for putting into the viewing graph
            printf("3DV Tutorial: Image %d - %d are selected.\n", i, j);
            match_pair.push_back(std::pair<int, int>(i, j));
            match_inlier.push_back(inlier);
            if (show_match)
            {
                cv::Mat match_image;
                cv::drawMatches(img_set[i], img_keypoint[i], img_set[j], img_keypoint[j], match, match_image, cv::Scalar(0, 0, 255), cv::Scalar(0, 127, 0), inlier_mask);
                cv::imshow("3DV Tutorial: Incremental Structure-from-Motion", match_image);
                cv::waitKey();
            }
        }
    }
    if (match_pair.size() < 1) return -1;

    // Prepare camera matrices, pose, 3D points, their observation and visibility
    std::vector<cv::Point3d> Xs;
    std::vector<std::vector<cv::Point2d> > xs;
    std::vector<std::vector<int> > visibility;
    std::vector<cv::Mat> Ks, dist_coeffs, Rs, ts;
    std::unordered_map<uint, uint> xs_visited;
    std::unordered_set<uint> img_added;

    // 1) Select the best pair
    int best_pair = 0;
    for (size_t i = 0; i < match_inlier.size(); i++)
        if (match_inlier[i].size() > match_inlier[best_pair].size()) best_pair = i;

    // 2) Estimate relative pose from the best two views (epipolar geometry)
    std::vector<cv::Point2d> src, dst;
    for (auto itr = match_inlier[best_pair].begin(); itr != match_inlier[best_pair].end(); itr++)
    {
        src.push_back(img_keypoint[match_pair[best_pair].first][itr->queryIdx].pt);
        dst.push_back(img_keypoint[match_pair[best_pair].second][itr->trainIdx].pt);
    }
    cv::Mat K = (cv::Mat_<double>(3, 3) << default_camera_f, 0, img_set[match_pair[best_pair].first].cols / 2, 0, default_camera_f, img_set[match_pair[best_pair].first].rows / 2, 0, 0, 1), R, t, Rt, inlier_mask;
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

    // 3) Reconstruct 3D points of the best two views (triangulation)
    cv::hconcat(R, t, Rt);
    cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F), P1 = K * Rt, X;
    cv::triangulatePoints(P0, P1, src, dst, X);
    X.row(0) = X.row(0) / X.row(3);
    X.row(1) = X.row(1) / X.row(3);
    X.row(2) = X.row(2) / X.row(3);

    xs.resize(2);
    for (size_t idx = 0; idx < src.size(); idx++)
    {
        cv::Mat p = X.col(idx).rowRange(0, 3);                          // A 3D point at 'idx'
        cv::Mat p2 = R * p + t;                                         // A 3D point with respect to the 2nd camera coordinate
        if (p.at<double>(2) <= 0 || p2.at<double>(2) <= 0) continue;    // Skip a point when it is beyond of one of 1st and 2nd cameras
        cv::Mat v2 = p + t;                                             // A vector from the 2nd camera to the 3D point
        double cos_parallax = p.dot(v2) / cv::norm(p) / cv::norm(v2);
        if (cos_parallax > max_cos_parallax) continue;                  // Skip a point when has small parallax angle
        int X_idx = Xs.size();
        Xs.push_back(cv::Point3d(p));
        xs[0].push_back(src[idx]);
        xs[1].push_back(dst[idx]);
        xs_visited[MAKE_KEY(match_pair[best_pair].first, match_inlier[best_pair][idx].queryIdx)] = X_idx;
        xs_visited[MAKE_KEY(match_pair[best_pair].second, match_inlier[best_pair][idx].trainIdx)] = X_idx;
    }
    visibility.resize(2, std::vector<int>(Xs.size(), 1));
    Ks.push_back(K.clone());
    Ks.push_back(K.clone());
    dist_coeffs.push_back(cv::Mat::zeros(5, 1, CV_64F));
    dist_coeffs.push_back(cv::Mat::zeros(5, 1, CV_64F));
    Rs.push_back(cv::Mat::eye(3, 3, CV_64F));
    Rs.push_back(R.clone());
    ts.push_back(cv::Mat::zeros(3, 1, CV_64F));
    ts.push_back(t.clone());
    printf("3DV Tutorial: Image %d and %d are complete (# of 3D points = %d).\n", match_pair[best_pair].first, match_pair[best_pair].second, Xs.size());
    img_added.insert(match_pair[best_pair].first);
    img_added.insert(match_pair[best_pair].second);
    match_pair.erase(match_pair.begin() + best_pair);       // Remove the completed pair
    match_inlier.erase(match_inlier.begin() + best_pair);   // Remove the completed inliers

    // Incrementally add more views
    cvsba::Sba sba;
    cvsba::Sba::Params param;
    param.type = cvsba::Sba::MOTIONSTRUCTURE;
    param.fixedIntrinsics = 4; // Free focal length
    param.fixedDistortion = 5;
    param.verbose = true;
    sba.setParams(param);
    while (!match_pair.empty())
    {
        // 4) Select the next image to add
        std::vector<size_t> img_score(img_set.size(), 0);
        std::vector<std::vector<uint> > match_table(img_set.size());
        for (size_t img = 0; img < img_score.size(); img++)
        {
            if (img_added.find(img) == img_added.end())                                                             // When the image is not added to the viewing graph
            {
                for (size_t i = 0; i < match_pair.size(); i++)
                {
                    if (match_pair[i].first == img && img_added.find(match_pair[i].second) != img_added.end())      // When 'first' is the current image and 'second' is already added
                    {
                        for (auto itr = match_inlier[i].begin(); itr != match_inlier[i].end(); itr++)
                            if (xs_visited.find(MAKE_KEY(match_pair[i].second, itr->trainIdx)) != xs_visited.end()) // When a matched inlier is in 'Xs', the current image gains more score
                                img_score[img]++;
                        match_table[img].push_back(i);
                    }
                    else if (match_pair[i].second == img && img_added.find(match_pair[i].first) != img_added.end()) // When 'second' is the current image and 'first' is already added
                    {
                        for (auto itr = match_inlier[i].begin(); itr != match_inlier[i].end(); itr++)
                            if (xs_visited.find(MAKE_KEY(match_pair[i].first, itr->queryIdx)) != xs_visited.end())  // When a matched inlier is in 'Xs', the current image gains more score
                                img_score[img]++;
                        match_table[img].push_back(i);
                    }
                }
            }
        }
        const auto next_score = std::max_element(img_score.begin(), img_score.end());
        if (*next_score <= min_add_pts_num) break;
        const int next_img = std::distance(img_score.begin(), next_score);
        std::vector<uint> next_match = match_table[next_img];

        // Separate points into known (pts_*) and unknown (new_*) for PnP (known) and triangulation (unknown)
        std::vector<cv::Point3d> pts_3d;
        std::vector<cv::Point2d> pts_2d;
        std::vector<uint> pts_key, pts_idx, new_pair_img;
        std::vector<std::vector<cv::Point2d> > new_next_2d(next_match.size()), new_pair_2d(next_match.size());
        std::vector<std::vector<uint> > new_next_key(next_match.size()), new_pair_key(next_match.size());
        for (size_t i = 0; i < next_match.size(); i++)
        {
            bool next_is_first = true;
            int pair_img = match_pair[next_match[i]].second;
            if (pair_img == next_img)
            {
                next_is_first = false;
                pair_img = match_pair[next_match[i]].first;
            }
            new_pair_img.push_back(pair_img);
            for (auto itr = match_inlier[next_match[i]].begin(); itr != match_inlier[next_match[i]].end(); itr++)
            {
                int next_idx = itr->queryIdx, pair_idx = itr->trainIdx;
                if (!next_is_first)
                {
                    next_idx = itr->trainIdx;
                    pair_idx = itr->queryIdx;
                }
                auto found = xs_visited.find(MAKE_KEY(pair_img, pair_idx));
                if (found != xs_visited.end())  // When the matched point is already known (--> PnP)
                {
                    pts_3d.push_back(Xs[found->second]);
                    pts_2d.push_back(img_keypoint[next_img][next_idx].pt);
                    pts_key.push_back(MAKE_KEY(next_img, next_idx));
                    pts_idx.push_back(found->second);
                }
                else                            // When the matched point is newly observed (--> triangulation)
                {
                    new_next_2d[i].push_back(img_keypoint[next_img][next_idx].pt);
                    new_pair_2d[i].push_back(img_keypoint[pair_img][pair_idx].pt);
                    new_next_key[i].push_back(MAKE_KEY(next_img, next_idx));
                    new_pair_key[i].push_back(MAKE_KEY(pair_img, pair_idx));
                }
            }
        }

        // 5) Estimate relative pose of the next view (PnP)
        cv::Mat K = (cv::Mat_<double>(3, 3) << default_camera_f, 0, img_set[next_img].cols / 2, 0, default_camera_f, img_set[next_img].rows / 2, 0, 0, 1), rvec;
        std::vector<int> inlier_idx;
        cv::solvePnPRansac(pts_3d, pts_2d, K, cv::Mat::zeros(5, 1, CV_64F), rvec, t, false, 100, 4.0, 0.999, inlier_idx);
        cv::Rodrigues(rvec, R);

        xs.push_back(std::vector<cv::Point2d>(xs.back().size()));
        visibility.push_back(std::vector<int>(visibility.back().size(), 0));
        for (auto idx = inlier_idx.begin(); idx != inlier_idx.end(); idx++)
        {
            const int& X_idx = pts_idx[*idx];
            xs.back()[X_idx] = pts_2d[*idx];
            visibility.back()[X_idx] = 1;
            xs_visited[pts_key[*idx]] = X_idx;
        }
        Ks.push_back(K.clone());
        dist_coeffs.push_back(cv::Mat::zeros(5, 1, CV_64F));
        Rs.push_back(R.clone());
        ts.push_back(t.clone());

        // 6) Reconstruct newly observed 3D points (triangulation)
        cv::hconcat(Rs.back(), ts.back(), Rt);
        cv::Mat P0 = Ks.back() * Rt;
        for (size_t i = 0; i < new_pair_img.size(); i++)
        {
            auto pair_itr = img_added.find(new_pair_img[i]);
            if (pair_itr == img_added.end()) continue;                          // When the pair image is not in 'img_added' (never happended)
            const int pair_idx = std::distance(img_added.begin(), pair_itr);    // The pair image's index in 'img_added' is not same with its index in 'img_set'.
            cv::hconcat(Rs[pair_idx], ts[pair_idx], Rt);
            cv::Mat P1 = Ks[pair_idx] * Rt, X;
            cv::triangulatePoints(P0, P1, new_next_2d[i], new_pair_2d[i], X);
            X.row(0) = X.row(0) / X.row(3);
            X.row(1) = X.row(1) / X.row(3);
            X.row(2) = X.row(2) / X.row(3);

            for (int j = 0; j < X.cols; j++)
            {
                cv::Mat p = X.col(j).rowRange(0, 3);
                cv::Point3d pt(p);
                cv::Mat p1 = Rs.back() * p + ts.back(), p2 = Rs[pair_idx] * p + ts[pair_idx];   // 3D points with respect to two cameras
                if (p1.at<double>(2) <= 0 || p2.at<double>(2) <= 0) continue;                   // Skip a point when it is beyond of one of two cameras
                cv::Mat v1 = p + ts.back(), v2 = p + ts[pair_idx];                              // Vectors from the next/pair camera to the 3D point
                double cos_parallax = v1.dot(v2) / cv::norm(v1) / cv::norm(v2);
                if (cos_parallax > max_cos_parallax) continue;                                  // Skip a point when it has small parallax angle
                int X_idx = Xs.size();
                Xs.push_back(cv::Point3d(p));
                for (size_t k = 0; k < xs.size(); k++)
                {
                    xs[k].push_back(cv::Point2d());
                    visibility[k].push_back(0);
                }
                xs.back().back() = new_next_2d[i][j];
                xs[pair_idx].back() = new_pair_2d[i][j];
                visibility.back().back() = 1;
                visibility[pair_idx].back() = 1;
                xs_visited[new_next_key[i][j]] = X_idx;
                xs_visited[new_pair_key[i][j]] = X_idx;
            }
        }
        printf("3DV Tutorial: Image %d is complete (# of 3D points = %d).\n", next_img, Xs.size());

        // 7) Optimize camera pose and 3D points (bundle adjustment)
        try { double error = sba.run(Xs, xs, visibility, Ks, Rs, ts, dist_coeffs); }
        catch (cv::Exception) {}
        for (size_t i = 0; i < next_match.size(); i++)
        {
            match_pair.erase(match_pair.begin() + next_match[i]);               // Remove the completed pair
            match_inlier.erase(match_inlier.begin() + next_match[i]);           // Remove the completed inliers
        }
        img_added.insert(next_img);
    } // End of 'while (!match_pair.empty())'

    // Store the 3D points to an XYZ file
    FILE* fpts = fopen("sfm_inc(point).xyz", "wt");
    if (fpts == NULL) return -1;
    for (size_t i = 0; i < Xs.size(); i++)
        fprintf(fpts, "%f %f %f\n", Xs[i].x, Xs[i].y, Xs[i].z);
    fclose(fpts);

    // Store the camera poses to an XYZ file 
    std::vector<int> cam_to_img(img_added.begin(), img_added.end());
    FILE* fcam = fopen("sfm_inc(camera).xyz", "wt");
    if (fcam == NULL) return -1;
    for (size_t i = 0; i < Rs.size(); i++)
    {
        cv::Mat p = -Rs[i].t() * ts[i];
        fprintf(fcam, "%f %f %f\n", p.at<double>(0), p.at<double>(1), p.at<double>(2));
        printf("3DV Tutorial: Image %d's focal length = %.3f\n", cam_to_img[i], Ks[i].at<double>(0, 0));
    }
    fclose(fcam);
    return 0;
}