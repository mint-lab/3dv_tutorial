#include "opencv_all.hpp"
#include "cvsba.h"
#include <unordered_map>

#define MAKE_KEY(img_idx, pt_idx)       ((uint(img_idx) << 16) + pt_idx)

int main(void)
{
    double default_camera_f = 500, image_resize = 0.25, default_point_depth = 2;
    size_t min_inlier_num = 500;
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
            if (inlier.size() < min_inlier_num) continue; // The threshold for putting into the viewing graph
            printf("3DV Tutorial: Image %d - %d are selected.\n", i, j);
            match_pair.push_back(std::pair<int, int>(i, j));
            match_inlier.push_back(inlier);
            if (show_match)
            {
                cv::Mat match_image;
                cv::drawMatches(img_set[i], img_keypoint[i], img_set[j], img_keypoint[j], match, match_image, cv::Scalar(0, 0, 255), cv::Scalar(0, 127, 0), inlier_mask);
                cv::imshow("3DV Tutorial: Global Structure-from-Motion", match_image);
                cv::waitKey();
            }
        }
    }
    if (match_pair.size() < 1) return -1;

    // 1) Arrange 3D points, their observation and visibility ('Xs', 'xs', and 'visibility')
    std::vector<cv::Point3d> Xs;
    std::vector<std::vector<cv::Point2d> > xs(img_set.size());
    std::vector<std::vector<int> > visibility(img_set.size());
    std::unordered_map<uint, uint> xs_visited;
    std::set<uint> Xs_rejected;
    for (size_t i = 0; i < match_pair.size(); i++)
    {
        for (size_t j = 0; j < match_inlier[i].size(); j++)
        {
            uint X_idx = 0;
            const int &img1 = match_pair[i].first, &img2 = match_pair[i].second, &x1_idx = match_inlier[i][j].queryIdx, &x2_idx = match_inlier[i][j].trainIdx;
            const uint key1 = MAKE_KEY(img1, x1_idx), key2 = MAKE_KEY(img2, x2_idx);
            auto value1 = xs_visited.find(key1), value2 = xs_visited.find(key2);
            if (value1 != xs_visited.end() && value2 != xs_visited.end())
            {
                // When the existing point correspondences are inconsistent, do not use these points.
                if (value1->second != value2->second)
                {
                    Xs_rejected.insert(value1->second);
                    Xs_rejected.insert(value2->second);
                    continue;
                }
                X_idx = value1->second;
            }
            else if (value1 != xs_visited.end()) X_idx = value1->second;
            else if (value2 != xs_visited.end()) X_idx = value2->second;
            else
            {
                // When the point observations are not visited, add a new point.
                X_idx = Xs.size();
                Xs.push_back(cv::Point3d(0, 0, default_point_depth));
                for (size_t k = 0; k < img_set.size(); k++)
                {
                    xs[k].push_back(cv::Point2d());
                    visibility[k].push_back(0);
                }
            }
            xs[img1][X_idx] = img_keypoint[img1][x1_idx].pt;
            xs[img2][X_idx] = img_keypoint[img2][x2_idx].pt;
            visibility[img1][X_idx] = 1;
            visibility[img2][X_idx] = 1;
            xs_visited[key1] = X_idx;
            xs_visited[key2] = X_idx;
        }
    }

    // Remove rejected 3D points, which were inconsistent at the previous step
    for (auto idx = Xs_rejected.rbegin(); idx != Xs_rejected.rend(); idx++)
    {
        Xs.erase(Xs.begin() + *idx);
        for (size_t j = 0; j < img_set.size(); j++)
        {
            xs[j].erase(xs[j].begin() + *idx);
            visibility[j].erase(visibility[j].begin() + *idx);
        }
    }
    printf("3DV Tutorial: # of 3D points = %d (# of rejected = %d, # of projections = %d).\n", Xs.size(), Xs_rejected.size(), xs_visited.size());

    // 2) Initialize each camera projection matrix
    std::vector<cv::Mat> Ks, dist_coeffs, Rs, ts;
    for (size_t i = 0; i < img_set.size(); i++)
    {
        cv::Mat K = (cv::Mat_<double>(3, 3) << default_camera_f, 0, img_set[i].cols / 2., 0, default_camera_f, img_set[i].rows / 2., 0, 0, 1);
        Ks.push_back(K.clone());                                // K for all cameras
        dist_coeffs.push_back(cv::Mat::zeros(5, 1, CV_64F));    // dist_coeff for all cameras
        Rs.push_back(cv::Mat::eye(3, 3, CV_64F));               // R for all cameras
        ts.push_back(cv::Mat::zeros(3, 1, CV_64F));             // t for all cameras
    }

    // 3) Optimize camera pose and 3D points
    try
    {
        cvsba::Sba sba;
        cvsba::Sba::Params param;
        param.type = cvsba::Sba::MOTIONSTRUCTURE;
        param.fixedIntrinsics = 4; // Free focal length
        param.fixedDistortion = 5;
        param.verbose = true;
        sba.setParams(param);
        double error = sba.run(Xs, xs, visibility, Ks, Rs, ts, dist_coeffs);
    }
    catch (cv::Exception) {}

    // Store the 3D points to an XYZ file
    FILE* fpts = fopen("sfm_global(point).xyz", "wt");
    if (fpts == NULL) return -1;
    for (size_t i = 0; i < Xs.size(); i++)
        fprintf(fpts, "%f %f %f\n", Xs[i].x, Xs[i].y, Xs[i].z);
    fclose(fpts);

    // Store the camera poses to an XYZ file 
    FILE* fcam = fopen("sfm_global(camera).xyz", "wt");
    if (fcam == NULL) return -1;
    for (size_t i = 0; i < Rs.size(); i++)
    {
        cv::Mat p = -Rs[i].t() * ts[i];
        fprintf(fcam, "%f %f %f\n", p.at<double>(0), p.at<double>(1), p.at<double>(2));
        printf("3DV Tutorial: Image %d's focal length = %.3f\n", i, Ks[i].at<double>(0, 0));
    }
    fclose(fcam);
    return 0;
}
