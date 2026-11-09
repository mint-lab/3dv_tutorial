#include "opencv2/opencv.hpp"

int main()
{
    const char *video_file = "../data/blais.mp4", *cover_file = "../data/blais.jpg";
    double f = 1000, cx = 320, cy = 240;
    size_t min_inlier_num = 100;

    cv::Ptr<cv::FeatureDetector> fdetector = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> fmatcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // Load the object image and extract features
    cv::Mat obj_image = cv::imread(cover_file);
    if (obj_image.empty()) return -1;

    std::vector<cv::KeyPoint> obj_keypoints;
    cv::Mat obj_descriptors;
    fdetector->detectAndCompute(obj_image, cv::Mat(), obj_keypoints, obj_descriptors);
    if (obj_keypoints.empty() || obj_descriptors.empty()) return -1;
    fmatcher->add(obj_descriptors);

    // Open a video
    cv::VideoCapture video;
    if (!video.open(video_file)) return -1;

    // Prepare a box for simple AR
    std::vector<cv::Point3f> box_lower = { cv::Point3f(30, 145,   0), cv::Point3f(30, 200,   0), cv::Point3f(200, 200,   0), cv::Point3f(200, 145,   0) };
    std::vector<cv::Point3f> box_upper = { cv::Point3f(30, 145, -50), cv::Point3f(30, 200, -50), cv::Point3f(200, 200, -50), cv::Point3f(200, 145, -50) };

    // Run pose estimation
    cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
    cv::Mat dist_coeff = cv::Mat::zeros(5, 1, CV_64F), rvec, tvec;
    while (true)
    {
        // Read an image from the video
        cv::Mat img;
        video >> img;
        if (img.empty()) break;

        // Extract features and match them to the object features
        std::vector<cv::KeyPoint> img_keypoints;
        cv::Mat img_descriptors;
        fdetector->detectAndCompute(img, cv::Mat(), img_keypoints, img_descriptors);
        if (img_keypoints.empty() || img_descriptors.empty()) continue;
        std::vector<cv::DMatch> match;
        fmatcher->match(img_descriptors, match);
        if (match.size() < min_inlier_num) continue;
        std::vector<cv::Point3f> obj_pts;
        std::vector<cv::Point2f> obj_project, img_pts;
        for (auto m = match.begin(); m < match.end(); m++)
        {
            obj_pts.push_back(cv::Point3f(obj_keypoints[m->trainIdx].pt));
            obj_project.push_back(obj_keypoints[m->trainIdx].pt);
            img_pts.push_back(img_keypoints[m->queryIdx].pt);
        }

        // Determine whether each matched feature is an inlier or not
        std::vector<int> inliers;
        cv::solvePnPRansac(obj_pts, img_pts, K, dist_coeff, rvec, tvec, false, 500, 2, 0.99, inliers);
        cv::Mat inlier_mask = cv::Mat::zeros(int(match.size()), 1, CV_8U);
        for (size_t i = 0; i < inliers.size(); i++) inlier_mask.at<uchar>(inliers[i]) = 1;
        cv::Mat img_result;
        cv::drawMatches(img, img_keypoints, obj_image, obj_keypoints, match, img_result, cv::Vec3b(0, 0, 255), cv::Vec3b(0, 127, 0), inlier_mask);

        // Check whether inliers are enough or not
        size_t inlier_num = inliers.size();
        if (inlier_num > min_inlier_num)
        {
            // Estimate camera pose with inliers
            std::vector<cv::Point3f> obj_inliers;
            std::vector<cv::Point2f> img_inliers;
            for (int idx = 0; idx < inlier_mask.rows; idx++)
            {
                if (inlier_mask.at<uchar>(idx))
                {
                    obj_inliers.push_back(obj_pts[idx]);
                    img_inliers.push_back(img_pts[idx]);
                }
            }
            cv::solvePnP(obj_inliers, img_inliers, K, dist_coeff, rvec, tvec);

            // Draw the box on the image
            cv::Mat line_lower, line_upper;
            cv::projectPoints(box_lower, rvec, tvec, K, dist_coeff, line_lower);
            cv::projectPoints(box_upper, rvec, tvec, K, dist_coeff, line_upper);
            line_lower.reshape(1).convertTo(line_lower, CV_32S); // Change 4 x 1 matrix (CV_64FC2) to 4 x 2 matrix (CV_32SC1)
            line_upper.reshape(1).convertTo(line_upper, CV_32S); //  because 'cv::polylines()' only accepts 'CV_32S' depth.
            cv::polylines(img_result, line_lower, true, cv::Vec3b(255, 0, 0), 2);
            for (int i = 0; i < line_lower.rows; i++)
                cv::line(img_result, cv::Point(line_lower.row(i)), cv::Point(line_upper.row(i)), cv::Vec3b(0, 255, 0), 2);
            cv::polylines(img_result, line_upper, true, cv::Vec3b(0, 0, 255), 2);
        }

        // Show the image and process the key event
        cv::String info = cv::format("Inliers: %zu (%.0f%%), Focal length: %.0f", inlier_num, 100.0 * inlier_num / match.size(), K.at<double>(0));
        cv::putText(img_result, info, cv::Point(10, 25), cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Vec3b(0, 255, 0));
        cv::imshow("Pose Estimation (Book)", img_result);
        int key = cv::waitKey(1);
        if (key == 32) key = cv::waitKey(); // Space
        if (key == 27) break;               // ESC
    }

    video.release();
    return 0;
}
