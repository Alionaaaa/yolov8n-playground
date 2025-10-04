#ifndef ORBMATCHER_HPP
#define ORBMATCHER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <random>

class ORBMatcher   {
private:
    bool grayscale;
    float scale;
    bool draw_keypoint_matches;
    bool align;
    
    cv::Ptr<cv::FastFeatureDetector> detector;
    cv::Ptr<cv::ORB> extractor;
    cv::Ptr<cv::BFMatcher> matcher;
    
    cv::Mat prev_img;
    std::vector<cv::KeyPoint> prev_keypoints;
    cv::Mat prev_descriptors;
    Eigen::MatrixXf prev_dets;  
    cv::Mat matches_img;
    cv::Mat prev_img_aligned;
    
    std::mt19937 rng;

public:
    /**
     * Compute the warp matrix from src to dst.
     * 
     * @param feature_detector_threshold The threshold for feature extraction. Defaults to 20.
     * @param matcher_norm_type The norm type of the matcher. Defaults to cv::NORM_HAMMING.
     * @param scale Scale ratio. Defaults to 0.15.
     * @param grayscale Whether to transform 3-channel RGB to single-channel grayscale for faster computations.
     * @param draw_keypoint_matches Whether to draw keypoint matches on the output image.
     * @param align Whether to align the images based on keypoint matches.
     */
    ORBMatcher(int feature_detector_threshold = 20,
        int matcher_norm_type = cv::NORM_HAMMING,
        float scale = 0.15f,
        bool grayscale = true,
        bool draw_keypoint_matches = false,
        bool align = false)
        : grayscale(grayscale), scale(scale), draw_keypoint_matches(draw_keypoint_matches), 
          align(align), rng(std::random_device{}()) {
        
        detector = cv::FastFeatureDetector::create(feature_detector_threshold);
        extractor = cv::ORB::create();
        matcher = cv::BFMatcher::create(matcher_norm_type);
    }
    
    /**
     * Apply ORB-based sparse optical flow to compute the warp matrix.
     * 
     * @param img The input image.
     * @param dets Detected bounding boxes in the image (Eigen::MatrixXf format: [x1, y1, x2, y2, conf, class]).
     * @return The warp matrix from the matching keypoint in the previous image to the current.
     *         The warp matrix is always 2x3.
     */
     cv::Mat generate_mask(const cv::Mat& img, const Eigen::MatrixXf& dets, float scale) {
        int h = img.rows;
        int w = img.cols;
        
        cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
    
        int y_start = static_cast<int>(0.02f * h);
        int y_end = static_cast<int>(0.98f * h);
        int x_start = static_cast<int>(0.02f * w);
        int x_end = static_cast<int>(0.98f * w);
        

        y_start = std::max(0, y_start);
        y_end = std::min(h, y_end);
        x_start = std::max(0, x_start);
        x_end = std::min(w, x_end);
        
        if (y_end > y_start && x_end > x_start) {
            cv::Rect valid_area(x_start, y_start, x_end - x_start, y_end - y_start);
            mask(valid_area) = 255;
        }
        
        if (dets.rows() > 0) {
            for (int i = 0; i < dets.rows(); ++i) {
                int x1 = static_cast<int>(dets(i, 0) * scale);
                int y1 = static_cast<int>(dets(i, 1) * scale);
                int x2 = static_cast<int>(dets(i, 2) * scale);
                int y2 = static_cast<int>(dets(i, 3) * scale);
                
                x1 = std::max(0, std::min(x1, w - 1));
                y1 = std::max(0, std::min(y1, h - 1));
                x2 = std::max(0, std::min(x2, w));
                y2 = std::max(0, std::min(y2, h));
                
                if (x2 > x1 && y2 > y1) {
                    cv::Rect det_rect(x1, y1, x2 - x1, y2 - y1);
                    mask(det_rect) = 0;
                }
            }
        }
        
        return mask;
    }
    
    /**
     * Preprocess input image
     * @param img Input image
     * @return Preprocessed image
     */
    cv::Mat preprocess(const cv::Mat& img) {
        cv::Mat processed_img = img.clone();
        
        if (grayscale && img.channels() == 3) {
            cv::cvtColor(processed_img, processed_img, cv::COLOR_BGR2GRAY);
        } else if (grayscale && img.channels() == 4) {
            cv::cvtColor(processed_img, processed_img, cv::COLOR_BGRA2GRAY);
        }
        
        if (scale != 1.0f && scale > 0.0f) {
            cv::resize(processed_img, processed_img, cv::Size(0, 0), 
                      scale, scale, cv::INTER_LINEAR);
        }
        
        return processed_img;
    }

    cv::Mat apply(const cv::Mat& img, const Eigen::MatrixXf& dets) {
        cv::Mat H = cv::Mat::eye(2, 3, CV_64F);
        
        cv::Mat processed_img = preprocess(img);
        int h = processed_img.rows;
        int w = processed_img.cols;
        
        cv::Mat mask = generate_mask(processed_img, dets, scale);
        
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(processed_img, keypoints, mask);
        
        cv::Mat descriptors;
        extractor->compute(processed_img, keypoints, descriptors);
        
        if (prev_img.empty()) {
            prev_dets = dets;
            prev_img = processed_img.clone();
            prev_keypoints = keypoints;
            prev_descriptors = descriptors.clone();
            return H;
        }
        
        std::vector<std::vector<cv::DMatch>> knnMatches;
        if (!prev_descriptors.empty() && !descriptors.empty()) {
            matcher->knnMatch(prev_descriptors, descriptors, knnMatches, 2);
        }
        
        if (knnMatches.empty()) {
            prev_img = processed_img.clone();
            prev_keypoints = keypoints;
            prev_descriptors = descriptors.clone();
            return H;
        }
        
        std::vector<cv::DMatch> matches;
        std::vector<cv::Point2f> spatial_distances;
        cv::Point2f max_spatial_distance(0.25f * w, 0.25f * h);
        
        for (const auto& match_pair : knnMatches) {
            if (match_pair.size() < 2) continue;
            
            const cv::DMatch& m = match_pair[0];
            const cv::DMatch& n = match_pair[1];
            
            if (m.distance < 0.9f * n.distance) {
                cv::Point2f prevKeyPointLocation = prev_keypoints[m.queryIdx].pt;
                cv::Point2f currKeyPointLocation = keypoints[m.trainIdx].pt;
                
                cv::Point2f spatial_distance = prevKeyPointLocation - currKeyPointLocation;
                
                if (std::abs(spatial_distance.x) < max_spatial_distance.x && 
                    std::abs(spatial_distance.y) < max_spatial_distance.y) {
                    spatial_distances.push_back(spatial_distance);
                    matches.push_back(m);
                }
            }
        }
        
        if (spatial_distances.empty()) {
            prev_img = processed_img.clone();
            prev_keypoints = keypoints;
            prev_descriptors = descriptors.clone();
            return H;
        }
        
        cv::Point2f mean_spatial_distances(0, 0);
        for (const auto& dist : spatial_distances) {
            mean_spatial_distances += dist;
        }
        mean_spatial_distances.x /= spatial_distances.size();
        mean_spatial_distances.y /= spatial_distances.size();
        
        cv::Point2f std_spatial_distances(0, 0);
        for (const auto& dist : spatial_distances) {
            cv::Point2f diff = dist - mean_spatial_distances;
            std_spatial_distances.x += diff.x * diff.x;
            std_spatial_distances.y += diff.y * diff.y;
        }
        std_spatial_distances.x = std::sqrt(std_spatial_distances.x / spatial_distances.size());
        std_spatial_distances.y = std::sqrt(std_spatial_distances.y / spatial_distances.size());
        
        std::vector<cv::DMatch> goodMatches;
        std::vector<cv::Point2f> prevPoints, currPoints;
        
        for (size_t i = 0; i < matches.size(); ++i) {
            cv::Point2f diff = spatial_distances[i] - mean_spatial_distances;
            bool inlier_x = std::abs(diff.x) < 2.5f * std_spatial_distances.x;
            bool inlier_y = std::abs(diff.y) < 2.5f * std_spatial_distances.y;
            
            if (inlier_x && inlier_y) {
                goodMatches.push_back(matches[i]);
                prevPoints.push_back(prev_keypoints[matches[i].queryIdx].pt);
                currPoints.push_back(keypoints[matches[i].trainIdx].pt);
            }
        }
        
        if (draw_keypoint_matches) {
            cv::Mat prev_img_masked = prev_img.clone();
            prev_img_masked.setTo(0, mask);
            
            cv::hconcat(prev_img_masked, processed_img, matches_img);
            cv::cvtColor(matches_img, matches_img, cv::COLOR_GRAY2BGR);
            
            std::uniform_int_distribution<int> color_dist(0, 255);
            
            for (const auto& match : goodMatches) {
                cv::Point prev_pt(static_cast<int>(prev_keypoints[match.queryIdx].pt.x),
                                static_cast<int>(prev_keypoints[match.queryIdx].pt.y));
                cv::Point curr_pt(static_cast<int>(keypoints[match.trainIdx].pt.x + w),
                                static_cast<int>(keypoints[match.trainIdx].pt.y));
                
                cv::Scalar color(color_dist(rng), color_dist(rng), color_dist(rng));
                
                cv::line(matches_img, prev_pt, curr_pt, color, 1, cv::LINE_AA);
                cv::circle(matches_img, prev_pt, 2, color, -1);
                cv::circle(matches_img, curr_pt, 2, color, -1);
            }
            
            for (int i = 0; i < dets.rows(); ++i) {
                cv::Rect det(static_cast<int>(dets(i, 0) * scale + w),  // x1
                           static_cast<int>(dets(i, 1) * scale),        // y1
                           static_cast<int>((dets(i, 2) - dets(i, 0)) * scale),  // width
                           static_cast<int>((dets(i, 3) - dets(i, 1)) * scale)); // height
                cv::rectangle(matches_img, det, cv::Scalar(0, 0, 255), 2);
            }
            
            for (int i = 0; i < prev_dets.rows(); ++i) {
                cv::Rect det(static_cast<int>(prev_dets(i, 0) * scale),
                           static_cast<int>(prev_dets(i, 1) * scale),
                           static_cast<int>((prev_dets(i, 2) - prev_dets(i, 0)) * scale),
                           static_cast<int>((prev_dets(i, 3) - prev_dets(i, 1)) * scale));
                cv::rectangle(matches_img, det, cv::Scalar(0, 0, 255), 2);
            }
        }
        
        if (prevPoints.size() > 4 && prevPoints.size() == currPoints.size()) {
            std::vector<uchar> inliers;
            cv::Mat H_estimated = cv::estimateAffinePartial2D(prevPoints, currPoints, 
                                                            inliers, cv::RANSAC);
            
            if (!H_estimated.empty()) {
                H = H_estimated;
                
                if (scale < 1.0f) {
                    H.at<double>(0, 2) /= scale;
                    H.at<double>(1, 2) /= scale;
                }
                
                if (align) {
                    cv::warpAffine(prev_img, prev_img_aligned, H, cv::Size(w, h), 
                                 cv::INTER_LINEAR);
                }
            }
        } else {
            std::cout << "Warning: not enough matching points" << std::endl;
        }
        
        prev_img = processed_img.clone();
        prev_keypoints = keypoints;
        prev_descriptors = descriptors.clone();
        prev_dets = dets;  
        
        return H;
    }
    
    cv::Mat getMatchesImage() const { return matches_img; }
    cv::Mat getPrevImageAligned() const { return prev_img_aligned; }
};

#endif