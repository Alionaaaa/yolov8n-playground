#ifndef OC_SORT_CPP_OCSORT_HPP
#define OC_SORT_CPP_OCSORT_HPP
#include "KalmanBoxTracker.hpp"
#include "Association.hpp"
#include "ORB.hpp"
#include "lapjv.hpp"
#include <functional>
#include <unordered_map>
namespace ocsort {

    class OCSort {
    public:
        OCSort(float det_thresh_, int max_age_ = 30, int min_hits_ = 3, float iou_threshold_ = 0.3,
            int delta_t_ = 3, std::string asso_func_ = "iou", float inertia_ = 0.2, 
            int max_kalman = 5, bool use_byte_ = false);
        std::vector<Eigen::RowVectorXf> update(Eigen::MatrixXf dets);

    public:
        float det_thresh;
        int max_age;
        int min_hits;
        float iou_threshold;
        int delta_t;
        ORBMatcher orb_matcher;
        std::function<Eigen::MatrixXf(const Eigen::MatrixXf&, const Eigen::MatrixXf&)> asso_func;
        float inertia;
        int max_kalman;
        bool use_byte;
        std::vector<KalmanBoxTracker> trackers;
        int frame_count;
    };

}// namespace ocsort
#endif//OC_SORT_CPP_OCSORT_H