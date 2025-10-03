#include "yolo_engine.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <thread>

YoloEngine::YoloEngine(const std::string& model_path, float conf_threshold, float iou_threshold)
    : model_path_(model_path),
      is_initialized_(false),
      conf_threshold_(conf_threshold),
      iou_threshold_(iou_threshold),
      input_size_(640, 640),
      num_classes_(80),
      output_numprob_(84),
      output_numbox_(8400) {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
        session_options.SetInterOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;

        // Input information
        input_names_ = { session_->GetInputNameAllocated(0, allocator).get() };
        auto input_type_info = session_->GetInputTypeInfo(0);
        input_shape_ = input_type_info.GetTensorTypeAndShapeInfo().GetShape();

        // Output information
        output_names_ = { session_->GetOutputNameAllocated(0, allocator).get() };

        memory_info_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        is_initialized_ = true;
        std::cout << "Model loaded. Input shape: [";
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            std::cout << input_shape_[i];
            if (i < input_shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("ONNX Runtime init error: ") + e.what());
    }
}

YoloEngine::~YoloEngine() {
    std::cout << "YOLO ONNX Engine destroyed" << std::endl;
}

Results YoloEngine::process(const cv::Mat& image) {
    if (!is_initialized_) return {};

    Results results;

    float scale;
    int dx, dy;

    auto t0 = std::chrono::high_resolution_clock::now();
    pre_process(image, scale, dx, dy);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<int64_t> input_dims = {1, 3, input_size_.height, input_size_.width};
    Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
    *memory_info_, input_tensor_.data(), input_tensor_.size(), input_dims.data(), input_dims.size());

    std::vector<const char*> input_names_c{input_names_[0].c_str()};
    std::vector<const char*> output_names_c{output_names_[0].c_str()};

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                input_names_c.data(),
                &input_tensor_ort,
                1,
                output_names_c.data(),
                1);

    auto t2 = std::chrono::high_resolution_clock::now();

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    results.boxes = post_process(output_data, image.size(), scale, dx, dy);

    auto t3 = std::chrono::high_resolution_clock::now();

    results.speed["preprocess"] = std::chrono::duration<float, std::milli>(t1 - t0).count();
    results.speed["inference"]  = std::chrono::duration<float, std::milli>(t2 - t1).count();
    results.speed["postprocess"]= std::chrono::duration<float, std::milli>(t3 - t2).count();

    return results;
}

void YoloEngine::pre_process(const cv::Mat& image, float& scale, int& dx, int& dy) 
{
    cv::Mat letterbox_img;
    letterbox(image, letterbox_img, scale, dx, dy);

    cv::Mat rgb;
    cv::cvtColor(letterbox_img, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    int H = rgb.rows;
    int W = rgb.cols;
    int C = 3;

    input_tensor_.resize(H * W * C);

    // HWC -> CHW conversion
    for (int y = 0; y < H; ++y) {
        const cv::Vec3f* row = rgb.ptr<cv::Vec3f>(y);
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                input_tensor_[c * H * W + y * W + x] = row[x][c];
            }
        }
    }
}

void YoloEngine::letterbox(const cv::Mat& image, cv::Mat& out, float& scale, int& dx, int& dy) {
    cv::Size shape = image.size();
    scale = std::min((float)input_size_.width / shape.width, (float)input_size_.height / shape.height);
    int new_w = int(shape.width * scale);
    int new_h = int(shape.height * scale);
    cv::resize(image, out, cv::Size(new_w, new_h));

    dx = (input_size_.width - new_w) / 2;
    dy = (input_size_.height - new_h) / 2;
    cv::copyMakeBorder(out, out, dy, input_size_.height - new_h - dy, dx, input_size_.width - new_w - dx,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
}

std::vector<Detection> YoloEngine::post_process(float* outputs, const cv::Size& original_size,
    float scale, int dx, int dy)
{
    std::vector<Detection> detections;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    // const int numFeatures = output_numprob_; // 84 (4 bbox coordinates + 80 classes)
    const int numBoxes = output_numbox_;     // 8400

    // Helper function for clamping values (compatible with C++14)
    auto clamp = [](float value, float min_val, float max_val) {
        return std::max(min_val, std::min(value, max_val));
    };

    for (int box_idx = 0; box_idx < numBoxes; ++box_idx) {
        // Get data for each box from the output array
        float x = outputs[box_idx + 0 * numBoxes];
        float y = outputs[box_idx + 1 * numBoxes];
        float w = outputs[box_idx + 2 * numBoxes];
        float h = outputs[box_idx + 3 * numBoxes];

        // Class scores
        float* class_scores = outputs + 4 * numBoxes + box_idx; // first class
        int best_class = 0;
        float max_score = class_scores[0];
        for (int c = 1; c < num_classes_; ++c) {
            float score = class_scores[c * numBoxes];
            if (score > max_score) {
                max_score = score;
                best_class = c;
            }
        }

        if (max_score >= conf_threshold_) {
            // Convert bbox coordinates considering scale and offset
            float x1 = (x - w / 2 - dx) / scale;
            float y1 = (y - h / 2 - dy) / scale;
            float x2 = (x + w / 2 - dx) / scale;
            float y2 = (y + h / 2 - dy) / scale;

            // Clamp coordinates to image boundaries
            x1 = clamp(x1, 0.f, float(original_size.width));
            y1 = clamp(y1, 0.f, float(original_size.height));
            x2 = clamp(x2, 0.f, float(original_size.width));
            y2 = clamp(y2, 0.f, float(original_size.height));

            if (x2 > x1 && y2 > y1) {
                detections.emplace_back(x1, y1, x2, y2, max_score, best_class);
                boxes.emplace_back(cv::Point(x1, y1), cv::Point(x2, y2));
                scores.push_back(max_score);
            }
        }
    }

    // Apply Non-Maximum Suppression
    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, iou_threshold_, nmsResult);

    std::vector<Detection> final_detections;
    for (int idx : nmsResult)
        final_detections.push_back(detections[idx]);

    return final_detections;
}

cv::Mat YoloEngine::visualize(const cv::Mat& image, const std::vector<Detection>& detections) {
    cv::Mat result = image.clone();
    for (auto& det : detections) {
        cv::rectangle(result, cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2), cv::Scalar(0, 255, 0), 2);
        
        std::string label = static_cast<size_t>(det.class_id) < class_names_.size() 
                ? class_names_[det.class_id] : "unknown";
        
        label += " " + std::to_string(int(det.confidence * 100)) + "%";
        
        int base_line;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);
        cv::rectangle(result, cv::Point(det.x1, det.y1 - label_size.height - base_line),
                      cv::Point(det.x1 + label_size.width, det.y1), cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(result, label, cv::Point(det.x1, det.y1 - base_line),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    return result;
}