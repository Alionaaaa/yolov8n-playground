#ifndef YOLO_ENGINE_H
#define YOLO_ENGINE_H

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

// Forward declarations for ONNX Runtime
namespace Ort {
    class Env;
    class Session;
    class Value;
    class MemoryInfo;
}

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;

    Detection(float x1, float y1, float x2, float y2, float conf, int cls_id)
        : x1(x1), y1(y1), x2(x2), y2(y2), confidence(conf), class_id(cls_id) {}
};

struct Results {
    std::vector<Detection> boxes;
    std::map<std::string, float> speed;

    Results() {
        speed["preprocess"] = 0.0f;
        speed["inference"] = 0.0f;
        speed["postprocess"] = 0.0f;
    }
};

class YoloEngine {
public:
    YoloEngine(const std::string& model_path,
               float conf_threshold = 0.25f,
               float iou_threshold = 0.5f);
    ~YoloEngine();

    Results process(const cv::Mat& image);
    cv::Mat visualize(const cv::Mat& image, const std::vector<Detection>& detections);

private:
    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;

    // Model info
    std::string model_path_;
    bool is_initialized_ = false;
    float conf_threshold_ = 0.5;
    float iou_threshold_ = 0.5;
    cv::Size input_size_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<int64_t> input_shape_;
    std::vector<float> input_tensor_;

    // YOLO parameters
    int num_classes_;
    int output_numprob_;
    int output_numbox_;
    

    // Class names
    const std::vector<std::string> class_names_ = {
        "person","bicycle","car","motorcycle","airplane","bus","train","truck",
        "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
        "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
        "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
        "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
        "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
        "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
        "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
        "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
        "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
        "hair drier","toothbrush"
    };

    // Private methods
    void letterbox(const cv::Mat& image, cv::Mat& out, float& scale, int& dx, int& dy);

    void pre_process(const cv::Mat& image, float& scale, int& dx, int& dy);       
    std::vector<Detection> post_process(float* outputs, const cv::Size& original_size,
                                float scale, int dx, int dy);
};

#endif
