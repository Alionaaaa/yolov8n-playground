#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "yolo_engine.h"

namespace py = pybind11;

// Convert numpy array to cv::Mat
cv::Mat numpy_to_mat(py::array_t<uint8_t>& input) {
    py::buffer_info buf_info = input.request();
    
    if (buf_info.ndim != 3) {
        throw std::runtime_error("Number of dimensions must be 3 (H, W, C)");
    }
    
    int height = buf_info.shape[0];
    int width = buf_info.shape[1];
    int channels = buf_info.shape[2];
    
    return cv::Mat(height, width, CV_8UC(channels), buf_info.ptr);
}

// Convert cv::Mat to numpy array
py::array_t<uint8_t> mat_to_numpy(const cv::Mat& image) {
    py::array_t<uint8_t> output({image.rows, image.cols, image.channels()});
    py::buffer_info buf_info = output.request();
    std::memcpy(buf_info.ptr, image.data, image.total() * image.elemSize());
    return output;
}

PYBIND11_MODULE(yolo_onnx, m) {
    m.doc() = "YOLOv8 ONNX Runtime C++ Inference Engine";
    
    py::class_<Detection>(m, "Detection")
        .def(py::init<float, float, float, float, float, int>())
        .def_readwrite("x1", &Detection::x1)
        .def_readwrite("y1", &Detection::y1)
        .def_readwrite("x2", &Detection::x2)
        .def_readwrite("y2", &Detection::y2)
        .def_readwrite("confidence", &Detection::confidence)
        .def_readwrite("class_id", &Detection::class_id)
        .def("__repr__", [](const Detection& d) {
            return "Detection(x1=" + std::to_string(d.x1) + 
                   ", y1=" + std::to_string(d.y1) +
                   ", x2=" + std::to_string(d.x2) + 
                   ", y2=" + std::to_string(d.y2) +
                   ", conf=" + std::to_string(d.confidence) +
                   ", class_id=" + std::to_string(d.class_id) + ")";
        });

        py::class_<Results>(m, "Results")
        .def_readwrite("boxes", &Results::boxes)
        .def_readwrite("speed", &Results::speed)
        .def("__repr__", [](const Results& r) {
            return "<Results: " + std::to_string(r.boxes.size()) + " boxes>";
        })
        .def("plot", [](Results& self, py::array_t<uint8_t> image_np, YoloEngine& engine) {
            cv::Mat image = numpy_to_mat(image_np);
            cv::Mat annotated = engine.visualize(image, self.boxes);
            return mat_to_numpy(annotated);
        });
    

    py::class_<YoloEngine>(m, "YoloEngine")
        .def(py::init<const std::string&, float, float>(),
             py::arg("model_path"), 
             py::arg("conf_threshold") = 0.25f,
             py::arg("iou_threshold") = 0.5f)
        

        .def("process", [](YoloEngine& self, py::array_t<uint8_t> image_np) {
            cv::Mat image = numpy_to_mat(image_np);
            return self.process(image);
        }, "Run inference on numpy image")
        
        .def("visualize", [](YoloEngine& self, py::array_t<uint8_t> image_np, 
                            const std::vector<Detection>& detections) {
            cv::Mat image = numpy_to_mat(image_np);
            cv::Mat result = self.visualize(image, detections);
            return mat_to_numpy(result);
        }, "Visualize detections on image")
        .def_property_readonly("names", [](const YoloEngine &self) {
            return self.class_names_;
        });
}
