#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "OCSort.hpp"

namespace py = pybind11;
using namespace ocsort;

PYBIND11_MODULE(ocsort_module, m) {
    py::class_<ocsort::OCSort>(m, "OCSort")
        .def(py::init<float, int, int, float, int, std::string, float, int, bool>(),
             py::arg("det_thresh") = 0.2,
             py::arg("max_age") = 30,
             py::arg("min_hits") = 3,
             py::arg("iou_threshold") = 0.3,
             py::arg("delta_t") = 3,
             py::arg("asso_func") = "iou",
             py::arg("inertia") = 0.2,
             py::arg("max_kalman") = 5,
             py::arg("use_byte") = false)
        .def("update", [](ocsort::OCSort& self, py::array_t<float> detections) {
            if (detections.ndim() != 2 || detections.shape(1) != 6) {
                throw std::runtime_error("Detections must be (N, 6) array");
            }
            
            
            auto buf = detections.request();
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 6, Eigen::RowMajor>> 
                eigen_dets(static_cast<float*>(buf.ptr), buf.shape[0], buf.shape[1]);
            
            std::vector<Eigen::RowVectorXf> results = self.update(eigen_dets); 
            
            if (results.empty()) {
                py::array_t<float> output = py::array_t<float>(0 * 8);
                output.resize({0, 8});
                return output;
            }
            
            py::array_t<float> output({static_cast<py::ssize_t>(results.size()), static_cast<py::ssize_t>(8)});
            auto out_buf = output.request();
            float* ptr = static_cast<float*>(out_buf.ptr);
            
            for (size_t i = 0; i < results.size(); ++i) {
                for (int j = 0; j < 8; ++j) {
                    ptr[i * 8 + j] = results[i](j);
                }
            }
            
            return output;
        })
        .def_readwrite("max_age", &ocsort::OCSort::max_age)
        .def_readwrite("min_hits", &ocsort::OCSort::min_hits)
        .def_readwrite("iou_threshold", &ocsort::OCSort::iou_threshold)
        .def_readwrite("frame_count", &ocsort::OCSort::frame_count);
}
