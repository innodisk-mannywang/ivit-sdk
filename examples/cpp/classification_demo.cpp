/**
 * @file classification_demo.cpp
 * @brief Image classification demo
 */

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "ivit/vision/classifier.hpp"

using namespace ivit;
using namespace ivit::vision;

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " <model_path> <image_path> [device]\n";
    std::cout << "\n";
    std::cout << "Arguments:\n";
    std::cout << "  model_path  Path to classification model (ONNX, OpenVINO XML, etc.)\n";
    std::cout << "  image_path  Path to input image\n";
    std::cout << "  device      Target device (auto, cpu, gpu:0, npu) [default: auto]\n";
    std::cout << "\n";
    std::cout << "Example:\n";
    std::cout << "  " << program << " efficientnet_b0.onnx cat.jpg gpu:0\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string device = argc > 3 ? argv[3] : "auto";

    try {
        // Load classifier
        std::cout << "Loading model: " << model_path << std::endl;
        std::cout << "Device: " << device << std::endl;

        Classifier classifier(model_path, device);

        // Load and classify image
        std::cout << "Classifying: " << image_path << std::endl;

        auto results = classifier.predict(image_path, 5);

        // Print results
        std::cout << "\n=== Classification Results ===" << std::endl;
        std::cout << "Inference time: " << results.inference_time_ms << " ms" << std::endl;
        std::cout << "Device used: " << results.device_used << std::endl;
        std::cout << "\nTop-5 predictions:" << std::endl;

        for (size_t i = 0; i < results.classifications.size(); i++) {
            const auto& cls = results.classifications[i];
            std::cout << "  " << (i + 1) << ". " << cls.label
                      << " (class " << cls.class_id << "): "
                      << std::fixed << std::setprecision(2)
                      << (cls.score * 100) << "%" << std::endl;
        }

        // Display image with results
        cv::Mat image = cv::imread(image_path);
        if (!image.empty()) {
            // Draw top prediction on image
            if (!results.classifications.empty()) {
                const auto& top = results.classifications[0];
                std::string text = top.label + ": " +
                    std::to_string(static_cast<int>(top.score * 100)) + "%";

                cv::putText(image, text, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            }

            cv::imshow("Classification Result", image);
            std::cout << "\nPress any key to exit..." << std::endl;
            cv::waitKey(0);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
