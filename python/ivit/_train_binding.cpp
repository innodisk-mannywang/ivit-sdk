/**
 * @file _train_binding.cpp
 * @brief Python bindings for iVIT Training Module using pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <opencv2/opencv.hpp>

#include "ivit/train/train.hpp"

namespace py = pybind11;

// ============================================================================
// NumPy <-> cv::Mat conversion helpers (reused from _binding.cpp)
// ============================================================================

namespace {

cv::Mat numpy_to_mat(py::array array) {
    py::buffer_info buf = array.request();

    int cv_type;
    if (buf.format == py::format_descriptor<uint8_t>::format()) {
        cv_type = CV_8U;
    } else if (buf.format == py::format_descriptor<float>::format()) {
        cv_type = CV_32F;
    } else if (buf.format == py::format_descriptor<double>::format()) {
        cv_type = CV_64F;
    } else if (buf.format == py::format_descriptor<int32_t>::format()) {
        cv_type = CV_32S;
    } else {
        throw std::runtime_error("Unsupported numpy dtype for cv::Mat conversion");
    }

    int rows, cols, channels;

    if (buf.ndim == 2) {
        rows = buf.shape[0];
        cols = buf.shape[1];
        channels = 1;
    } else if (buf.ndim == 3) {
        rows = buf.shape[0];
        cols = buf.shape[1];
        channels = buf.shape[2];
    } else if (buf.ndim == 4) {
        // NCHW format - take first batch
        int n = buf.shape[0];
        channels = buf.shape[1];
        rows = buf.shape[2];
        cols = buf.shape[3];
        // Reshape to CHW then HWC
        cv_type = CV_MAKETYPE(cv_type, channels);
        cv::Mat mat(rows, cols, cv_type, buf.ptr);
        return mat.clone();
    } else {
        throw std::runtime_error("Array must be 2D, 3D, or 4D");
    }

    cv_type = CV_MAKETYPE(cv_type, channels);
    cv::Mat mat(rows, cols, cv_type, buf.ptr);
    return mat.clone();
}

py::array mat_to_numpy(const cv::Mat& mat) {
    if (mat.empty()) {
        return py::array();
    }

    std::string format;
    size_t elem_size;

    int depth = mat.depth();
    switch (depth) {
        case CV_8U:  format = py::format_descriptor<uint8_t>::format(); elem_size = 1; break;
        case CV_32F: format = py::format_descriptor<float>::format(); elem_size = 4; break;
        case CV_64F: format = py::format_descriptor<double>::format(); elem_size = 8; break;
        default:
            throw std::runtime_error("Unsupported cv::Mat type");
    }

    int channels = mat.channels();

    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;

    // Handle multi-dimensional matrices (e.g., 4D tensors from ToTensor)
    if (mat.dims > 2) {
        // Multi-dimensional matrix
        ssize_t stride = elem_size;
        for (int i = mat.dims - 1; i >= 0; --i) {
            shape.insert(shape.begin(), mat.size[i]);
            strides.insert(strides.begin(), stride);
            stride *= mat.size[i];
        }
    } else if (channels == 1) {
        shape = {mat.rows, mat.cols};
        strides = {static_cast<ssize_t>(mat.step[0]), static_cast<ssize_t>(elem_size)};
    } else {
        shape = {mat.rows, mat.cols, channels};
        strides = {static_cast<ssize_t>(mat.step[0]),
                   static_cast<ssize_t>(channels * elem_size),
                   static_cast<ssize_t>(elem_size)};
    }

    py::array result(py::buffer_info(
        mat.data, elem_size, format, shape.size(), shape, strides
    ));

    return result.attr("copy")();
}

} // anonymous namespace

// ============================================================================
// Python Module Definition
// ============================================================================

PYBIND11_MODULE(_ivit_train_core, m) {
    m.doc() = "iVIT-SDK Training Module C++ bindings";

    // ========================================================================
    // Module Functions
    // ========================================================================

    m.def("is_training_available", &ivit::train::is_training_available,
          "Check if training module (LibTorch) is available");

    m.def("training_version", &ivit::train::training_version,
          "Get training module version");

    // ========================================================================
    // Configuration Structures
    // ========================================================================

    py::class_<ivit::train::TrainerConfig>(m, "TrainerConfig",
        "Training configuration")
        .def(py::init<>())
        .def_readwrite("epochs", &ivit::train::TrainerConfig::epochs)
        .def_readwrite("learning_rate", &ivit::train::TrainerConfig::learning_rate)
        .def_readwrite("batch_size", &ivit::train::TrainerConfig::batch_size)
        .def_readwrite("optimizer", &ivit::train::TrainerConfig::optimizer)
        .def_readwrite("device", &ivit::train::TrainerConfig::device)
        .def_readwrite("freeze_backbone", &ivit::train::TrainerConfig::freeze_backbone)
        .def_readwrite("num_workers", &ivit::train::TrainerConfig::num_workers)
        .def_readwrite("weight_decay", &ivit::train::TrainerConfig::weight_decay)
        .def_readwrite("momentum", &ivit::train::TrainerConfig::momentum)
        .def_readwrite("mixed_precision", &ivit::train::TrainerConfig::mixed_precision)
        .def_readwrite("log_interval", &ivit::train::TrainerConfig::log_interval)
        .def_readwrite("checkpoint_dir", &ivit::train::TrainerConfig::checkpoint_dir)
        .def_readwrite("seed", &ivit::train::TrainerConfig::seed)
        .def("__repr__", [](const ivit::train::TrainerConfig& c) {
            return "<TrainerConfig epochs=" + std::to_string(c.epochs) +
                   " lr=" + std::to_string(c.learning_rate) +
                   " device='" + c.device + "'>";
        });

    py::class_<ivit::train::ExportOptions>(m, "ExportOptions",
        "Model export options")
        .def(py::init<>())
        .def_readwrite("format", &ivit::train::ExportOptions::format)
        .def_readwrite("optimize_for", &ivit::train::ExportOptions::optimize_for)
        .def_readwrite("quantize", &ivit::train::ExportOptions::quantize)
        .def_readwrite("input_shape", &ivit::train::ExportOptions::input_shape)
        .def_readwrite("opset_version", &ivit::train::ExportOptions::opset_version)
        .def_readwrite("dynamic_batch", &ivit::train::ExportOptions::dynamic_batch)
        .def_readwrite("calibration_data", &ivit::train::ExportOptions::calibration_data);

    py::class_<ivit::train::TrainingMetrics>(m, "TrainingMetrics",
        "Training metrics for an epoch")
        .def(py::init<>())
        .def_readwrite("epoch", &ivit::train::TrainingMetrics::epoch)
        .def_readwrite("loss", &ivit::train::TrainingMetrics::loss)
        .def_readwrite("accuracy", &ivit::train::TrainingMetrics::accuracy)
        .def_readwrite("val_loss", &ivit::train::TrainingMetrics::val_loss)
        .def_readwrite("val_accuracy", &ivit::train::TrainingMetrics::val_accuracy)
        .def_readwrite("learning_rate", &ivit::train::TrainingMetrics::learning_rate)
        .def_readwrite("elapsed_seconds", &ivit::train::TrainingMetrics::elapsed_seconds)
        .def("__repr__", [](const ivit::train::TrainingMetrics& m) {
            return "<TrainingMetrics epoch=" + std::to_string(m.epoch) +
                   " loss=" + std::to_string(m.loss) +
                   " accuracy=" + std::to_string(m.accuracy) + ">";
        });

    // ========================================================================
    // Detection Target
    // ========================================================================

    py::class_<ivit::train::DetectionTarget>(m, "DetectionTarget",
        "Detection target with boxes and labels")
        .def(py::init<>())
        .def_readwrite("boxes", &ivit::train::DetectionTarget::boxes)
        .def_readwrite("labels", &ivit::train::DetectionTarget::labels)
        .def_readwrite("image_id", &ivit::train::DetectionTarget::image_id);

    // ========================================================================
    // Dataset Classes
    // ========================================================================

    py::class_<ivit::train::IDataset, std::shared_ptr<ivit::train::IDataset>>(
        m, "IDataset", "Abstract base class for datasets")
        .def("size", &ivit::train::IDataset::size)
        .def("num_classes", &ivit::train::IDataset::num_classes)
        .def("class_names", &ivit::train::IDataset::class_names)
        .def("is_detection_dataset", &ivit::train::IDataset::is_detection_dataset)
        .def("__len__", &ivit::train::IDataset::size)
        .def("__getitem__", [](ivit::train::IDataset& ds, size_t idx) {
            auto [image, label] = ds.get_item(idx);
            return py::make_tuple(mat_to_numpy(image), label);
        });

    py::class_<ivit::train::ImageFolderDataset,
               ivit::train::IDataset,
               std::shared_ptr<ivit::train::ImageFolderDataset>>(
        m, "ImageFolderDataset",
        R"doc(
Dataset that loads images from folder structure.

Expected structure:
    root/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img3.jpg

Args:
    root: Root directory containing class folders
    transform: Transform function or ITransform object (optional)
    train_split: Fraction of data for training (0-1)
    split: "train", "val", or "all"
    seed: Random seed for reproducible splits

Examples:
    >>> dataset = ImageFolderDataset("./data", train_split=0.8, split="train")
    >>> image, label = dataset[0]
)doc")
        .def(py::init([](const std::string& root,
                         py::object transform,
                         float train_split,
                         const std::string& split,
                         int seed) {
            std::function<cv::Mat(const cv::Mat&)> transform_fn = nullptr;
            if (!transform.is_none()) {
                // Check if it's an ITransform subclass
                try {
                    auto itransform = transform.cast<std::shared_ptr<ivit::train::ITransform>>();
                    transform_fn = [itransform](const cv::Mat& img) {
                        return itransform->apply(img);
                    };
                } catch (const py::cast_error&) {
                    // Try as callable
                    transform_fn = [transform](const cv::Mat& img) {
                        py::gil_scoped_acquire acquire;
                        py::array np_img = mat_to_numpy(img);
                        py::object result = transform(np_img);
                        return numpy_to_mat(result.cast<py::array>());
                    };
                }
            }
            return std::make_shared<ivit::train::ImageFolderDataset>(
                root, transform_fn, train_split, split, seed);
        }),
             py::arg("root"),
             py::arg("transform") = py::none(),
             py::arg("train_split") = 0.8f,
             py::arg("split") = "train",
             py::arg("seed") = 42)
        .def("calibration_set", [](ivit::train::ImageFolderDataset& ds, size_t n) {
            auto images = ds.calibration_set(n);
            py::list result;
            for (const auto& img : images) {
                result.append(mat_to_numpy(img));
            }
            return result;
        }, py::arg("n_samples") = 100);

    py::class_<ivit::train::COCODataset,
               ivit::train::IDataset,
               std::shared_ptr<ivit::train::COCODataset>>(
        m, "COCODataset", "Dataset for COCO format annotations")
        .def(py::init<const std::string&,
                      const std::string&,
                      std::function<std::tuple<cv::Mat, ivit::train::DetectionTarget>(
                          const cv::Mat&, const ivit::train::DetectionTarget&)>,
                      const std::string&>(),
             py::arg("root"),
             py::arg("annotation_file"),
             py::arg("transform") = nullptr,
             py::arg("split") = "train")
        .def("get_detection_item", [](ivit::train::COCODataset& ds, size_t idx) {
            auto [image, target] = ds.get_detection_item(idx);
            return py::make_tuple(mat_to_numpy(image), target);
        });

    py::class_<ivit::train::YOLODataset,
               ivit::train::IDataset,
               std::shared_ptr<ivit::train::YOLODataset>>(
        m, "YOLODataset", "Dataset for YOLO format annotations")
        .def(py::init<const std::string&,
                      const std::string&,
                      std::function<std::tuple<cv::Mat, ivit::train::DetectionTarget>(
                          const cv::Mat&, const ivit::train::DetectionTarget&)>,
                      const std::vector<std::string>&>(),
             py::arg("root"),
             py::arg("split") = "train",
             py::arg("transform") = nullptr,
             py::arg("class_names") = std::vector<std::string>{})
        .def("get_detection_item", [](ivit::train::YOLODataset& ds, size_t idx) {
            auto [image, target] = ds.get_detection_item(idx);
            return py::make_tuple(mat_to_numpy(image), target);
        });

    m.def("split_dataset", &ivit::train::split_dataset,
          py::arg("dataset"),
          py::arg("train_ratio") = 0.8f,
          py::arg("seed") = 42,
          "Split dataset indices into train and validation sets");

    // ========================================================================
    // Transform Classes
    // ========================================================================

    py::class_<ivit::train::ITransform, std::shared_ptr<ivit::train::ITransform>>(
        m, "ITransform", "Abstract base class for transforms")
        .def("apply", [](ivit::train::ITransform& t, py::array image) {
            return mat_to_numpy(t.apply(numpy_to_mat(image)));
        })
        .def("__call__", [](ivit::train::ITransform& t, py::array image) {
            return mat_to_numpy(t.apply(numpy_to_mat(image)));
        })
        .def("__repr__", &ivit::train::ITransform::repr);

    py::class_<ivit::train::Compose,
               ivit::train::ITransform,
               std::shared_ptr<ivit::train::Compose>>(
        m, "Compose", "Compose multiple transforms")
        .def(py::init<std::vector<std::shared_ptr<ivit::train::ITransform>>>(),
             py::arg("transforms"))
        .def("add", &ivit::train::Compose::add)
        .def("__len__", &ivit::train::Compose::size);

    py::class_<ivit::train::Resize,
               ivit::train::ITransform,
               std::shared_ptr<ivit::train::Resize>>(
        m, "Resize", "Resize image")
        .def(py::init<int, bool, int>(),
             py::arg("size"),
             py::arg("keep_ratio") = false,
             py::arg("pad_value") = 114)
        .def(py::init<int, int, bool, int>(),
             py::arg("height"),
             py::arg("width"),
             py::arg("keep_ratio") = false,
             py::arg("pad_value") = 114);

    py::class_<ivit::train::RandomHorizontalFlip,
               ivit::train::ITransform,
               std::shared_ptr<ivit::train::RandomHorizontalFlip>>(
        m, "RandomHorizontalFlip", "Random horizontal flip")
        .def(py::init<float>(), py::arg("p") = 0.5f);

    py::class_<ivit::train::RandomVerticalFlip,
               ivit::train::ITransform,
               std::shared_ptr<ivit::train::RandomVerticalFlip>>(
        m, "RandomVerticalFlip", "Random vertical flip")
        .def(py::init<float>(), py::arg("p") = 0.5f);

    py::class_<ivit::train::RandomRotation,
               ivit::train::ITransform,
               std::shared_ptr<ivit::train::RandomRotation>>(
        m, "RandomRotation", "Random rotation")
        .def(py::init<float, float>(),
             py::arg("degrees"),
             py::arg("p") = 0.5f);

    py::class_<ivit::train::ColorJitter,
               ivit::train::ITransform,
               std::shared_ptr<ivit::train::ColorJitter>>(
        m, "ColorJitter", "Random color jittering")
        .def(py::init<float, float, float, float>(),
             py::arg("brightness") = 0.2f,
             py::arg("contrast") = 0.2f,
             py::arg("saturation") = 0.2f,
             py::arg("hue") = 0.1f);

    py::class_<ivit::train::Normalize,
               ivit::train::ITransform,
               std::shared_ptr<ivit::train::Normalize>>(
        m, "Normalize", "Normalize with mean and std")
        .def(py::init<>())
        .def(py::init<const std::vector<float>&, const std::vector<float>&>(),
             py::arg("mean"),
             py::arg("std"));

    py::class_<ivit::train::ToTensor,
               ivit::train::ITransform,
               std::shared_ptr<ivit::train::ToTensor>>(
        m, "ToTensor", "Convert to NCHW tensor format")
        .def(py::init<>());

    py::class_<ivit::train::CenterCrop,
               ivit::train::ITransform,
               std::shared_ptr<ivit::train::CenterCrop>>(
        m, "CenterCrop", "Center crop")
        .def(py::init<int>(), py::arg("size"))
        .def(py::init<int, int>(), py::arg("height"), py::arg("width"));

    py::class_<ivit::train::RandomCrop,
               ivit::train::ITransform,
               std::shared_ptr<ivit::train::RandomCrop>>(
        m, "RandomCrop", "Random crop")
        .def(py::init<int>(), py::arg("size"))
        .def(py::init<int, int>(), py::arg("height"), py::arg("width"));

    py::class_<ivit::train::GaussianBlur,
               ivit::train::ITransform,
               std::shared_ptr<ivit::train::GaussianBlur>>(
        m, "GaussianBlur", "Gaussian blur")
        .def(py::init<int, float>(),
             py::arg("kernel_size") = 5,
             py::arg("sigma") = 0.0f);

    // Convenience functions
    m.def("get_default_augmentation", &ivit::train::get_default_augmentation,
          py::arg("size") = 224);
    m.def("get_train_augmentation", &ivit::train::get_train_augmentation,
          py::arg("size") = 224,
          py::arg("flip_p") = 0.5f,
          py::arg("color_jitter") = true);
    m.def("get_val_augmentation", &ivit::train::get_val_augmentation,
          py::arg("size") = 224);

    // ========================================================================
    // Callback Classes
    // ========================================================================

    py::class_<ivit::train::ITrainingCallback,
               std::shared_ptr<ivit::train::ITrainingCallback>>(
        m, "ITrainingCallback", "Abstract base class for training callbacks")
        .def("__repr__", &ivit::train::ITrainingCallback::repr);

    py::class_<ivit::train::EarlyStopping,
               ivit::train::ITrainingCallback,
               std::shared_ptr<ivit::train::EarlyStopping>>(
        m, "EarlyStopping", "Stop training when metric stops improving")
        .def(py::init<const std::string&, int, float, const std::string&>(),
             py::arg("monitor") = "val_loss",
             py::arg("patience") = 5,
             py::arg("min_delta") = 0.0f,
             py::arg("mode") = "min")
        .def_property_readonly("stopped_epoch", &ivit::train::EarlyStopping::stopped_epoch)
        .def_property_readonly("should_stop", &ivit::train::EarlyStopping::should_stop);

    py::class_<ivit::train::ModelCheckpoint,
               ivit::train::ITrainingCallback,
               std::shared_ptr<ivit::train::ModelCheckpoint>>(
        m, "ModelCheckpoint", "Save model checkpoints")
        .def(py::init<const std::string&, const std::string&, bool, const std::string&>(),
             py::arg("filepath") = "checkpoint.pt",
             py::arg("monitor") = "val_loss",
             py::arg("save_best_only") = true,
             py::arg("mode") = "min")
        .def_property_readonly("best_path", &ivit::train::ModelCheckpoint::best_path);

    py::class_<ivit::train::ProgressLogger,
               ivit::train::ITrainingCallback,
               std::shared_ptr<ivit::train::ProgressLogger>>(
        m, "ProgressLogger", "Log training progress")
        .def(py::init<int>(), py::arg("log_frequency") = 10);

    py::class_<ivit::train::LRScheduler,
               ivit::train::ITrainingCallback,
               std::shared_ptr<ivit::train::LRScheduler>>(
        m, "LRScheduler", "Learning rate scheduler")
        .def(py::init<const std::string&, int, float, int>(),
             py::arg("scheduler_type") = "step",
             py::arg("step_size") = 10,
             py::arg("gamma") = 0.1f,
             py::arg("patience") = 5)
        .def_property_readonly("current_lr", &ivit::train::LRScheduler::current_lr);

    py::class_<ivit::train::TensorBoardLogger,
               ivit::train::ITrainingCallback,
               std::shared_ptr<ivit::train::TensorBoardLogger>>(
        m, "TensorBoardLogger", "Log to TensorBoard")
        .def(py::init<const std::string&>(), py::arg("log_dir") = "runs");

    py::class_<ivit::train::CSVLogger,
               ivit::train::ITrainingCallback,
               std::shared_ptr<ivit::train::CSVLogger>>(
        m, "CSVLogger", "Log to CSV file")
        .def(py::init<const std::string&, bool>(),
             py::arg("filepath") = "training_log.csv",
             py::arg("append") = false);

    // ========================================================================
    // Trainer
    // ========================================================================

    py::class_<ivit::train::Trainer>(m, "Trainer",
        R"doc(
Trainer for fine-tuning models with transfer learning.

Args:
    model: Model name (e.g., "resnet50") or path to checkpoint
    dataset: Training dataset
    config: Training configuration
    val_dataset: Validation dataset (optional)

Examples:
    >>> dataset = ImageFolderDataset("./data", train_split=0.8)
    >>> config = TrainerConfig()
    >>> config.epochs = 20
    >>> config.learning_rate = 0.001
    >>> trainer = Trainer("resnet50", dataset, config)
    >>> history = trainer.fit([EarlyStopping(), ModelCheckpoint()])
    >>> trainer.export_model("model.onnx")
)doc")
        .def(py::init<const std::string&,
                      std::shared_ptr<ivit::train::IDataset>,
                      const ivit::train::TrainerConfig&,
                      std::shared_ptr<ivit::train::IDataset>>(),
             py::arg("model"),
             py::arg("dataset"),
             py::arg("config") = ivit::train::TrainerConfig{},
             py::arg("val_dataset") = nullptr)

        // Training methods
        .def("fit", &ivit::train::Trainer::fit,
             py::arg("callbacks") = std::vector<std::shared_ptr<ivit::train::ITrainingCallback>>{},
             "Train the model")
        .def("evaluate", &ivit::train::Trainer::evaluate,
             py::arg("dataset") = nullptr,
             "Evaluate model on dataset")
        .def("export_model", &ivit::train::Trainer::export_model,
             py::arg("path"),
             py::arg("options") = ivit::train::ExportOptions{},
             "Export model to deployment format")

        // Checkpoint methods
        .def("load_checkpoint", &ivit::train::Trainer::load_checkpoint,
             py::arg("path"))
        .def("save_checkpoint", &ivit::train::Trainer::save_checkpoint,
             py::arg("path"))

        // Backbone control
        .def("unfreeze_backbone", &ivit::train::Trainer::unfreeze_backbone,
             py::arg("num_layers") = -1)
        .def("freeze_backbone", &ivit::train::Trainer::freeze_backbone)
        .def("set_learning_rate", &ivit::train::Trainer::set_learning_rate,
             py::arg("lr"))

        // Properties
        .def_property_readonly("config", &ivit::train::Trainer::config)
        .def_property_readonly("history", &ivit::train::Trainer::history)
        .def_property_readonly("current_epoch", &ivit::train::Trainer::current_epoch)
        .def_property_readonly("epochs", &ivit::train::Trainer::epochs)
        .def_property_readonly("batches_per_epoch", &ivit::train::Trainer::batches_per_epoch)
        .def_property_readonly("learning_rate", &ivit::train::Trainer::learning_rate)
        .def_property_readonly("num_classes", &ivit::train::Trainer::num_classes)
        .def_property_readonly("class_names", &ivit::train::Trainer::class_names)
        .def_property_readonly("device", &ivit::train::Trainer::device)
        .def_property_readonly("model_name", &ivit::train::Trainer::model_name)
        .def_property_readonly("is_cuda", &ivit::train::Trainer::is_cuda)

        // Static methods
        .def_static("available_models", &ivit::train::Trainer::available_models)
        .def_static("is_torch_available", &ivit::train::Trainer::is_torch_available)
        .def_static("is_cuda_available", &ivit::train::Trainer::is_cuda_available)
        .def_static("torch_version", &ivit::train::Trainer::torch_version)

        .def("__repr__", [](const ivit::train::Trainer& t) {
            return "<Trainer model='" + t.model_name() +
                   "' device='" + t.device() + "'>";
        });

    // ========================================================================
    // ModelExporter
    // ========================================================================

    py::class_<ivit::train::ModelExporter>(m, "ModelExporter",
        "Export trained models to deployment formats")
        .def(py::init<const std::string&>(), py::arg("checkpoint_path"))
        .def("export_model", &ivit::train::ModelExporter::export_model,
             py::arg("path"),
             py::arg("options") = ivit::train::ExportOptions{})
        .def("export_onnx", &ivit::train::ModelExporter::export_onnx,
             py::arg("path"),
             py::arg("input_shape") = std::vector<int64_t>{1, 3, 224, 224},
             py::arg("opset_version") = 17,
             py::arg("dynamic_batch") = true)
        .def("export_torchscript", &ivit::train::ModelExporter::export_torchscript,
             py::arg("path"),
             py::arg("input_shape") = std::vector<int64_t>{1, 3, 224, 224})
        .def("set_class_names", &ivit::train::ModelExporter::set_class_names)
        .def_property_readonly("class_names", &ivit::train::ModelExporter::class_names)
        .def_static("quantize_onnx_fp16", &ivit::train::ModelExporter::quantize_onnx_fp16);

    m.def("export_model", &ivit::train::export_model,
          py::arg("model_or_checkpoint"),
          py::arg("path"),
          py::arg("options") = ivit::train::ExportOptions{},
          "Convenience function to export a model");
}
