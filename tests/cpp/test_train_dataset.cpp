/**
 * @file test_train_dataset.cpp
 * @brief Unit tests for training dataset classes
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

#include "ivit/train/dataset.hpp"

namespace fs = std::filesystem;

using namespace ivit::train;

// ============================================================================
// Test Fixtures
// ============================================================================

class DatasetTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary test dataset directory
        test_dir_ = fs::temp_directory_path() / "ivit_test_dataset";

        // Clean up if exists
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        // Clean up test directory
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
    }

    void CreateImageFolderDataset() {
        // Create class directories
        fs::create_directories(test_dir_ / "cat");
        fs::create_directories(test_dir_ / "dog");

        // Create dummy images (3x3 RGB)
        cv::Mat img = cv::Mat::ones(64, 64, CV_8UC3) * 128;

        cv::imwrite((test_dir_ / "cat" / "cat1.jpg").string(), img);
        cv::imwrite((test_dir_ / "cat" / "cat2.jpg").string(), img);
        cv::imwrite((test_dir_ / "dog" / "dog1.jpg").string(), img);
        cv::imwrite((test_dir_ / "dog" / "dog2.jpg").string(), img);
        cv::imwrite((test_dir_ / "dog" / "dog3.jpg").string(), img);
    }

    void CreateYOLODataset() {
        fs::create_directories(test_dir_ / "images" / "train");
        fs::create_directories(test_dir_ / "labels" / "train");

        // Create dummy images
        cv::Mat img = cv::Mat::ones(640, 640, CV_8UC3) * 128;
        cv::imwrite((test_dir_ / "images" / "train" / "img1.jpg").string(), img);
        cv::imwrite((test_dir_ / "images" / "train" / "img2.jpg").string(), img);

        // Create labels (YOLO format: class cx cy w h)
        std::ofstream label1(test_dir_ / "labels" / "train" / "img1.txt");
        label1 << "0 0.5 0.5 0.1 0.2\n";
        label1 << "1 0.3 0.3 0.15 0.15\n";
        label1.close();

        std::ofstream label2(test_dir_ / "labels" / "train" / "img2.txt");
        label2 << "0 0.7 0.7 0.2 0.1\n";
        label2.close();

        // Create classes.txt
        std::ofstream classes(test_dir_ / "classes.txt");
        classes << "person\n";
        classes << "car\n";
        classes.close();
    }

    fs::path test_dir_;
};

// ============================================================================
// ImageFolderDataset Tests
// ============================================================================

TEST_F(DatasetTest, ImageFolderDataset_LoadsCorrectly) {
    CreateImageFolderDataset();

    ImageFolderDataset dataset(test_dir_.string(), nullptr, 1.0f, "all");

    EXPECT_EQ(dataset.size(), 5);
    EXPECT_EQ(dataset.num_classes(), 2);

    auto class_names = dataset.class_names();
    EXPECT_EQ(class_names.size(), 2);
    // Classes should be sorted alphabetically
    EXPECT_EQ(class_names[0], "cat");
    EXPECT_EQ(class_names[1], "dog");
}

TEST_F(DatasetTest, ImageFolderDataset_TrainSplit) {
    CreateImageFolderDataset();

    ImageFolderDataset train_dataset(test_dir_.string(), nullptr, 0.6f, "train");
    ImageFolderDataset val_dataset(test_dir_.string(), nullptr, 0.6f, "val");

    // 5 samples * 0.6 = 3 train, 2 val
    EXPECT_EQ(train_dataset.size(), 3);
    EXPECT_EQ(val_dataset.size(), 2);
}

TEST_F(DatasetTest, ImageFolderDataset_GetItem) {
    CreateImageFolderDataset();

    ImageFolderDataset dataset(test_dir_.string(), nullptr, 1.0f, "all");

    auto [image, label] = dataset.get_item(0);

    EXPECT_FALSE(image.empty());
    EXPECT_EQ(image.rows, 64);
    EXPECT_EQ(image.cols, 64);
    EXPECT_EQ(image.channels(), 3);
    EXPECT_GE(label, 0);
    EXPECT_LT(label, 2);
}

TEST_F(DatasetTest, ImageFolderDataset_CalibrationSet) {
    CreateImageFolderDataset();

    ImageFolderDataset dataset(test_dir_.string(), nullptr, 1.0f, "all");

    auto calib = dataset.calibration_set(3);
    EXPECT_EQ(calib.size(), 3);

    for (const auto& img : calib) {
        EXPECT_FALSE(img.empty());
    }
}

TEST_F(DatasetTest, ImageFolderDataset_ThrowsOnInvalidPath) {
    EXPECT_THROW(
        ImageFolderDataset("/nonexistent/path"),
        std::runtime_error
    );
}

// ============================================================================
// YOLODataset Tests
// ============================================================================

TEST_F(DatasetTest, YOLODataset_LoadsCorrectly) {
    CreateYOLODataset();

    YOLODataset dataset(test_dir_.string(), "train");

    EXPECT_EQ(dataset.size(), 2);
    EXPECT_EQ(dataset.num_classes(), 2);

    auto class_names = dataset.class_names();
    EXPECT_EQ(class_names[0], "person");
    EXPECT_EQ(class_names[1], "car");
}

TEST_F(DatasetTest, YOLODataset_GetDetectionItem) {
    CreateYOLODataset();

    YOLODataset dataset(test_dir_.string(), "train");

    auto [image, target] = dataset.get_detection_item(0);

    EXPECT_FALSE(image.empty());
    EXPECT_GE(target.boxes.size(), 1);
    EXPECT_EQ(target.boxes.size(), target.labels.size());

    // Check that boxes are in pixel coordinates
    for (const auto& box : target.boxes) {
        EXPECT_GE(box[0], 0);  // x1
        EXPECT_GE(box[1], 0);  // y1
        EXPECT_LE(box[2], 640);  // x2
        EXPECT_LE(box[3], 640);  // y2
    }
}

TEST_F(DatasetTest, YOLODataset_IsDetectionDataset) {
    CreateYOLODataset();

    YOLODataset dataset(test_dir_.string(), "train");
    EXPECT_TRUE(dataset.is_detection_dataset());
}

// ============================================================================
// split_dataset Tests
// ============================================================================

TEST_F(DatasetTest, SplitDataset_CorrectSizes) {
    CreateImageFolderDataset();

    ImageFolderDataset dataset(test_dir_.string(), nullptr, 1.0f, "all");

    auto [train_idx, val_idx] = split_dataset(dataset, 0.8f, 42);

    EXPECT_EQ(train_idx.size() + val_idx.size(), dataset.size());
    EXPECT_EQ(train_idx.size(), 4);  // 5 * 0.8 = 4
    EXPECT_EQ(val_idx.size(), 1);
}

TEST_F(DatasetTest, SplitDataset_Reproducible) {
    CreateImageFolderDataset();

    ImageFolderDataset dataset(test_dir_.string(), nullptr, 1.0f, "all");

    auto [train1, val1] = split_dataset(dataset, 0.8f, 42);
    auto [train2, val2] = split_dataset(dataset, 0.8f, 42);

    EXPECT_EQ(train1, train2);
    EXPECT_EQ(val1, val2);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
