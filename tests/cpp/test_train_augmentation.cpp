/**
 * @file test_train_augmentation.cpp
 * @brief Unit tests for training augmentation transforms
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "ivit/train/augmentation.hpp"

using namespace ivit::train;

// ============================================================================
// Test Fixtures
// ============================================================================

class AugmentationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a test image (100x100 RGB)
        test_image_ = cv::Mat(100, 100, CV_8UC3, cv::Scalar(128, 64, 32));

        // Create a detection target for testing box transforms
        test_target_.boxes = {
            {10.0f, 20.0f, 50.0f, 60.0f},  // x1, y1, x2, y2
            {30.0f, 40.0f, 80.0f, 90.0f},
        };
        test_target_.labels = {0, 1};
    }

    cv::Mat test_image_;
    DetectionTarget test_target_;
};

// ============================================================================
// Resize Tests
// ============================================================================

TEST_F(AugmentationTest, Resize_SquareResize) {
    Resize resize(64);
    cv::Mat result = resize.apply(test_image_);

    EXPECT_EQ(result.rows, 64);
    EXPECT_EQ(result.cols, 64);
    EXPECT_EQ(result.channels(), 3);
}

TEST_F(AugmentationTest, Resize_RectangularResize) {
    Resize resize(128, 64);
    cv::Mat result = resize.apply(test_image_);

    EXPECT_EQ(result.rows, 128);
    EXPECT_EQ(result.cols, 64);
}

TEST_F(AugmentationTest, Resize_KeepRatio) {
    Resize resize(50, 100, true, 114);
    cv::Mat result = resize.apply(test_image_);

    EXPECT_EQ(result.rows, 50);
    EXPECT_EQ(result.cols, 100);
    // Should have padding
}

TEST_F(AugmentationTest, Resize_UpdatesBoxes) {
    Resize resize(50);  // 50x50 from 100x100, scale = 0.5
    auto [img, target] = resize.apply(test_image_, test_target_);

    // Boxes should be scaled by 0.5
    EXPECT_NEAR(target.boxes[0][0], 5.0f, 0.1f);   // x1
    EXPECT_NEAR(target.boxes[0][1], 10.0f, 0.1f);  // y1
    EXPECT_NEAR(target.boxes[0][2], 25.0f, 0.1f);  // x2
    EXPECT_NEAR(target.boxes[0][3], 30.0f, 0.1f);  // y2
}

// ============================================================================
// RandomHorizontalFlip Tests
// ============================================================================

TEST_F(AugmentationTest, RandomHorizontalFlip_FlipsImage) {
    RandomHorizontalFlip flip(1.0f);  // Always flip
    cv::Mat result = flip.apply(test_image_);

    EXPECT_EQ(result.rows, test_image_.rows);
    EXPECT_EQ(result.cols, test_image_.cols);
}

TEST_F(AugmentationTest, RandomHorizontalFlip_UpdatesBoxes) {
    RandomHorizontalFlip flip(1.0f);
    auto [img, target] = flip.apply(test_image_, test_target_);

    // For horizontal flip, x values are flipped
    // Original box: x1=10, x2=50, width=100
    // Flipped: x1 = 100 - 50 = 50, x2 = 100 - 10 = 90
    EXPECT_NEAR(target.boxes[0][0], 50.0f, 0.1f);  // new x1
    EXPECT_NEAR(target.boxes[0][2], 90.0f, 0.1f);  // new x2
}

TEST_F(AugmentationTest, RandomHorizontalFlip_NoFlip) {
    RandomHorizontalFlip flip(0.0f);  // Never flip
    cv::Mat result = flip.apply(test_image_);

    // Should be a copy, not the same
    EXPECT_NE(result.data, test_image_.data);
}

// ============================================================================
// RandomVerticalFlip Tests
// ============================================================================

TEST_F(AugmentationTest, RandomVerticalFlip_FlipsImage) {
    RandomVerticalFlip flip(1.0f);
    cv::Mat result = flip.apply(test_image_);

    EXPECT_EQ(result.rows, test_image_.rows);
    EXPECT_EQ(result.cols, test_image_.cols);
}

TEST_F(AugmentationTest, RandomVerticalFlip_UpdatesBoxes) {
    RandomVerticalFlip flip(1.0f);
    auto [img, target] = flip.apply(test_image_, test_target_);

    // For vertical flip, y values are flipped
    // Original box: y1=20, y2=60, height=100
    // Flipped: y1 = 100 - 60 = 40, y2 = 100 - 20 = 80
    EXPECT_NEAR(target.boxes[0][1], 40.0f, 0.1f);  // new y1
    EXPECT_NEAR(target.boxes[0][3], 80.0f, 0.1f);  // new y2
}

// ============================================================================
// RandomRotation Tests
// ============================================================================

TEST_F(AugmentationTest, RandomRotation_RotatesImage) {
    RandomRotation rotate(30.0f, 1.0f);
    cv::Mat result = rotate.apply(test_image_);

    EXPECT_EQ(result.rows, test_image_.rows);
    EXPECT_EQ(result.cols, test_image_.cols);
}

// ============================================================================
// ColorJitter Tests
// ============================================================================

TEST_F(AugmentationTest, ColorJitter_AppliesChanges) {
    ColorJitter jitter(0.2f, 0.2f, 0.2f, 0.1f);
    cv::Mat result = jitter.apply(test_image_);

    EXPECT_EQ(result.rows, test_image_.rows);
    EXPECT_EQ(result.cols, test_image_.cols);
    EXPECT_EQ(result.type(), test_image_.type());
}

TEST_F(AugmentationTest, ColorJitter_DoesNotChangeTarget) {
    ColorJitter jitter(0.2f, 0.2f, 0.2f, 0.1f);
    auto [img, target] = jitter.apply(test_image_, test_target_);

    EXPECT_EQ(target.boxes.size(), test_target_.boxes.size());
    EXPECT_EQ(target.boxes[0], test_target_.boxes[0]);
}

// ============================================================================
// Normalize Tests
// ============================================================================

TEST_F(AugmentationTest, Normalize_ImageNet) {
    Normalize normalize;
    cv::Mat result = normalize.apply(test_image_);

    EXPECT_EQ(result.rows, test_image_.rows);
    EXPECT_EQ(result.cols, test_image_.cols);
    EXPECT_EQ(result.type(), CV_32FC3);

    // Values should be roughly centered around 0
    cv::Scalar mean = cv::mean(result);
    EXPECT_LT(std::abs(mean[0]), 5.0);  // Should be small-ish
}

TEST_F(AugmentationTest, Normalize_CustomMeanStd) {
    Normalize normalize({0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f});
    cv::Mat result = normalize.apply(test_image_);

    EXPECT_EQ(result.type(), CV_32FC3);
}

// ============================================================================
// ToTensor Tests
// ============================================================================

TEST_F(AugmentationTest, ToTensor_ConvertToNCHW) {
    ToTensor to_tensor;
    cv::Mat result = to_tensor.apply(test_image_);

    // Should be 4D (1, C, H, W)
    EXPECT_EQ(result.dims, 4);
    EXPECT_EQ(result.size[0], 1);  // N
    EXPECT_EQ(result.size[1], 3);  // C
    EXPECT_EQ(result.size[2], test_image_.rows);  // H
    EXPECT_EQ(result.size[3], test_image_.cols);  // W
}

TEST_F(AugmentationTest, ToTensor_ScalesToFloat) {
    ToTensor to_tensor;
    cv::Mat result = to_tensor.apply(test_image_);

    EXPECT_EQ(result.type(), CV_32FC1);
}

// ============================================================================
// CenterCrop Tests
// ============================================================================

TEST_F(AugmentationTest, CenterCrop_CropsCenter) {
    CenterCrop crop(50);
    cv::Mat result = crop.apply(test_image_);

    EXPECT_EQ(result.rows, 50);
    EXPECT_EQ(result.cols, 50);
}

TEST_F(AugmentationTest, CenterCrop_Rectangular) {
    CenterCrop crop(60, 40);
    cv::Mat result = crop.apply(test_image_);

    EXPECT_EQ(result.rows, 60);
    EXPECT_EQ(result.cols, 40);
}

// ============================================================================
// RandomCrop Tests
// ============================================================================

TEST_F(AugmentationTest, RandomCrop_CropsCorrectSize) {
    RandomCrop crop(50);
    cv::Mat result = crop.apply(test_image_);

    EXPECT_EQ(result.rows, 50);
    EXPECT_EQ(result.cols, 50);
}

// ============================================================================
// GaussianBlur Tests
// ============================================================================

TEST_F(AugmentationTest, GaussianBlur_BlursImage) {
    GaussianBlur blur(5, 1.0f);
    cv::Mat result = blur.apply(test_image_);

    EXPECT_EQ(result.rows, test_image_.rows);
    EXPECT_EQ(result.cols, test_image_.cols);
}

// ============================================================================
// Compose Tests
// ============================================================================

TEST_F(AugmentationTest, Compose_ChainsTransforms) {
    auto compose = std::make_shared<Compose>(std::vector<std::shared_ptr<ITransform>>{
        std::make_shared<Resize>(64),
        std::make_shared<Normalize>(),
        std::make_shared<ToTensor>(),
    });

    cv::Mat result = compose->apply(test_image_);

    EXPECT_EQ(result.dims, 4);
    EXPECT_EQ(result.size[0], 1);  // N
    EXPECT_EQ(result.size[1], 3);  // C
    EXPECT_EQ(result.size[2], 64); // H
    EXPECT_EQ(result.size[3], 64); // W
}

TEST_F(AugmentationTest, Compose_AddTransform) {
    auto compose = std::make_shared<Compose>(std::vector<std::shared_ptr<ITransform>>{});
    compose->add(std::make_shared<Resize>(50));
    compose->add(std::make_shared<Normalize>());

    EXPECT_EQ(compose->size(), 2);

    cv::Mat result = compose->apply(test_image_);
    EXPECT_EQ(result.rows, 50);
    EXPECT_EQ(result.cols, 50);
}

// ============================================================================
// Convenience Functions Tests
// ============================================================================

TEST_F(AugmentationTest, GetDefaultAugmentation) {
    auto transform = get_default_augmentation(224);
    cv::Mat result = transform->apply(test_image_);

    EXPECT_EQ(result.dims, 4);
    EXPECT_EQ(result.size[2], 224);
    EXPECT_EQ(result.size[3], 224);
}

TEST_F(AugmentationTest, GetTrainAugmentation) {
    auto transform = get_train_augmentation(224, 0.5f, true);
    cv::Mat result = transform->apply(test_image_);

    EXPECT_EQ(result.dims, 4);
    EXPECT_EQ(result.size[2], 224);
    EXPECT_EQ(result.size[3], 224);
}

TEST_F(AugmentationTest, GetValAugmentation) {
    auto transform = get_val_augmentation(224);
    cv::Mat result = transform->apply(test_image_);

    EXPECT_EQ(result.dims, 4);
    EXPECT_EQ(result.size[2], 224);
    EXPECT_EQ(result.size[3], 224);
}

// ============================================================================
// Repr Tests
// ============================================================================

TEST_F(AugmentationTest, Transforms_HaveRepr) {
    EXPECT_FALSE(Resize(64).repr().empty());
    EXPECT_FALSE(RandomHorizontalFlip().repr().empty());
    EXPECT_FALSE(RandomVerticalFlip().repr().empty());
    EXPECT_FALSE(RandomRotation(30).repr().empty());
    EXPECT_FALSE(ColorJitter().repr().empty());
    EXPECT_FALSE(Normalize().repr().empty());
    EXPECT_FALSE(ToTensor().repr().empty());
    EXPECT_FALSE(CenterCrop(50).repr().empty());
    EXPECT_FALSE(RandomCrop(50).repr().empty());
    EXPECT_FALSE(GaussianBlur().repr().empty());

    auto compose = std::make_shared<Compose>(std::vector<std::shared_ptr<ITransform>>{
        std::make_shared<Resize>(64),
    });
    EXPECT_FALSE(compose->repr().empty());
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
