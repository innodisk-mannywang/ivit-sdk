/**
 * @file test_model.cpp
 * @brief Unit tests for Model class
 */

#include <gtest/gtest.h>
#include "ivit/core/model.hpp"
#include "ivit/core/tensor.hpp"

using namespace ivit;

class ModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code
    }

    void TearDown() override {
        // Teardown code
    }
};

TEST_F(ModelTest, ModelManagerSingleton) {
    auto& manager1 = ModelManager::instance();
    auto& manager2 = ModelManager::instance();
    EXPECT_EQ(&manager1, &manager2);
}

TEST_F(ModelTest, LoadConfigDefaults) {
    LoadConfig config;
    EXPECT_EQ(config.device, "auto");
    EXPECT_EQ(config.precision, "fp32");
    EXPECT_TRUE(config.use_cache);
}

TEST_F(ModelTest, TensorCreation) {
    std::vector<int64_t> shape = {1, 3, 224, 224};
    Tensor tensor(shape, DataType::Float32);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.dtype(), DataType::Float32);
    EXPECT_EQ(tensor.numel(), 1 * 3 * 224 * 224);
}

TEST_F(ModelTest, TensorByteSize) {
    std::vector<int64_t> shape = {1, 3, 224, 224};

    Tensor fp32_tensor(shape, DataType::Float32);
    EXPECT_EQ(fp32_tensor.byte_size(), 1 * 3 * 224 * 224 * sizeof(float));

    Tensor fp16_tensor(shape, DataType::Float16);
    EXPECT_EQ(fp16_tensor.byte_size(), 1 * 3 * 224 * 224 * 2);

    Tensor int8_tensor(shape, DataType::Int8);
    EXPECT_EQ(int8_tensor.byte_size(), 1 * 3 * 224 * 224 * 1);
}

TEST_F(ModelTest, TensorDataAccess) {
    std::vector<int64_t> shape = {2, 3};
    Tensor tensor(shape, DataType::Float32);

    float* data = tensor.data_ptr<float>();
    EXPECT_NE(data, nullptr);

    // Write and read data
    data[0] = 1.0f;
    data[1] = 2.0f;

    EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(tensor.data_ptr<float>()[1], 2.0f);
}

TEST_F(ModelTest, BBoxIoU) {
    BBox box1(0, 0, 10, 10);
    BBox box2(5, 5, 15, 15);
    BBox box3(20, 20, 30, 30);

    // Overlapping boxes
    float iou12 = box1.iou(box2);
    EXPECT_GT(iou12, 0.0f);
    EXPECT_LT(iou12, 1.0f);

    // Non-overlapping boxes
    float iou13 = box1.iou(box3);
    EXPECT_FLOAT_EQ(iou13, 0.0f);

    // Same box
    float iou11 = box1.iou(box1);
    EXPECT_FLOAT_EQ(iou11, 1.0f);
}

TEST_F(ModelTest, BBoxArea) {
    BBox box(0, 0, 10, 20);
    EXPECT_FLOAT_EQ(box.area(), 200.0f);

    BBox zero_box(5, 5, 5, 5);
    EXPECT_FLOAT_EQ(zero_box.area(), 0.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
