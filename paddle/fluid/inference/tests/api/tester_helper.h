// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <vector>

#ifdef WITH_GPERFTOOLS
#include <gperftools/profiler.h>
#endif
#include <highgui.hpp>
#include <opencv.hpp>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/analysis_predictor.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/tests/api/config_printer.h"
#include "paddle/fluid/inference/tests/test_helper.h"
#include "paddle/fluid/inference/utils/benchmark.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_string(model_name, "", "model name");
DEFINE_string(infer_model, "", "model path");
DEFINE_string(infer_data, "", "data file");
DEFINE_string(refer_result, "", "reference result for comparison");
DEFINE_int32(batch_size, 1, "batch size");
DEFINE_int32(warmup_batch_size, 100, "batch size for quantization warmup");
// setting iterations to 0 means processing the whole dataset
DEFINE_int32(iterations, 0, "number of batches to process");
DEFINE_int32(repeat, 1, "Running the inference program repeat times.");
DEFINE_bool(test_all_data, false, "Test the all dataset in data file.");
DEFINE_int32(num_threads, 1, "Running the inference program in multi-threads.");
DEFINE_bool(use_analysis, true,
            "Running the inference program in analysis mode.");

DEFINE_double(accuracy, 1e-3, "Result Accuracy.");
DEFINE_double(quantized_accuracy, 1e-2, "Result Quantized Accuracy.");
DEFINE_bool(zero_copy, false, "Use ZeroCopy to speedup Feed/Fetch.");
DEFINE_bool(warmup, false,
            "Use warmup to calculate elapsed_time more accurately. "
            "To reduce CI time, it sets false in default.");

DEFINE_string(title, "", "title");
DEFINE_bool(is_int8, false, "is int8 and vnni");
DEFINE_string(image_data_path, "", "image_path");
DEFINE_double(render_confidence, 0.2, "render confidence");
DEFINE_string(images_list_path, "", "images list path");
DEFINE_bool(draw_gt, false, "draw ground truth bounding box");

DECLARE_bool(profile);
DECLARE_int32(paddle_num_threads);

namespace paddle {
namespace inference {

using paddle::framework::proto::VarType;

std::vector<std::string> read_labels(std::string path) {
  std::vector<std::string> result;
  std::ifstream labels(path);
  std::string tmp;
  while (std::getline(labels, tmp)) {
    result.push_back(tmp);
  }

  return result;
}

template <typename T>
constexpr paddle::PaddleDType GetPaddleDType();

template <>
constexpr paddle::PaddleDType GetPaddleDType<int64_t>() {
  return paddle::PaddleDType::INT64;
}

template <>
constexpr paddle::PaddleDType GetPaddleDType<float>() {
  return paddle::PaddleDType::FLOAT32;
}

void PrintConfig(const PaddlePredictor::Config *config, bool use_analysis) {
  const auto *analysis_config =
      reinterpret_cast<const AnalysisConfig *>(config);
  if (use_analysis) {
    LOG(INFO) << *analysis_config;
    return;
  }
  LOG(INFO) << analysis_config->ToNativeConfig();
}

std::unique_ptr<PaddlePredictor> CreateTestPredictor(
    const PaddlePredictor::Config *config, bool use_analysis = true) {
  const auto *analysis_config =
      reinterpret_cast<const AnalysisConfig *>(config);
  if (use_analysis) {
    return CreatePaddlePredictor<AnalysisConfig>(*analysis_config);
  }
  auto native_config = analysis_config->ToNativeConfig();
  return CreatePaddlePredictor<NativeConfig>(native_config);
}

void PredictionWarmUp(PaddlePredictor *predictor,
                      const std::vector<std::vector<PaddleTensor>> &inputs,
                      std::vector<std::vector<PaddleTensor>> *outputs,
                      int num_threads, int tid,
                      const VarType::Type data_type = VarType::FP32) {
  int batch_size = FLAGS_batch_size;
  LOG(INFO) << "Running thread " << tid << ", warm up run...";
  outputs->resize(1);
  Timer warmup_timer;
  warmup_timer.tic();
  if (!FLAGS_zero_copy) {
    predictor->Run(inputs[0], &(*outputs)[0], batch_size);
  } else {
    predictor->ZeroCopyRun();
  }
  PrintTime(batch_size, 1, num_threads, tid, warmup_timer.toc(), 1, data_type);
  if (FLAGS_profile) {
    paddle::platform::ResetProfiler();
  }
}

void PredictionRun(PaddlePredictor *predictor,
                   const std::vector<std::vector<PaddleTensor>> &inputs,
                   std::vector<std::vector<PaddleTensor>> *outputs,
                   int num_threads, int tid,
                   const VarType::Type data_type = VarType::FP32) {
  int num_times = FLAGS_repeat;
  int iterations = inputs.size();
  LOG(INFO) << iterations << ";" << FLAGS_batch_size;
  LOG(INFO) << "Thread " << tid << ", number of threads " << num_threads
            << ", run " << num_times << " times...";

  std::vector<std::string> images_list = read_labels(FLAGS_images_list_path);
  cv::namedWindow(FLAGS_title, CV_WINDOW_NORMAL);

  std::vector<std::string> text_labels{
      "background", "aeroplane",   "bicycle", "bird",  "boat",
      "bottle",     "bus",         "car",     "cat",   "chair",
      "cow",        "diningtable", "dog",     "horse", "motorbike",
      "person",     "pottedplant", "sheep",   "sofa",  "train",
      "tvmonitor"};

  std::vector<cv::Scalar> colors;
  for (int i = 0; i < 255;) {
    for (int j = 0; j < 255;) {
      for (int k = 0; k < 255;) {
        colors.push_back(cv::Scalar(i, j, k));
        k += 70;
      }
      j += 70;
    }
    i += 70;
  }

  cv::Scalar gt_color = cv::Scalar(255, 255, 255);

  int TEXT_BACKGROUND_H = 20.0;
  for (int i = 0; i < iterations; i++) {
    LOG(INFO) << "reading image batch id " << i;
    std::string image_name = images_list[i];
    LOG(INFO) << "reading image name " << image_name;
    std::string image_path = FLAGS_image_data_path + std::string("/") +
                             image_name + std::string(".jpg");
    LOG(INFO) << "reading image " << image_path;
    cv::Mat img = cv::imread(image_path, 1);
    LOG(INFO) << img.channels() << ", " << img.rows << ", " << img.cols;
    cv::Mat img2;
    cv::resize(img, img2, cv::Size(static_cast<int>(img.cols * 3),
                                   static_cast<int>(img.rows * 3)),
               0, 0, cv::INTER_AREA);
    LOG(INFO) << img2.channels() << ", " << img2.rows << ", " << img2.cols;

    std::vector<PaddleTensor> output;

    Timer run_timer;
    run_timer.tic();
    predictor->Run(inputs[i], &output, FLAGS_batch_size);
    double elapsed_time = run_timer.toc();

    PaddleTensor bboxes = output[0];
    PaddleTensor top1 = output[1];
    PaddleTensor top5 = output[2];

    float num = static_cast<float>(std::accumulate(
        bboxes.shape.begin(), bboxes.shape.end(), 1, std::multiplies<int>()));
    LOG(INFO) << "num of prediction " << num;
    float *bbs = static_cast<float *>(bboxes.data.data());

    // draw predict bbox
    for (int k = 0; k < static_cast<int>(num);) {
      int label_index = static_cast<int>(bbs[k++]);
      if (label_index == 0) {
        continue;
      }
      std::string text_label = text_labels[label_index];
      float confidence = bbs[k++];
      float x1 = std::min(std::max(bbs[k++], 0.0f), 0.993f) * img2.cols;
      float y1 = std::min(std::max(bbs[k++], 0.0f), 0.993f) * img2.rows;
      float x2 = std::min(std::max(bbs[k++], 0.0f), 0.993f) * img2.cols;
      float y2 = std::min(std::max(bbs[k++], 0.0f), 0.993f) * img2.rows;
      if (confidence > FLAGS_render_confidence) {
        LOG(INFO) << "rendering predict bbox (" << x1 << ", " << y1 << "), ("
                  << x2 << ", " << y2 << ")";

        cv::rectangle(img2, cv::Point(x1, y1), cv::Point(x2, y2),
                      colors[label_index], 2);

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(
            text_label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 0.5, &baseline);

        if ((y1 - TEXT_BACKGROUND_H) < 0.0) {
          cv::rectangle(
              img2, cv::Point(x1 - 1, y1),
              cv::Point(x1 + text_size.width + 1, y1 + TEXT_BACKGROUND_H),
              colors[label_index], -1);

          cv::putText(
              img2, text_label,
              cv::Point(x1, y1 + TEXT_BACKGROUND_H / 2 + text_size.height / 2),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 0.5);
        } else {
          cv::rectangle(img2, cv::Point(x1 - 1, y1 - TEXT_BACKGROUND_H),
                        cv::Point(x1 + text_size.width + 1, y1),
                        colors[label_index], -1);

          cv::putText(img2, text_label,
                      cv::Point(x1, y1 - (TEXT_BACKGROUND_H / 2 -
                                          text_size.height / 2)),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 0.5);
        }
      }
    }

    // draw gt bbox
    if (FLAGS_draw_gt) {
      PaddleTensor gt_labels = inputs[i][2];
      float num = static_cast<float>(std::accumulate(gt_labels.shape.begin(),
                                                     gt_labels.shape.end(), 1,
                                                     std::multiplies<int>()));
      PaddleTensor gt_bboxes = inputs[i][1];
      int l = 0;
      float *gt_bbs = static_cast<float *>(gt_bboxes.data.data());
      for (int k = 0; i < static_cast<int>(num); k++) {
        float x1 = std::min(std::max(gt_bbs[k++], 0.0f), 0.993f) * img2.cols;
        float y1 = std::min(std::max(gt_bbs[k++], 0.0f), 0.993f) * img2.rows;
        float x2 = std::min(std::max(gt_bbs[k++], 0.0f), 0.993f) * img2.cols;
        float y2 = std::min(std::max(gt_bbs[k++], 0.0f), 0.993f) * img2.rows;
        LOG(INFO) << "rendering ground truth bbox (" << x1 << ", " << y1
                  << "), (" << x2 << ", " << y2 << ")";

        int label_index =
            static_cast<int>(static_cast<float *>(gt_labels.data.data())[l]);
        cv::rectangle(img2, cv::Point(x1, y1), cv::Point(x2, y2), gt_color, 2);
        l += 1;
      }
    }

    // draw msg
    std::vector<std::string> msg;
    std::string mAP = "Current mAP: " +
                      std::to_string(*static_cast<float *>(top1.data.data()));
    msg.push_back(mAP);

    std::string avg_mAP =
        "Average mAP: " +
        std::to_string(*static_cast<float *>(top5.data.data()));
    msg.push_back(avg_mAP);

    std::string num_cores =
        "    Threads: " + std::to_string(FLAGS_paddle_num_threads);
    msg.push_back(num_cores);

    std::string vnni = "INT8 & VNNI: " + std::to_string(FLAGS_is_int8);
    msg.push_back(vnni);

    std::string fps = "        FPS: " + std::to_string(1000.f / elapsed_time);
    msg.push_back(fps);

    std::string latency = "    Latency: " + std::to_string(elapsed_time);
    msg.push_back(latency);

    std::string render_confidence =
        " Confidence: " + std::to_string(FLAGS_render_confidence);
    msg.push_back(render_confidence);

    for (int i = 0; i < msg.size(); i++) {
      LOG(INFO) << msg[i];
    }

    for (int i = 0; i < msg.size(); i++) {
      int baseline = 0;
      cv::Size text_size = cv::getTextSize(msg[i], cv::FONT_HERSHEY_SIMPLEX,
                                           0.5, 0.5, &baseline);

      cv::putText(img2, msg[i],
                  cv::Point(1.0, TEXT_BACKGROUND_H * i + TEXT_BACKGROUND_H / 2 +
                                     text_size.height / 2),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255, 155),
                  0.5);
    }

    cv::imshow(FLAGS_title, img2);
    cv::waitKey(0);
  }
}

void TestOneThreadPrediction(
    const PaddlePredictor::Config *config,
    const std::vector<std::vector<PaddleTensor>> &inputs,
    std::vector<std::vector<PaddleTensor>> *outputs, bool use_analysis = true,
    const VarType::Type data_type = VarType::FP32) {
  auto predictor = CreateTestPredictor(config, use_analysis);
  if (FLAGS_warmup) {
    PredictionWarmUp(predictor.get(), inputs, outputs, 1, 0, data_type);
  }
  PredictionRun(predictor.get(), inputs, outputs, 1, 0, data_type);
}

void CompareTopAccuracy(
    const std::vector<std::vector<PaddleTensor>> &output_slots_quant,
    const std::vector<std::vector<PaddleTensor>> &output_slots_ref) {
  if (output_slots_quant.size() == 0 || output_slots_ref.size() == 0)
    throw std::invalid_argument(
        "CompareTopAccuracy: output_slots vector is empty.");

  float total_accs1_quant{0};
  float total_accs1_ref{0};
  for (size_t i = 0; i < output_slots_quant.size(); ++i) {
    PADDLE_ENFORCE(output_slots_quant[i].size() >= 2UL);
    PADDLE_ENFORCE(output_slots_ref[i].size() >= 2UL);
    // second output: acc_top1
    if (output_slots_quant[i][1].lod.size() > 0 ||
        output_slots_ref[i][1].lod.size() > 0)
      throw std::invalid_argument(
          "CompareTopAccuracy: top1 accuracy output has nonempty LoD.");
    if (output_slots_quant[i][1].dtype != paddle::PaddleDType::FLOAT32 ||
        output_slots_ref[i][1].dtype != paddle::PaddleDType::FLOAT32)
      throw std::invalid_argument(
          "CompareTopAccuracy: top1 accuracy output is of a wrong type.");
    total_accs1_quant +=
        *static_cast<float *>(output_slots_quant[i][1].data.data());
    total_accs1_ref +=
        *static_cast<float *>(output_slots_ref[i][1].data.data());
  }
  float avg_acc1_quant = total_accs1_quant / output_slots_quant.size();
  float avg_acc1_ref = total_accs1_ref / output_slots_ref.size();

  LOG(INFO) << "Avg top1 INT8 accuracy: " << std::fixed << std::setw(6)
            << std::setprecision(4) << avg_acc1_quant;
  LOG(INFO) << "Avg top1 FP32 accuracy: " << std::fixed << std::setw(6)
            << std::setprecision(4) << avg_acc1_ref;
  LOG(INFO) << "Accepted accuracy drop threshold: " << FLAGS_quantized_accuracy;
  CHECK_LE(std::abs(avg_acc1_quant - avg_acc1_ref), FLAGS_quantized_accuracy);
}

void CompareQuantizedAndAnalysis(
    const AnalysisConfig *config, const AnalysisConfig *qconfig,
    const std::vector<std::vector<PaddleTensor>> &inputs) {
  PADDLE_ENFORCE_EQ(inputs[0][0].shape[0], FLAGS_batch_size,
                    "Input data has to be packed batch by batch.");
  LOG(INFO) << "FP32 & INT8 prediction run: batch_size " << FLAGS_batch_size
            << ", warmup batch size " << FLAGS_warmup_batch_size << ".";

  if (FLAGS_is_int8) {
    LOG(INFO) << "--- INT8 prediction start ---";
    auto *qcfg = reinterpret_cast<const PaddlePredictor::Config *>(qconfig);
    PrintConfig(qcfg, true);
    std::vector<std::vector<PaddleTensor>> quantized_outputs;
    TestOneThreadPrediction(qcfg, inputs, &quantized_outputs, true,
                            VarType::INT8);
  } else {
    LOG(INFO) << "--- FP32 prediction start ---";
    auto *cfg = reinterpret_cast<const PaddlePredictor::Config *>(config);
    PrintConfig(cfg, true);
    std::vector<std::vector<PaddleTensor>> analysis_outputs;
    TestOneThreadPrediction(cfg, inputs, &analysis_outputs, true,
                            VarType::FP32);
  }
}

}  // namespace inference
}  // namespace paddle
