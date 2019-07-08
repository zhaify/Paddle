/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
using Tensor = framework::Tensor;

template <typename T>
static void NearestNeighborInterpolate(const Tensor& input, Tensor* output,
                                       const float ratio_h, const float ratio_w,
                                       const int n, const int c,
                                       const int out_h, const int out_w,
                                       const bool align_corners) {
  auto input_t = EigenTensor<T, 4>::From(input);
  auto output_t = EigenTensor<T, 4>::From(*output);
  for (int k = 0; k < out_h; k++) {  // loop for images
    int in_k = (align_corners) ? static_cast<int>(ratio_h * k + 0.5)
                               : static_cast<int>(ratio_h * k);

    for (int l = 0; l < out_w; l++) {
      int in_l = (align_corners) ? static_cast<int>(ratio_w * l + 0.5)
                                 : static_cast<int>(ratio_w * l);

      for (int i = 0; i < n; i++) {    // loop for batches
        for (int j = 0; j < c; j++) {  // loop for channels
          output_t(i, j, k, l) = input_t(i, j, in_k, in_l);
        }
      }
    }
  }
}

class Timer {
 public:
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point startu;

  void tic() { start = std::chrono::high_resolution_clock::now(); }
  double toc() {
    startu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(startu -
                                                                  start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};

template <typename T>
static void BilinearInterpolation(const Tensor& input, Tensor* output,
                                  const float ratio_h, const float ratio_w,
                                  const int in_h, const int in_w, const int n,
                                  const int c, const int out_h, const int out_w,
                                  const bool align_corners,
                                  const bool align_mode) {
  auto input_t = EigenTensor<T, 4>::From(input);
  auto output_t = EigenTensor<T, 4>::From(*output);
  bool align_flag = (align_mode == 0 && !align_corners);

  // double elapsed_time = 0.0;
  // run_timer.tic();

  // for (int k = 0; k < out_h; k++) {  // loop for images 16
  //   int y_n = align_flag ? static_cast<int>(ratio_h * (k + 0.5) - 0.5)
  //                         : static_cast<int>(ratio_h * k);
  //   y_n = (y_n > 0) ? y_n : 0;
  //   int y_s = (y_n + 1) < (in_h - 1) ? (y_n + 1) : (in_h - 1);
  //   float d_n =
  //       align_flag ? ratio_h * (k + 0.5) - 0.5 - y_n : ratio_h * k - y_n;
  //   float d_s = 1.f - d_n;

  //   for (int l = 0; l < out_w; l++) {  // 16
  //     int x_w = (align_mode == 0 && !align_corners)
  //                   ? static_cast<int>(ratio_w * (l + 0.5) - 0.5)
  //                   : static_cast<int>(ratio_w * l);
  //     x_w = (x_w > 0) ? x_w : 0;
  //     int x_e = (x_w + 1) < (in_w - 1) ? (x_w + 1) : (in_w - 1);
  //     float d_w =
  //         align_flag ? ratio_w * (l + 0.5) - 0.5 - x_w : ratio_w * l - x_w;
  //     float d_e = 1.f - d_w;
  //     #pragma omp parallel for collapse(2)
  //     for (int i = 0; i < n; i++) {    // loop for batches 1
  //       for (int j = 0; j < c; j++) {  // loop for channels 128
  //         // bilinear interpolation

  //         output_t(i, j, k, l) = input_t(i, j, y_n, x_w) * d_s * d_e +
  //                                 input_t(i, j, y_n, x_e) * d_s * d_w +
  //                                 input_t(i, j, y_s, x_w) * d_n * d_e +
  //                                 input_t(i, j, y_s, x_e) * d_n * d_w;
  //       }
  //     }
  //   }
  // }

  std::vector<int> src_index;
  std::vector<float> src_index_w;

  for (int k = 0; k < out_h; k++) {
    int y_n = align_flag ? static_cast<int>(ratio_h * (k + 0.5) - 0.5)
                         : static_cast<int>(ratio_h * k);
    if (y_n < 0) {
      y_n = 0;
    }

    int y_s = (y_n + 1) < (in_h - 1) ? (y_n + 1) : (in_h - 1);
    float d_n =
        align_flag ? ratio_h * (k + 0.5) - 0.5 - y_n : ratio_h * k - y_n;
    float d_s = 1.f - d_n;
    for (int l = 0; l < out_w; l++) {  // 16
      src_index.emplace_back(y_n);

      src_index.emplace_back(y_s);

      int x_w = (align_mode == 0 && !align_corners)
                    ? static_cast<int>(ratio_w * (l + 0.5) - 0.5)
                    : static_cast<int>(ratio_w * l);

      if (x_w < 0) {
        x_w = 0;
      }
      src_index.emplace_back(x_w);

      int x_e = (x_w + 1) < (in_w - 1) ? (x_w + 1) : (in_w - 1);
      src_index.emplace_back(x_e);

      float d_w =
          align_flag ? ratio_w * (l + 0.5) - 0.5 - x_w : ratio_w * l - x_w;
      src_index_w.emplace_back(d_w);

      float d_e = 1.f - d_w;
      src_index_w.emplace_back(d_e);

      src_index_w.emplace_back(d_s);
      src_index_w.emplace_back(d_n);
    }
  }

#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; i++) {            // loop for batches 1
    for (int j = 0; j < c; j++) {          // loop for channels 128
      for (int k = 0; k < out_h; k++) {    // loop for images 16
        for (int l = 0; l < out_w; l++) {  // 16
          // output_t(i, j, k, l) =  (input_t(i, j, y_n, x_e) * d_w + input_t(i,
          // j, y_n, x_w) * d_e) * d_s +
          //                         (input_t(i, j, y_s, x_e) * d_w + input_t(i,
          //                         j, y_s, x_w) * d_e) * d_n;
          int offset = k * out_h + l;

          output_t(i, j, k, l) =
              (input_t(i, j, src_index[offset], src_index[offset + 3]) *
                   src_index_w[offset] +
               input_t(i, j, src_index[offset], src_index[offset + 2]) *
                   src_index_w[offset + 1]) *
                  src_index_w[offset + 2] +

              (input_t(i, j, src_index[offset + 1], src_index[offset + 3]) *
                   src_index_w[offset] +
               input_t(i, j, src_index[offset + 1], src_index[offset + 2]) *
                   src_index_w[offset + 1]) *
                  src_index_w[offset + 3];
        }
      }
    }
  }
}

template <typename T>
static void NearestNeighborInterpolateGrad(
    const Tensor& output_grad, Tensor* input_grad, const float ratio_h,
    const float ratio_w, const int n, const int c, const int out_h,
    const int out_w, const bool align_corners) {
  auto input_grad_t = EigenTensor<T, 4>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 4>::From(output_grad);

  for (int k = 0; k < out_h; k++) {  // loop for images
    int in_k = (align_corners) ? static_cast<int>(ratio_h * k + 0.5)
                               : static_cast<int>(ratio_h * k);

    for (int l = 0; l < out_w; l++) {
      int in_l = (align_corners) ? static_cast<int>(ratio_w * l + 0.5)
                                 : static_cast<int>(ratio_w * l);

      for (int i = 0; i < n; i++) {    // loop for batches
        for (int j = 0; j < c; j++) {  // loop for channels
          input_grad_t(i, j, in_k, in_l) += output_grad_t(i, j, k, l);
        }
      }
    }
  }
}

template <typename T>
static void BilinearInterpolationGrad(const Tensor& output_grad,
                                      Tensor* input_grad, const float ratio_h,
                                      const float ratio_w, const int in_h,
                                      const int in_w, const int n, const int c,
                                      const int out_h, const int out_w,
                                      const bool align_corners,
                                      const int align_mode) {
  auto input_grad_t = EigenTensor<T, 4>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 4>::From(output_grad);
  bool align_flag = (align_mode == 0 && !align_corners);
  for (int k = 0; k < out_h; k++) {  // loop for images
    int y_n = align_flag ? static_cast<int>(ratio_h * (k + 0.5) - 0.5)
                         : static_cast<int>(ratio_h * k);
    y_n = (y_n > 0) ? y_n : 0;
    int y_s = (y_n + 1) < (in_h - 1) ? (y_n + 1) : (in_h - 1);
    float d_n =
        align_flag ? ratio_h * (k + 0.5) - 0.5 - y_n : ratio_h * k - y_n;
    float d_s = 1.f - d_n;

    for (int l = 0; l < out_w; l++) {
      int x_w = align_flag ? static_cast<int>(ratio_w * (l + 0.5) - 0.5)
                           : static_cast<int>(ratio_w * l);
      x_w = (x_w > 0) ? x_w : 0;
      int x_e = (x_w + 1) < (in_w - 1) ? (x_w + 1) : (in_w - 1);
      float d_w =
          align_flag ? ratio_w * (l + 0.5) - 0.5 - x_w : ratio_w * l - x_w;
      float d_e = 1.f - d_w;

      for (int i = 0; i < n; i++) {    // loop for batches
        for (int j = 0; j < c; j++) {  // loop for channels
          // bilinear interpolation grad
          const T grad = output_grad_t(i, j, k, l);
          input_grad_t(i, j, y_n, x_w) += static_cast<T>(grad * d_s * d_e);
          input_grad_t(i, j, y_s, x_w) += static_cast<T>(grad * d_n * d_e);
          input_grad_t(i, j, y_n, x_e) += static_cast<T>(grad * d_s * d_w);
          input_grad_t(i, j, y_s, x_e) += static_cast<T>(grad * d_n * d_w);
        }
      }
    }
  }
}
template <typename T>
class InterpolateKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");

    const int n = input->dims()[0];
    const int c = input->dims()[1];
    const int in_h = input->dims()[2];
    const int in_w = input->dims()[3];

    std::string interp_method = ctx.Attr<std::string>("interp_method");
    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");

    float scale = ctx.Attr<float>("scale");
    if (scale > 0) {
      out_h = static_cast<int>(in_h * scale);
      out_w = static_cast<int>(in_w * scale);
    }

    auto out_size = ctx.Input<Tensor>("OutSize");
    if (out_size != nullptr) {
      auto out_size_data = out_size->data<int>();
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    }
    bool align_corners = ctx.Attr<bool>("align_corners");
    int align_mode = ctx.Attr<int>("align_mode");

    output->mutable_data<T>({n, c, out_h, out_w}, ctx.GetPlace());
    auto& device_ctx =
        ctx.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, T> zero;
    zero(device_ctx, output, static_cast<T>(0.0));

    if (in_h == out_h && in_w == out_w) {
      framework::TensorCopy(*input, ctx.GetPlace(), output);
      return;
    }

    float ratio_h = 0.f;
    float ratio_w = 0.f;

    if (out_h > 1) {
      ratio_h = (align_corners) ? static_cast<float>(in_h - 1) / (out_h - 1)
                                : static_cast<float>(in_h) / out_h;
    }
    if (out_w > 1) {
      ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                                : static_cast<float>(in_w) / out_w;
    }

    if ("bilinear" == interp_method) {
      BilinearInterpolation<T>(*input, output, ratio_h, ratio_w, in_h, in_w, n,
                               c, out_h, out_w, align_corners, align_mode);
    } else if ("nearest" == interp_method) {
      NearestNeighborInterpolate<T>(*input, output, ratio_h, ratio_w, n, c,
                                    out_h, out_w, align_corners);
    }
  }
};

template <typename T>
class InterpolateGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));

    const int n = input->dims()[0];
    const int c = input->dims()[1];
    const int in_h = input->dims()[2];
    const int in_w = input->dims()[3];

    std::string interp_method = ctx.Attr<std::string>("interp_method");
    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");

    float scale = ctx.Attr<float>("scale");
    if (scale > 0) {
      out_h = static_cast<int>(in_h * scale);
      out_w = static_cast<int>(in_w * scale);
    }

    auto out_size = ctx.Input<Tensor>("OutSize");
    if (out_size != nullptr) {
      auto out_size_data = out_size->data<int>();
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    }

    bool align_corners = ctx.Attr<bool>("align_corners");
    int align_mode = ctx.Attr<int>("align_mode");

    input_grad->mutable_data<T>({n, c, in_h, in_w}, ctx.GetPlace());
    auto& device_ctx =
        ctx.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, T> zero;
    zero(device_ctx, input_grad, static_cast<T>(0.0));

    if (in_h == out_h && in_w == out_w) {
      framework::TensorCopy(*output_grad, ctx.GetPlace(), input_grad);
      return;
    }

    float ratio_h = 0.f;
    float ratio_w = 0.f;

    if (out_h > 1) {
      ratio_h = (align_corners) ? static_cast<float>(in_h - 1) / (out_h - 1)
                                : static_cast<float>(in_h) / out_h;
    }
    if (out_w > 1) {
      ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                                : static_cast<float>(in_w) / out_w;
    }

    if ("bilinear" == interp_method) {
      BilinearInterpolationGrad<T>(*output_grad, input_grad, ratio_h, ratio_w,
                                   in_h, in_w, n, c, out_h, out_w,
                                   align_corners, align_mode);
    } else if ("nearest" == interp_method) {
      NearestNeighborInterpolateGrad<T>(*output_grad, input_grad, ratio_h,
                                        ratio_w, n, c, out_h, out_w,
                                        align_corners);
    }
  }
};

}  // namespace operators
}  // namespace paddle
