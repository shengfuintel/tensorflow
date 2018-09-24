/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/quantize_and_dequantize_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

// Simulate quantization precision loss in a float tensor by:
// 1. Quantize the tensor to fixed point numbers, which should match the target
//    quantization method when it is used in inference.
// 2. Dequantize it back to floating point numbers for the following ops, most
//    likely matmul.
template <typename Device, typename T>
class QuantizeAndDequantizeV2Op : public OpKernel {
 public:
  explicit QuantizeAndDequantizeV2Op(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("signed_input", &signed_input_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bits", &num_bits_));
    OP_REQUIRES(ctx, num_bits_ > 0 && num_bits_ < (signed_input_ ? 62 : 63),
                errors::InvalidArgument("num_bits is out of range: ", num_bits_,
                                        " with signed_input_ ", signed_input_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("range_given", &range_given_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    auto device = ctx->eigen_device<Device>();
    Tensor input_min_tensor;
    Tensor input_max_tensor;
    if (range_given_) {
      input_min_tensor = ctx->input(1);
      input_max_tensor = ctx->input(2);
      auto input_min = input_min_tensor.scalar<T>();
      auto input_max = input_max_tensor.scalar<T>();
      T min_val;
      T max_val;
#ifdef TENSORFLOW_USE_SYCL
      device.memcpyDeviceToHost(&min_val, input_min.data(), sizeof(T));
      device.memcpyDeviceToHost(&max_val, input_max.data(), sizeof(T));
#else
      min_val = input_min();
      max_val = input_max();
#endif // TENSORFLOW_USE_SYCL
      OP_REQUIRES(ctx, min_val <= max_val,
                  errors::InvalidArgument("Invalid range: input_min ", min_val,
                                          " > input_max ", max_val));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             TensorShape(), &input_min_tensor));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             TensorShape(), &input_max_tensor));
    }

    functor::QuantizeAndDequantizeOneScaleFunctor<Device, T> f;
    f(device, input.flat<T>(), signed_input_, num_bits_,
      range_given_, &input_min_tensor, &input_max_tensor, output->flat<T>());
  }

 private:
  bool signed_input_;
  int num_bits_;
  bool range_given_;
};

// Simulate quantization precision loss in a float tensor by:
// 1. Quantize the tensor to fixed point numbers, which should match the target
//    quantization method when it is used in inference.
// 2. Dequantize it back to floating point numbers for the following ops, most
//    likely matmul.
// Almost identical to QuantizeAndDequantizeV2Op, except that num_bits is a
// tensor.
template <typename Device, typename T>
class QuantizeAndDequantizeV3Op : public OpKernel {
 public:
  explicit QuantizeAndDequantizeV3Op(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("signed_input", &signed_input_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("range_given", &range_given_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    Tensor num_bits_tensor;
    num_bits_tensor = ctx->input(3);
    int num_bits_val = num_bits_tensor.scalar<int32>()();

    OP_REQUIRES(
        ctx, num_bits_val > 0 && num_bits_val < (signed_input_ ? 62 : 63),
        errors::InvalidArgument("num_bits is out of range: ", num_bits_val,
                                " with signed_input_ ", signed_input_));

    auto device = ctx->eigen_device<Device>();
    Tensor input_min_tensor;
    Tensor input_max_tensor;
    if (range_given_) {
      input_min_tensor = ctx->input(1);
      input_max_tensor = ctx->input(2);
      auto input_min = input_min_tensor.scalar<T>();
      auto input_max = input_max_tensor.scalar<T>();
      T min_val;
      T max_val;
#ifdef TENSORFLOW_USE_SYCL
      device.memcpyDeviceToHost(&min_val, input_min.data(), sizeof(T));
      device.memcpyDeviceToHost(&max_val, input_max.data(), sizeof(T));
#else
      min_val = input_min();
      max_val = input_max();
#endif // TENSORFLOW_USE_SYCL
      OP_REQUIRES(ctx, min_val <= max_val,
                  errors::InvalidArgument("Invalid range: input_min ", min_val,
                                          " > input_max ", max_val));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             TensorShape(), &input_min_tensor));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             TensorShape(), &input_max_tensor));
    }

    functor::QuantizeAndDequantizeOneScaleFunctor<Device, T> f;
    f(device, input.flat<T>(), signed_input_, num_bits_val,
      range_given_, &input_min_tensor, &input_max_tensor, output->flat<T>());
  }

 private:
  bool signed_input_;
  bool range_given_;
};

// DEPRECATED: Use QuantizeAndDequantizeV2Op.
template <typename Device, typename T>
class QuantizeAndDequantizeOp : public OpKernel {
 public:
  explicit QuantizeAndDequantizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("signed_input", &signed_input_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bits", &num_bits_));
    OP_REQUIRES(ctx, num_bits_ > 0 && num_bits_ < (signed_input_ ? 62 : 63),
                errors::InvalidArgument("num_bits is out of range: ", num_bits_,
                                        " with signed_input_ ", signed_input_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("range_given", &range_given_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("input_min", &input_min_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("input_max", &input_max_));
    if (range_given_) {
      OP_REQUIRES(
          ctx, input_min_ <= input_max_,
          errors::InvalidArgument("Invalid range: input_min ", input_min_,
                                  " > input_max ", input_max_));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    // One global scale.
    Tensor input_min_tensor(DataTypeToEnum<T>::value, TensorShape());
    Tensor input_max_tensor(DataTypeToEnum<T>::value, TensorShape());
    // Initialize the tensors with the values in the Attrs.
    auto device = ctx->eigen_device<Device>();
    auto input_min = input_min_tensor.scalar<T>();
    auto input_max = input_max_tensor.scalar<T>();
#ifdef TENSORFLOW_USE_SYCL
    input_min.device(device) = input_min.constant(static_cast<T>(input_min_));
    input_max.device(device) = input_max.constant(static_cast<T>(input_max_));
#else
    input_min() = static_cast<T>(input_min_);
    input_max() = static_cast<T>(input_max_);
#endif // TENSORFLOW_USE_SYCL

    functor::QuantizeAndDequantizeOneScaleFunctor<Device, T> functor;
    functor(device, input.flat<T>(), signed_input_,
            num_bits_, range_given_, &input_min_tensor, &input_max_tensor,
            output->flat<T>());
  }

 private:
  bool signed_input_;
  int num_bits_;
  bool range_given_;
  float input_min_;
  float input_max_;
};

namespace functor {
// Specialization for CPUDevice.
template <typename T>
struct QuantizeAndDequantizeOneScaleFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstVec input,
                  const bool signed_input, const int num_bits,
                  const bool range_given, Tensor* input_min_tensor,
                  Tensor* input_max_tensor, typename TTypes<T>::Vec out) {
    QuantizeAndDequantizeOneScaleImpl<CPUDevice, T>::Compute(
        d, input, signed_input, num_bits, range_given, input_min_tensor,
        input_max_tensor, out);
  }
};

#ifdef TENSORFLOW_USE_SYCL
// Specialization for SYCLDevice.
template <typename T>
struct QuantizeAndDequantizeOneScaleFunctor<SYCLDevice, T> {
  void operator()(const SYCLDevice& d, typename TTypes<T>::ConstVec input,
                  const bool signed_input, const int num_bits,
                  const bool range_given, Tensor* input_min_tensor,
                  Tensor* input_max_tensor, typename TTypes<T>::Vec out) {
    QuantizeAndDequantizeOneScaleImpl<SYCLDevice, T>::Compute(
        d, input, signed_input, num_bits, range_given, input_min_tensor,
        input_max_tensor, out);
  }
};
#endif  // TENSORFLOW_USE_SYCL
}  // namespace functor

#define REGISTER_CPU_KERNEL(T)                                                 \
  REGISTER_KERNEL_BUILDER(Name("QuantizeAndDequantizeV2")                      \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T"),                         \
                          QuantizeAndDequantizeV2Op<CPUDevice, T>);            \
  REGISTER_KERNEL_BUILDER(Name("QuantizeAndDequantizeV3")                      \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<T>("T"),                         \
                          QuantizeAndDequantizeV3Op<CPUDevice, T>);            \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("QuantizeAndDequantize").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      QuantizeAndDequantizeOp<CPUDevice, T>);
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(T)                                                 \
  REGISTER_KERNEL_BUILDER(Name("QuantizeAndDequantizeV2")                      \
                              .Device(DEVICE_GPU)                              \
                              .HostMemory("input_max")                         \
                              .HostMemory("input_min")                         \
                              .TypeConstraint<T>("T"),                         \
                          QuantizeAndDequantizeV2Op<GPUDevice, T>);            \
  REGISTER_KERNEL_BUILDER(Name("QuantizeAndDequantizeV3")                      \
                              .Device(DEVICE_GPU)                              \
                              .HostMemory("input_max")                         \
                              .HostMemory("input_min")                         \
                              .HostMemory("num_bits")                          \
                              .TypeConstraint<T>("T"),                         \
                          QuantizeAndDequantizeV3Op<GPUDevice, T>);            \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("QuantizeAndDequantize").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      QuantizeAndDequantizeOp<GPUDevice, T>);
TF_CALL_float(REGISTER_GPU_KERNEL);
TF_CALL_double(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
#endif

#ifdef TENSORFLOW_USE_SYCL

#define REGISTER_SYCL_KERNEL(T)                                                 \
  REGISTER_KERNEL_BUILDER(Name("QuantizeAndDequantizeV2")                       \
                              .Device(DEVICE_SYCL)                              \
                              .TypeConstraint<T>("T"),                          \
                          QuantizeAndDequantizeV2Op<SYCLDevice, T>);            \
  REGISTER_KERNEL_BUILDER(Name("QuantizeAndDequantizeV3")                       \
                              .Device(DEVICE_SYCL)                              \
                              .HostMemory("num_bits")                           \
                              .TypeConstraint<T>("T"),                          \
                          QuantizeAndDequantizeV3Op<SYCLDevice, T>);            \
  REGISTER_KERNEL_BUILDER(                                                      \
      Name("QuantizeAndDequantize").Device(DEVICE_SYCL).TypeConstraint<T>("T"), \
      QuantizeAndDequantizeOp<SYCLDevice, T>);
TF_CALL_float(REGISTER_SYCL_KERNEL);
TF_CALL_SYCL_double(REGISTER_SYCL_KERNEL);
#undef REGISTER_SYCL_KERNEL

#endif  // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
