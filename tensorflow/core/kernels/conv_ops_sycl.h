#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_SYCL_H_
#define TENSORFLOW_KERNELS_CONV_OPS_SYCL_H_

#ifdef ARM_NON_MOBILE
#define SNN_ARM 1
#define SNN_SELECTOR arm_selector
#else
#define SNN_SELECTOR default_selector
#endif

#include "tensorflow/core/kernels/conv_2d.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/kernels/conv_grad_ops.h"

#include "tensorflow/core/kernels/conv_ops_sycl_launcher.h"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;
// Forward declarations needed for later specializations.
template <typename Device, typename T>
struct LaunchConv2DOp;

template <typename T>
struct LaunchConv2DOp<SYCLDevice, T> {
  void operator()(OpKernelContext* context, bool /*use_cudnn*/,
                  bool /*cudnn_use_autotune*/, const Tensor& input,
                  const Tensor& filter, int row_dilation, int col_dilation,
                  int64 stride_rows, int64 stride_cols, const Padding& padding,
                  Tensor* output, TensorFormat data_format) {
    if (filter.dim_size(0) == 1 && filter.dim_size(1) == 1 && stride_rows == 1 &&
        stride_cols == 1) {
      // For 1x1 kernel, the 2D convolution is reduced to matrix
      // multiplication.
      //
      // TODO(vrv): We should be able to call SpatialConvolution
      // and it will produce the same result, but doing so
      // led to NaNs during training.  Using matmul instead for now.
      int conv_width = 1;  // Width for the convolution step.
      for (int i = 0; i < 3; ++i) {
        conv_width *= output->dim_size(i);
      }

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<SYCLDevice, T>()(
          context->eigen_device<SYCLDevice>(),
          output->shaped<T, 2>({conv_width, filter.dim_size(3)}),
          input.shaped<T, 2>({conv_width, filter.dim_size(2)}),
          filter.shaped<T, 2>({filter.dim_size(2), filter.dim_size(3)}),
          dim_pair);
    } else if (filter.dim_size(0) == input.dim_size(1) &&
               filter.dim_size(1) == input.dim_size(2) && row_dilation == 1 &&
               col_dilation == 1 && padding == VALID) {
      // If the input data and filter have the same height/width,
      // the 2D convolution is reduced to matrix multiplication.
      const int k =  // Length of reduction dimension.
          filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2);

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<SYCLDevice, T>()(
          context->eigen_device<SYCLDevice>(),
          output->shaped<T, 2>({input.dim_size(0), filter.dim_size(3)}),
          input.shaped<T, 2>({input.dim_size(0), k}),
          filter.shaped<T, 2>({k, filter.dim_size(3)}), dim_pair);
    } else {
      const int64 batch = GetTensorDim(input, data_format, 'N');
      const int64 input_rows = GetTensorDim(input, data_format, 'H');
      const int64 input_cols = GetTensorDim(input, data_format, 'W');

      const int64 filter_rows = filter.dim_size(0);
      const int64 filter_cols = filter.dim_size(1);
      const int64 in_depth = filter.dim_size(2);
      const int64 out_depth = filter.dim_size(3);

      int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
      OP_REQUIRES_OK(context,
                     GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                           padding, &out_rows, &pad_rows));
      OP_REQUIRES_OK(context,
                     GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                           padding, &out_cols, &pad_cols));

      SYCLConv2DParams params{
          in_depth,    out_depth,   batch,       input_rows,  input_cols,
          filter_rows, filter_cols, stride_rows, stride_cols, out_rows,
          out_cols,    pad_rows,    pad_cols};

      T const* const in_ptr = input.template flat<T>().data();
      T const* const fil_ptr = filter.template flat<T>().data();
      T* const out_ptr = output->template flat<T>().data();

      SNN_SELECTOR sel;
      if (data_format == FORMAT_NCHW) {
        launch_conv2d_nchw<T, ConvType::Forward>(
            context->eigen_device<SYCLDevice>(), in_ptr, fil_ptr, params,
            out_ptr, sel);
      } else {
        launch_conv2d<T, ConvType::Forward>(context->eigen_device<SYCLDevice>(),
                                            in_ptr, fil_ptr, params, out_ptr,
                                            sel);
      }
    }
  }
};
template <typename T>
struct LaunchConv2DBackpropInputOp<SYCLDevice, T> {
  void operator()(OpKernelContext* context, bool /*use_cudnn*/,
                  bool /*cudnn_use_autotune*/, const Tensor& out_backprop,
                  const Tensor& filter, int32 row_dilation, int32 col_dilation,
                  int64 stride_rows, int64 stride_cols, const Padding& padding,
                  Tensor* in_backprop, TensorFormat data_format) {
    const int64 batch = GetTensorDim(*in_backprop, data_format, 'N');
    const int64 input_rows = GetTensorDim(*in_backprop, data_format, 'H');
    const int64 input_cols = GetTensorDim(*in_backprop, data_format, 'W');

    const int64 filter_rows = filter.dim_size(0);
    const int64 filter_cols = filter.dim_size(1);
    const int64 in_depth = filter.dim_size(2);
    const int64 out_depth = filter.dim_size(3);

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                         padding, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                         padding, &out_cols, &pad_cols));

    SYCLConv2DParams params{in_depth,    out_depth,   batch,       input_rows,
                            input_cols,  filter_rows, filter_cols, stride_rows,
                            stride_cols, out_rows,    out_cols,    pad_rows,
                            pad_cols};

    T const* const in_ptr = out_backprop.template flat<T>().data();
    T const* const fil_ptr = filter.template flat<T>().data();
    T* const out_ptr = in_backprop->template flat<T>().data();

    SNN_SELECTOR sel;
    if (data_format == FORMAT_NCHW) {
      launch_conv2d_nchw<T, ConvType::InputBackprop>(
          context->eigen_device<SYCLDevice>(), in_ptr, fil_ptr, params, out_ptr,
          sel);
    } else {
      launch_conv2d<T, ConvType::InputBackprop>(
          context->eigen_device<SYCLDevice>(), in_ptr, fil_ptr, params, out_ptr,
          sel);
    }
  }
};
template <typename T>
struct LaunchConv2DBackpropFilterOp<SYCLDevice, T> {
  void operator()(OpKernelContext* context, bool /*use_cudnn*/,
                  bool /*cudnn_use_autotune*/, const Tensor& out_backprop,
                  const Tensor& input, int32 row_dilation, int32 col_dilation,
                  int64 stride_rows, int64 stride_cols, const Padding& padding,
                  Tensor* filter_backprop, TensorFormat data_format) {
    const int64 batch = GetTensorDim(input, data_format, 'N');
    const int64 input_rows = GetTensorDim(input, data_format, 'H');
    const int64 input_cols = GetTensorDim(input, data_format, 'W');

    const int64 filter_rows = filter_backprop->dim_size(0);
    const int64 filter_cols = filter_backprop->dim_size(1);
    const int64 in_depth = filter_backprop->dim_size(2);
    const int64 out_depth = filter_backprop->dim_size(3);

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                         padding, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                         padding, &out_cols, &pad_cols));

    SYCLConv2DParams params{in_depth,    out_depth,   batch,       input_rows,
                            input_cols,  filter_rows, filter_cols, stride_rows,
                            stride_cols, out_rows,    out_cols,    pad_rows,
                            pad_cols};

    T const* const in_ptr = input.template flat<T>().data();
    T const* const fil_ptr = out_backprop.template flat<T>().data();
    T* const out_ptr = filter_backprop->template flat<T>().data();

    SNN_SELECTOR sel;
    if (data_format == FORMAT_NCHW) {
      launch_conv2d_nchw<T, ConvType::FilterBackprop>(
          context->eigen_device<SYCLDevice>(), in_ptr, fil_ptr, params, out_ptr,
          sel);
    } else {
      launch_conv2d<T, ConvType::FilterBackprop>(
          context->eigen_device<SYCLDevice>(), in_ptr, fil_ptr, params, out_ptr,
          sel);
    }
  }
};
}  // namespace tensorflow
#undef SNN_SELECTOR
#endif  // TENSORFLOW_KERNELS_CONV_OPS_SYCL_H_
