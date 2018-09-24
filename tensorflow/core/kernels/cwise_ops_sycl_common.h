/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_USE_SYCL
#error This file must only be included when building TensorFlow with SYCL support
#endif

#ifndef TENSORFLOW_CORE_KERNELS_CWISE_OPS_SYCL_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_CWISE_OPS_SYCL_COMMON_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace functor {

typedef Eigen::SyclDevice SYCLDevice;

// Partial specialization of UnaryFunctor<Device=SYCLDevice, Functor>.
template <typename Functor>
struct UnaryFunctor<SYCLDevice, Functor> {
  void operator()(const SYCLDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in) {
    To32Bit(out).device(d) = To32Bit(in).unaryExpr(typename Functor::func());
  }
};

// Partial specialization of BinaryFunctor<Device=SYCLDevice, Functor>.
// SYCL cannot use the bool pointer to check for errors
template <typename Functor, int NDIMS, bool has_errors>
struct BinaryFunctor<SYCLDevice, Functor, NDIMS, has_errors> {
  void operator()(const SYCLDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1, bool*) {
    To32Bit(out).device(d) =
        To32Bit(in0).binaryExpr(To32Bit(in1), typename Functor::func());
  }

  void Left(const SYCLDevice& d, typename Functor::tout_type out,
            typename Functor::tscalar_type scalar,
            typename Functor::tin_type in, bool*) {
    typedef typename Functor::func Binary;
    constexpr int NumDims = Functor::tin_type::NumDimensions;
    static_assert(NumDims == 1, "Unexpected size");
    Eigen::Sizes<1> scalar_dim;
    out.device(d) = scalar.reshape(scalar_dim)
                        .broadcast(in.dimensions())
                        .binaryExpr(in, Binary());
  }

  void Right(const SYCLDevice& d, typename Functor::tout_type out,
             typename Functor::tin_type in,
             typename Functor::tscalar_type scalar, bool*) {
    typedef typename Functor::func Binary;
    constexpr int NumDims = Functor::tin_type::NumDimensions;
    static_assert(NumDims == 1, "Unexpected size");
    Eigen::Sizes<1> scalar_dim;
    out.device(d) = in.binaryExpr(
        scalar.reshape(scalar_dim).broadcast(in.dimensions()), Binary());
  }

  void BCast(const SYCLDevice& d,
             typename TTypes<typename Functor::out_type, NDIMS>::Tensor out,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in0,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast0,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in1,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast1,
             bool*) {
    typedef typename Functor::in_type T;
    typename Functor::func func;
    if ((NDIMS == 2) && Functor::use_bcast_optimization &&
        use_bcast_optimization<T>::value) {
      const bool bcast0_all_one = AllOne<NDIMS>(bcast0);
      const bool bcast1_all_one = AllOne<NDIMS>(bcast1);
      if (bcast0_all_one && !bcast1_all_one) {
        To32Bit(out).device(d) =
            To32Bit(in0).binaryExpr(To32Bit(in1).broadcast(bcast1), func);
        return;
      }
      if (!bcast0_all_one && bcast1_all_one) {
        To32Bit(out).device(d) =
            To32Bit(in0).broadcast(bcast0).binaryExpr(To32Bit(in1), func);
        return;
      }
    }
    To32Bit(out).device(d) = To32Bit(in0).broadcast(bcast0).binaryExpr(
        To32Bit(in1).broadcast(bcast1), func);
  }
};

// Partial specialization of ApproximateEqual<Device=SYCLDevice, T>.
template <typename T>
struct ApproximateEqual<SYCLDevice, T> {
  void operator()(const SYCLDevice& d, typename TTypes<T>::ConstFlat x,
                  typename TTypes<T>::ConstFlat y, T tolerance,
                  typename TTypes<bool>::Flat z) {
    auto diff = x - y;
    z.device(d) = diff.abs() <= tolerance;
  }
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CWISE_OPS_SYCL_COMMON_H_
