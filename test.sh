#!/bin/bash

export TEST_TMPDIR=/dataset/sfu2/cache/sycl-debug

cp /dataset/sfu2/nervana/git/mkldnn-sycl/mkl-dnn/build/mkldnn_config.h /dataset/sfu2/cache/sycl-debug/_bazel_sfu2/84450c0e49ceae2aed4bdc930f0c43cb/external/mkl_dnn/include

cp /dataset/sfu2/nervana/git/mkldnn-sycl/mkl-dnn/build/src/ocl/*_cl.cpp /dataset/sfu2/cache/sycl-debug/_bazel_sfu2/84450c0e49ceae2aed4bdc930f0c43cb/external/mkl_dnn/src/ocl

bazel build --verbose_failures -c dbg --strip=never --copt='-g' --config=sycl --config=mkl --cxxopt=-no-serial-memop //tensorflow/tools/pip_package:build_pip_package


bazel-bin/tensorflow/tools/pip_package/build_pip_package /dataset/sfu2/nervana/pkg-sycl-dbg

pip2 install --upgrade --target=/dataset/sfu2/nervana/install-sycl-dbh /dataset/sfu2/nervana/pkg-sycl-dbg/

rm /dataset/sfu2/nervana/install-sycl/tensorflow/libtensorflow_framework.so

ln -s /dataset/sfu2/cache/sycl/_bazel_sfu2/972ddc02f896eef702e3ea03a320b700/execroot/org_tensorflow/bazel-out/k8-dbg/bin/tensorflow/libtensorflow_framework.so /dataset/sfu2/nervana/install-sycl/tensorflow/libtensorflow_framework.so

rm /dataset/sfu2/nervana/install-sycl/tensorflow/python/_pywrap_tensorflow_internal.so

ln -s /dataset/sfu2/cache/sycl/_bazel_sfu2/972ddc02f896eef702e3ea03a320b700/execroot/org_tensorflow/bazel-out/k8-dbg/bin/tensorflow/python/_pywrap_tensorflow_internal.so /dataset/sfu2/nervana/install-sycl/tensorflow/python/_pywrap_tensorflow_internal.so 


cp /dataset/sfu2/nervana/git/mkldnn-sycl/mkl-dnn/build/mkldnn_config.h /dataset/sfu2/cache/sycl/_bazel_sfu2/972ddc02f896eef702e3ea03a320b700/external/mkl_dnn/include
