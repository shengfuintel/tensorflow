#!/bin/bash

export TEST_TMPDIR=/dataset/sfu2/cache/sycl

 cp /dataset/sfu2/nervana/git/mkldnn-sycl/mkl-dnn/build/mkldnn_config.h /dataset/sfu2/cache/sycl/_bazel_sfu2/972ddc02f896eef702e3ea03a320b700/external/mkl_dnn/include

cp /dataset/sfu2/nervana/git/mkldnn-sycl/mkl-dnn/build/src/ocl/*_cl.cpp /dataset/sfu2/cache/sycl/_bazel_sfu2/972ddc02f896eef702e3ea03a320b700/external/mkl_dnn/src/ocl

bazel build --verbose_failures -c opt --config=sycl --config=mkl --cxxopt=-no-serial-memop //tensorflow/tools/pip_package:build_pip_package

bazel build --verbose_failures -c opt --config=sycl --config=mkl --copt='-DTENSORFLOW_SYCL_NO_HALF=1' --copt='-DTENSORFLOW_SYCL_NO_DOUBLE=1' --host_copt='-DTENSORFLOW_SYCL_NO_HALF=1'  --host_copt='-DTENSORFLOW_SYCL_NO_DOUBLE=1'  --cxxopt=-no-serial-memop //tensorflow/tools/pip_package:build_pip_package 


bazel-bin/tensorflow/tools/pip_package/build_pip_package /dataset/sfu2/nervana/pkg-sycl

pip2 install --upgrade --target=/dataset/sfu2/nervana/install-sycl /dataset/sfu2/nervana/pkg-sycl/

rm /dataset/sfu2/nervana/install-sycl/tensorflow/libtensorflow_framework.so

ln -s /dataset/sfu2/cache/sycl/_bazel_sfu2/972ddc02f896eef702e3ea03a320b700/execroot/org_tensorflow/bazel-out/k8-dbg/bin/tensorflow/libtensorflow_framework.so /dataset/sfu2/nervana/install-sycl/tensorflow/libtensorflow_framework.so

rm /dataset/sfu2/nervana/install-sycl/tensorflow/python/_pywrap_tensorflow_internal.so

ln -s /dataset/sfu2/cache/sycl/_bazel_sfu2/972ddc02f896eef702e3ea03a320b700/execroot/org_tensorflow/bazel-out/k8-dbg/bin/tensorflow/python/_pywrap_tensorflow_internal.so /dataset/sfu2/nervana/install-sycl/tensorflow/python/_pywrap_tensorflow_internal.so 


cp /dataset/sfu2/nervana/git/mkldnn-sycl/mkl-dnn/build/mkldnn_config.h /dataset/sfu2/cache/sycl/_bazel_sfu2/972ddc02f896eef702e3ea03a320b700/external/mkl_dnn/include
