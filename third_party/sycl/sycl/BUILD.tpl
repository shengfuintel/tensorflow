licenses(["notice"])  # Apache 2.0

load("@local_config_sycl//sycl:build_defs.bzl", "if_sycl")

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE.text"])

config_setting(
    name = "using_sycl_ccpp",
    define_values = {
        "using_sycl": "true",
        "using_trisycl": "false",
    },
)

config_setting(
    name = "using_sycl_trisycl",
    define_values = {
        "using_sycl": "true",
        "using_trisycl": "true",
    },
)


cc_library(
    name = "sycl",
    hdrs = glob([
        "**/*.h",
        "**/*.hpp",
    ]) + ["@opencl_headers//:OpenCL-Headers"],
    includes = [".", "include"],
    deps = ["@opencl_headers//:OpenCL-Headers"],
)
