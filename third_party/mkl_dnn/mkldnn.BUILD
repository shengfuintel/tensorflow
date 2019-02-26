exports_files(["LICENSE"])

config_setting(
    name = "clang_linux_x86_64",
    values = {
        "cpu": "k8",
        "define": "using_clang=true",
    },
)

cc_library(
    name = "mkl_dnn",
    srcs = glob([
        "src/common/*.cpp",
        "src/common/*.hpp",
        "src/cpu/*.cpp",
        "src/cpu/*.hpp",
        "src/cpu/gemm/*.cpp",
        "src/cpu/gemm/*.hpp",
        "src/cpu/xbyak/*.h",
        "src/sycl/*.cpp",
        "src/sycl/*.hpp",
        "src/sycl/cxxapi/*.cpp",
        "src/ocl/*.cpp",
        "src/ocl/*.hpp",
    ]),
    hdrs = glob(["include/*"]),
    copts = ["-fexceptions"] + select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            "-fopenmp",  # only works with gcc
        ],
        # TODO(ibiryukov): enable openmp with clang by including libomp as a
        # dependency.
        ":clang_linux_x86_64": [],
        "//conditions:default": [],
    }),
    includes = [
        "include",
        "src",
        "src/common",
        "src/cpu",
        "src/cpu/xbyak",
    ],
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
    deps = ["@local_config_sycl//sycl"],
)
