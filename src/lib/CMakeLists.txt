add_library(branchnet_lib STATIC
            utils.cc utils.h
            trace_interface.cc trace_interface.h)
target_include_directories(branchnet_lib PUBLIC .)
target_compile_features(branchnet_lib PUBLIC cxx_std_17)