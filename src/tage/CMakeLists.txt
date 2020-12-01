cmake_minimum_required (VERSION 3.5.1)

project(tage CXX)

add_subdirectory(../lib ./lib)

set(bins tagescl64 tagescl56 mtagesc gtagesc gtagesc_nolocal gtage)
foreach(bin IN LISTS bins)
    add_executable(${bin} main.cc)
    target_link_libraries(${bin} branchnet_lib)
    target_compile_features(${bin} PRIVATE cxx_std_17)
endforeach()
target_compile_definitions(tagescl64 PRIVATE PREDICTOR_CONFIG=TAGESCL_64)
target_compile_definitions(tagescl56 PRIVATE PREDICTOR_CONFIG=TAGESCL_56)
target_compile_definitions(mtagesc PRIVATE PREDICTOR_CONFIG=MTAGE_SC)
target_compile_definitions(gtagesc PRIVATE PREDICTOR_CONFIG=GTAGE_SC)
target_compile_definitions(gtagesc_nolocal PRIVATE PREDICTOR_CONFIG=GTAGE_SC_NOLOCAL)
target_compile_definitions(gtage PRIVATE PREDICTOR_CONFIG=GTAGE)