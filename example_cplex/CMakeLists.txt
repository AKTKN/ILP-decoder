cmake_minimum_required(VERSION 3.10)
project(CplexDemo)

set(CMAKE_CXX_STANDARD 17)

set(CPLEX_STUDIO_DIR "/opt/ibm/ILOG/CPLEX_Studio2212")

# directories of cplex headerfile
include_directories(
    ${CPLEX_STUDIO_DIR}/cplex/include
    ${CPLEX_STUDIO_DIR}/concert/include
)

# directories of cplex and concert
link_directories(
    ${CPLEX_STUDIO_DIR}/cplex/lib/x86-64_linux/static_pic
    ${CPLEX_STUDIO_DIR}/concert/lib/x86-64_linux/static_pic
)

add_compile_options(-m64 -O3 -fPIC -fexceptions -fno-strict-aliasing -DIL_STD)

add_executable(cplex_demo cplex_tutorial.cpp)

target_link_libraries(cplex_demo
    ilocplex
    concert
    cplex
    m         # Math library
    pthread   # POSIX Threads
    dl        # Dynamic linking
)

