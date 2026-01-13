cmake_minimum_required(VERSION 3.16...3.26)
# cmake_policy(SET CMP0167 NEW)
project(vi)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIRS})

ament_auto_add_executable(ctsvi_node src/ctsvi_node.cpp)
ament_auto_add_executable(ctsvi_ad_node src/ctsvi_ad_node.cpp)
ament_auto_add_executable(atsvi_node src/atsvi_node.cpp)
ament_auto_add_executable(atsvi_ad_node src/atsvi_ad_node.cpp)
ament_auto_add_executable(etsvi_node src/etsvi_node.cpp)
ament_auto_add_executable(etsvi_lyapunov_node src/etsvi_lyapunov_node.cpp)
ament_auto_add_executable(etsvi_op_node src/etsvi_op_node.cpp)
ament_auto_add_executable(rk4_node src/rk4_node.cpp)

install(PROGRAMS
        scripts/ctsvi_sim.py
        scripts/ctsvi_ad_sim.py
        scripts/atsvi_sim.py
        scripts/atsvi_ad_sim.py
        scripts/vi_compare_sim.py
        scripts/etsvi_sim.py
        scripts/rk4_vi_compare_sim.py
        scripts/pybullet_sim.py
        DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()

