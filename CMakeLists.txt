cmake_minimum_required(VERSION 3.28...3.30)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/common/bootstrap.cmake" NO_POLICY_SCOPE)

project(${_name} VERSION ${_version})

option(ENABLE_FRONTEND_API "Use obs-frontend-api for UI functionality" OFF)
option(ENABLE_QT "Use Qt functionality" OFF)

include(compilerconfig)
include(defaults)
include(helpers)

add_library(${CMAKE_PROJECT_NAME} MODULE)

find_package(libobs REQUIRED)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core dnn imgproc objdetect)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::libobs ${OpenCV_LIBS})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_std_20)

if(ENABLE_FRONTEND_API)
  find_package(obs-frontend-api REQUIRED)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::obs-frontend-api)
endif()

if(ENABLE_QT)
  find_package(Qt6 COMPONENTS Widgets Core)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Qt6::Core Qt6::Widgets)
  target_compile_options(
    ${CMAKE_PROJECT_NAME}
    PRIVATE $<$<C_COMPILER_ID:Clang,AppleClang>:-Wno-quoted-include-in-framework-header -Wno-comma>
  )
  set_target_properties(
    ${CMAKE_PROJECT_NAME}
    PROPERTIES AUTOMOC ON AUTOUIC ON AUTORCC ON
  )
endif()

target_sources(
  ${CMAKE_PROJECT_NAME}
  PRIVATE src/plugin-main.cpp
          src/face_emotion_filter.cpp
          src/inference_worker.cpp
          src/tracker.cpp
          src/emotion_mapping.cpp
          src/face_emotion_filter.hpp
          src/inference_worker.hpp
          src/tracker.hpp
          src/emotion_mapping.hpp
)

if(OS_WINDOWS)
  get_filename_component(_opencv_bin_dir "${OpenCV_DIR}/../../bin" ABSOLUTE)
  set(
    _face_emotion_runtime_dlls
    opencv_core4.dll
    opencv_dnn4.dll
    opencv_imgproc4.dll
    opencv_objdetect4.dll
    opencv_calib3d4.dll
    opencv_features2d4.dll
    opencv_flann4.dll
    libprotobuf.dll
    abseil_dll.dll
  )

  foreach(_dll IN LISTS _face_emotion_runtime_dlls)
    if(EXISTS "${_opencv_bin_dir}/${_dll}")
      install(FILES "${_opencv_bin_dir}/${_dll}" DESTINATION "${CMAKE_PROJECT_NAME}/bin/64bit")
      add_custom_command(
        TARGET ${CMAKE_PROJECT_NAME}
        POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_opencv_bin_dir}/${_dll}"
                "${CMAKE_CURRENT_BINARY_DIR}/rundir/$<CONFIG>"
        COMMENT "Copy runtime dependency ${_dll} to rundir"
        VERBATIM
      )
    endif()
  endforeach()
endif()

set_target_properties_plugin(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME ${_name})
