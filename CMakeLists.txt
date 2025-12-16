cmake_minimum_required(VERSION 3.18)

project(ParticleCUDASimulation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA language
enable_language(CUDA)

# Find CUDA Toolkit (version 12.0 or higher preferred)
find_package(CUDAToolkit REQUIRED)
message(STATUS "Using CUDAToolkit: ${CUDAToolkit_VERSION}")

# Verify CUDA version is acceptable
if(CUDAToolkit_VERSION VERSION_LESS "11.0")
    message(WARNING "CUDA version ${CUDAToolkit_VERSION} is older than recommended (11.0+)")
endif()

# --- Options ---
option(IMGUI_EXAMPLE_BUILD_DEMO "Build ImGui demo" ON)

include(FetchContent)

# =====================================================================
# ImGui
# =====================================================================
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.0  # you can change to desired tag/branch
)

FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
    FetchContent_Populate(imgui)

    # Core ImGui sources
    set(IMGUI_SOURCES
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        # Backends
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )

    # Optional: if you really want a “no demo” build, define IMGUI_EXAMPLE_BUILD_DEMO option above
    # option(IMGUI_EXAMPLE_BUILD_DEMO "Build ImGui demo" ON)
    if(NOT IMGUI_EXAMPLE_BUILD_DEMO)
        list(REMOVE_ITEM IMGUI_SOURCES ${imgui_SOURCE_DIR}/imgui_demo.cpp)
    endif()

    add_library(imgui STATIC ${IMGUI_SOURCES})

    target_include_directories(imgui PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
    )

    # ImGui uses OpenGL3 backend with a GL loader (glad-style)
    target_compile_definitions(imgui PUBLIC
        IMGUI_IMPL_OPENGL_LOADER_GLAD
    )
endif()

# =====================================================================
# External libs
# =====================================================================

# Fetch GLM (header-only math library)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

# Fetch GLFW (window management)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Detect GLFW imported target name (depends on how it was installed)
set(GLFW_TARGET glfw)
if(NOT TARGET glfw AND TARGET glfw::glfw)
    set(GLFW_TARGET glfw::glfw)
endif()

# Very important: ImGui itself needs GLFW includes when compiling imgui_impl_glfw.cpp
target_link_libraries(imgui PUBLIC ${GLFW_TARGET})
target_compile_definitions(imgui PUBLIC GLFW_INCLUDE_NONE)

# =====================================================================
# Executable
# =====================================================================
add_executable(${PROJECT_NAME} 
    src/main_particle.cpp
    src/OpenglWindow.cpp
    src/glad.c
    src/ParticleSystem.cpp
    src/ParticleKernels.cu
    src/ParticleRenderer.cpp
    # Headers (for IDE organization)
    include/OpenglWindow.h
    include/Particle.h
    include/ParticleSystem.h
    include/ParticleRenderer.h
)

# Link everything together
target_link_libraries(${PROJECT_NAME} PRIVATE
    imgui
    ${GLFW_TARGET}
    glm::glm
    CUDA::cudart
    CUDA::curand
)

# Your own headers
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include  # Local include folder (GLAD, KHR)
    ${CMAKE_SOURCE_DIR}/src      # Source folder headers
)

# Don’t let GLFW auto-include GL headers; we use glad
target_compile_definitions(${PROJECT_NAME} PRIVATE
    GLFW_INCLUDE_NONE
)

# Set CUDA architectures (adjust for your GPU: 75=Turing, 86=Ampere, 89=Ada)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89 CACHE STRING "CUDA architectures")
endif()

# CUDA compilation properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

# Compiler warnings
# if(MSVC)
#     target_compile_options(${PROJECT_NAME} PRIVATE /W4)
# else()
#     target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
# endif()

# On Windows, copy CUDA runtime DLL to output directory
if(WIN32 AND TARGET CUDA::cudart)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:CUDA::cudart>
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()

# =====================================================================
# Testing
# =====================================================================
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    enable_testing()
    
    # Fetch Google Test
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    # Create a modified version of ParticleSystem.cpp for testing
    # We need to prevent CUDA headers from being included
    configure_file(
        ${CMAKE_SOURCE_DIR}/src/ParticleSystem.cpp
        ${CMAKE_BINARY_DIR}/tests/ParticleSystem_mock.cpp
        COPYONLY
    )
    
    # Test executable
    add_executable(ParticleSystemTests
        tests/ParticleSystem_test.cpp
        tests/cuda_mocks.cpp
        ${CMAKE_BINARY_DIR}/tests/ParticleSystem_mock.cpp
    )
    
    target_link_libraries(ParticleSystemTests PRIVATE
        GTest::gtest_main
        glm::glm
    )
    
    target_include_directories(ParticleSystemTests PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests
    )
    
    # Mock CUDA runtime for tests (don't link real CUDA, don't include CUDA headers)
    target_compile_definitions(ParticleSystemTests PRIVATE
        UNIT_TEST_MODE=1
    )
    
    # Exclude CUDA toolkit includes from test build
    # This prevents conflicts between our mocks and real CUDA headers
    set_target_properties(ParticleSystemTests PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
    )
    
    # Discover tests
    include(GoogleTest)
    gtest_discover_tests(ParticleSystemTests)
    
    message(STATUS "Unit tests enabled. Run with: ctest")
endif()
