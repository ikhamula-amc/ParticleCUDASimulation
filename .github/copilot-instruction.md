# GitHub Copilot Instructions for CMake C++ Development

This repository contains a particle simulation project using CUDA and C++ with CMake as the build system.

## Project Overview
- **Language**: C++ with CUDA extensions
- **Build System**: CMake
- **Target**: GPU-accelerated particle simulations

## Coding Standards
- Use C++17 standard with CUDA compatibility
- Follow consistent naming conventions (camelCase for variables, PascalCase for classes)
- Use RAII principles and smart pointers
- Prefer const correctness
- Document complex algorithms and CUDA kernels

## Project Structure
- `src/`: Source files (.cpp, .cu)
- `include/`: Header files (.h, .cuh)
- `CMakeLists.txt`: Root build configuration
- `README.md`: Project documentation

## CUDA Development Guidelines
- Use CUDA runtime API (not driver API)
- Optimize for memory coalescing and occupancy
- Handle CUDA errors properly with error checking macros
- Use thrust library for parallel algorithms when appropriate
- Profile kernels with nvprof or Nsight

## Build and Development Workflow

**Configure and build:**
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**Run executable:**
```bash
./build/Release/ParticleCUDASimulation      # Unix/macOS
.\build\Release\ParticleCUDASimulation.exe  # Windows
```

**Run tests (if available):**
```bash
ctest --test-dir build --output-on-failure
```

## Common Patterns
- Kernel launch configuration: `kernel<<<blocks, threads>>>(args)`
- Memory management: `cudaMalloc`, `cudaMemcpy`, `cudaFree`
- Error checking: `CUDA_CHECK(cudaDeviceSynchronize())`
- Particle data structures: SOA (Struct of Arrays) preferred for GPU access

## Dependencies
- CUDA Toolkit (nvcc compiler)
- CMake 3.18+
- Thrust (included with CUDA)
- Dear ImGui (for real-time visualization and parameter control)
- GLFW or SDL (for window and OpenGL context management)
- Optional: Google Test for unit testing

## GUI Development with ImGui
- Use Dear ImGui for creating interactive control panels and real-time visualization
- Integrate ImGui with GLFW for cross-platform window management
- Render GUI elements after simulation steps for parameter adjustment
- Common widgets: sliders for simulation parameters, checkboxes for toggles, plots for data visualization
- Keep GUI rendering separate from CUDA kernels to avoid blocking GPU computations
- Use ImGui's docking system for customizable layouts

When suggesting code, prioritize performance, correctness, and maintainability for GPU-accelerated simulations.