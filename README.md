# ParticleCUDASimulation

A real-time **Particle Fountain** simulation powered by CUDA and OpenGL. This demo showcases GPU-accelerated physics simulation with up to 100,000 particles running at 60+ FPS.

## Features

- **CUDA-accelerated physics**: All particle updates run on GPU using CUDA kernels
- **CUDA-OpenGL interop**: Direct GPU memory mapping for zero-copy rendering
- **Real-time controls**: ImGui interface for tweaking simulation parameters
- **Particle fountain effect**: Particles spawn, rise, fall, and bounce with realistic physics

## Physics Simulation

The demo implements:
- Gravity and drag forces
- Ground collision with bounce (restitution coefficient)
- Boundary collision detection
- Particle lifecycle management (respawn after timeout)
- Randomized initial velocities and colors

## Controls

- **Play/Pause**: Start or stop the simulation
- **Reset**: Re-initialize all particles
- **Particle Count**: Adjust from 1,000 to 100,000 particles
- **Particle Size**: Change visual size
- **Gravity**: Control downward acceleration
- **Bounce**: Adjust ground restitution (0 = no bounce, 1 = perfect bounce)
- **Drag**: Air resistance coefficient
- **Spawn Position**: Move the particle fountain source

## Requirements

- CUDA Toolkit 11.0+ (12.8 recommended)
- CMake 3.26+
- C++17 compatible compiler
- OpenGL 3.3+ capable GPU
- Visual Studio 2022 (Windows) or GCC/Clang (Linux/macOS)

## Project Structure

```
ParticleCUDASimulation/
├── include/              # Header files
│   ├── glad/            # OpenGL loader headers
│   ├── KHR/             # Khronos headers
│   ├── Particle.h       # Particle data structures
│   ├── ParticleSystem.h # Particle system interface
│   ├── ParticleRenderer.h # OpenGL rendering interface
│   └── OpenglWindow.h   # Window management
├── src/                 # Source files
│   ├── main_particle.cpp # Main application
│   ├── ParticleSystem.cpp # System implementation
│   ├── ParticleKernels.cu # CUDA kernels
│   ├── ParticleRenderer.cpp # Rendering implementation
│   ├── OpenglWindow.cpp
│   └── glad.c           # OpenGL loader implementation
├── CMakeLists.txt       # Build configuration
└── README.md
```

## Build (CMake)

1. Create a build directory and configure with CMake.

	 - Unix/macOS:

		 ```bash
		 mkdir build && cd build
		 cmake .. -DCMAKE_BUILD_TYPE=Release
		 ```

	 - Windows PowerShell:

		 ```powershell
		 New-Item -ItemType Directory -Name build -Force; Set-Location build
		 cmake .. -DCMAKE_BUILD_TYPE=Release
		 ```

2. Build using CMake's native build command (cross-platform):

	 - For Ninja or Makefile generators:

		 ```bash
		 cmake --build . --config Release -- -j$(nproc)
		 ```

	 - For Visual Studio generators on Windows:

		 ```powershell
		 cmake --build . --config Release
		 ```

3. Run tests:

	 ```bash
	 ctest --output-on-failure
	 ```
