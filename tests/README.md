# Unit Testing for ParticleCUDASimulation

This directory contains unit tests for the ParticleCUDASimulation project.

## 🔍 Testing Context

**Language:** C++17  
**Framework:** Google Test v1.14.0  
**Test Location:** `tests/`  
**Naming Convention:** `*_test.cpp`  
**Mocking:** Custom CUDA mocks (no GPU required)  
**Run Command:** `ctest` or `ctest --output-on-failure`

## 📦 Test Structure

```
tests/
├── ParticleSystem_test.cpp   # Unit tests for ParticleSystem class
├── cuda_mocks.h              # Mock CUDA runtime functions
├── cuda_mocks.cpp            # Mock CUDA kernel launchers
└── README.md                 # This file
```

## 🚀 Running Tests

### 1. Configure CMake (if not already done)

```powershell
cmake -S . -B build
```

### 2. Build Tests

```powershell
cmake --build build --config Debug --target ParticleSystemTests
```

### 3. Run Tests

**Option A: Using CTest (recommended)**
```powershell
cd build
ctest -C Debug --output-on-failure
```

**Option B: Run Test Executable Directly**
```powershell
.\build\Debug\ParticleSystemTests.exe
```

**Option C: Run Specific Tests**
```powershell
.\build\Debug\ParticleSystemTests.exe --gtest_filter=ParticleSystemTest.SetGravity*
```

### 4. Generate Test Report

```powershell
ctest -C Debug --output-on-failure --verbose
```

## ✅ Current Test Coverage

### Tested Components (CPU-side logic only)

- ✅ **Constructor** - Default parameter initialization
- ✅ **Destructor** - Memory cleanup
- ✅ **Getters** - `getNumParticles()`, `getParams()`
- ✅ **Setters with validation:**
  - `setGravity()` - No validation (accepts any vec3)
  - `setSpawnPosition()` - No validation (accepts any vec3)
  - `setNumParticles()` - Validates against max particles, rejects invalid values
  - `setParticleSize()` - No validation (accepts any positive value)
  - `setGroundRestitution()` - **Clamps to [0.0, 1.0]**
  - `setDrag()` - **Clamps to [0.0, 1.0]**
- ✅ **State management** - `initialize()`, `update()`, `reset()`

### Test Statistics

- **Total Tests:** 37
- **Test Cases Covered:**
  - Constructor/Destructor: 3 tests
  - Getters: 2 tests
  - Gravity setter: 3 tests
  - Spawn position setter: 2 tests
  - Particle count setter: 5 tests
  - Particle size setter: 3 tests
  - Ground restitution setter: 6 tests (including clamping)
  - Drag setter: 6 tests (including clamping)
  - State management: 3 tests
  - Integration tests: 2 tests

## ⚠️ Testing Limitations

### What IS Tested
✅ Parameter validation logic (clamping, bounds checking)  
✅ Setter/getter functionality  
✅ Memory allocation calls (via mocks)  
✅ State transition logic  

### What IS NOT Tested
❌ **Actual CUDA kernel execution** (GPU-side physics)  
❌ **Particle position updates** (requires GPU)  
❌ **OpenGL rendering integration** (requires graphics context)  
❌ **CUDA-OpenGL interop** (requires both GPU and GL context)  
❌ **Performance characteristics** (timing, throughput)  

### Why These Limitations?

The `ParticleSystem` class is **tightly coupled** to CUDA runtime:
- Constructor immediately calls `cudaMalloc` - can't instantiate without CUDA
- All meaningful operations invoke GPU kernels
- No abstraction layer for dependency injection

**These are UNIT tests with mocked dependencies, not integration tests.**

## 🔧 Mocking Strategy

### CUDA Runtime Mocks (`cuda_mocks.h`)

Provides stub implementations of:
- `cudaMalloc()` / `cudaFree()` - Tracks allocations using `std::map`
- `cudaMemcpy()` - Uses regular `memcpy` on host memory
- `cudaDeviceSynchronize()` - No-op
- `cudaGraphicsGL*()` - Stub implementations
- Error handling functions

### CUDA Kernel Mocks (`cuda_mocks.cpp`)

Stubs for external C functions:
- `launchInitRandomStates()` - No-op
- `launchInitParticles()` - No-op
- `launchUpdateParticles()` - No-op
- `launchCopyPositionsToVBO()` - No-op
- `cleanupRandomStates()` - No-op

### Mock State Tracking

`CudaMockState` singleton tracks:
- Active allocations (`std::map<void*, size_t>`)
- Call counts (`mallocCallCount`, `freeCallCount`, etc.)
- Failure injection (`shouldFailAlloc`)

## 🎯 Best Practices Demonstrated

### 1. **Arrange-Act-Assert Pattern**
```cpp
TEST_F(ParticleSystemTest, SetDrag_ClampsBelowZero) {
    // Arrange - Setup
    ParticleSystem* system = createDefaultSystem(1000);
    
    // Act - Execute
    system->setDrag(-0.3f);
    
    // Assert - Verify
    EXPECT_FLOAT_EQ(0.0f, system->getParams().drag);
    
    delete system;
}
```

### 2. **Test Fixtures for Common Setup**
```cpp
class ParticleSystemTest : public ::testing::Test {
protected:
    void SetUp() override {
        CudaMockState::getInstance().reset();
    }
    
    ParticleSystem* createDefaultSystem(int maxParticles) {
        return new ParticleSystem(maxParticles);
    }
};
```

### 3. **Descriptive Test Names**
- Format: `MethodName_StateUnderTest_ExpectedBehavior`
- Example: `SetGroundRestitution_ClampsBelowZero`

### 4. **Boundary Testing**
- Zero values, max values, negative values
- Edge cases like `setNumParticles(0)` and `setNumParticles(exceeds_max)`

### 5. **Floating-Point Comparison**
```cpp
EXPECT_FLOAT_EQ(0.8f, system->getParams().groundRestitution);
expectVec3Near(expected, actual, 0.001f);
```

## 🏗️ Refactoring Recommendations

To enable **comprehensive unit testing**, consider refactoring:

### Option 1: Dependency Injection

```cpp
// Interface for CUDA operations
class ICudaAllocator {
public:
    virtual cudaError_t malloc(void** ptr, size_t size) = 0;
    virtual cudaError_t free(void* ptr) = 0;
    virtual ~ICudaAllocator() = default;
};

// Real implementation
class CudaAllocator : public ICudaAllocator { ... };

// Mock for testing
class MockCudaAllocator : public ICudaAllocator { ... };

// Inject via constructor
class ParticleSystem {
public:
    ParticleSystem(int maxParticles, ICudaAllocator* allocator = nullptr);
};
```

### Option 2: Separate CPU and GPU Logic

```cpp
// CPU-only parameter management
class ParticleSystemParams {
public:
    void setGravity(glm::vec3 gravity);
    void setDrag(float drag);
    // ... all setters/getters
};

// GPU-specific operations
class ParticleSystemGPU {
public:
    ParticleSystemGPU(ParticleSystemParams& params);
    void initialize();
    void update(float dt);
};
```

### Option 3: Abstract Kernel Launcher

```cpp
class IParticleKernels {
public:
    virtual void initParticles(Particle* particles, SimulationParams params) = 0;
    virtual void updateParticles(Particle* particles, SimulationParams params) = 0;
    virtual ~IParticleKernels() = default;
};
```

## 📊 Example Test Output

```
[==========] Running 37 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 37 tests from ParticleSystemTest
[ RUN      ] ParticleSystemTest.Constructor_SetsDefaultParameters
[       OK ] ParticleSystemTest.Constructor_SetsDefaultParameters (0 ms)
[ RUN      ] ParticleSystemTest.SetDrag_ClampsBelowZero
[       OK ] ParticleSystemTest.SetDrag_ClampsBelowZero (0 ms)
...
[----------] 37 tests from ParticleSystemTest (15 ms total)

[==========] 37 tests from 1 test suite ran. (15 ms total)
[  PASSED  ] 37 tests.
```

## 🐛 Debugging Failed Tests

### View Detailed Output
```powershell
ctest -C Debug --output-on-failure --verbose
```

### Run Single Test with Debug Info
```powershell
.\build\Debug\ParticleSystemTests.exe --gtest_filter=ParticleSystemTest.SetDrag_ClampsBelowZero --gtest_break_on_failure
```

### Enable Visual Studio Debugger
1. Set `ParticleSystemTests` as startup project
2. Set breakpoint in test
3. Press F5 to debug

## 📚 Additional Resources

- [Google Test Documentation](https://google.github.io/googletest/)
- [Google Mock Documentation](https://google.github.io/googletest/gmock_for_dummies.html)
- [CMake Testing](https://cmake.org/cmake/help/latest/manual/ctest.1.html)

## 🤝 Contributing

When adding new tests:

1. **Follow naming convention:** `ClassName_MethodName_ExpectedBehavior`
2. **Use Arrange-Act-Assert** pattern
3. **Test one behavior per test**
4. **Include boundary cases** (zero, negative, max values)
5. **Clean up resources** in test teardown
6. **Document complex test logic** with comments

## ✨ Future Improvements

- [ ] Add integration tests that run on actual GPU
- [ ] Test VBO registration/unregistration (requires OpenGL context)
- [ ] Performance benchmarks for parameter updates
- [ ] Memory leak detection with Valgrind/Dr. Memory
- [ ] Code coverage analysis (gcov/lcov)
- [ ] Refactor `ParticleSystem` for better testability
- [ ] Add tests for `ParticleRenderer` class
- [ ] Add tests for `OpenglWindow` class

---

**Last Updated:** December 15, 2025  
**Test Framework Version:** Google Test 1.14.0
