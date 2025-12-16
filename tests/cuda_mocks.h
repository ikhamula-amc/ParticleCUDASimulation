#pragma once

/**
 * Mock CUDA runtime and kernel functions for unit testing.
 * 
 * This header provides stub implementations of CUDA functions
 * to allow testing CPU-side logic without requiring a GPU.
 * 
 * Usage: Include this header in test files before including
 * headers that reference CUDA functions.
 */

#include <cstddef>
#include <map>
#include <glad/glad.h>

// Mock CUDA types
enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2
};

typedef cudaError cudaError_t;

// Mock cudaGraphicsResource as a struct pointer
struct cudaGraphicsResource {
    int dummy; // Need at least one member
};
typedef struct cudaGraphicsResource* cudaGraphicsResource_t;

enum cudaGraphicsMapFlags {
    cudaGraphicsMapFlagsNone = 0,
    cudaGraphicsMapFlagsReadOnly = 1,
    cudaGraphicsMapFlagsWriteDiscard = 2
};

// Mock cudaMemcpy kinds
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3
};

// Mock memory tracking for validation
class CudaMockState {
public:
    static CudaMockState& getInstance() {
        static CudaMockState instance;
        return instance;
    }
    
    std::map<void*, size_t> allocations;
    bool shouldFailAlloc = false;
    int mallocCallCount = 0;
    int freeCallCount = 0;
    int memcpyCallCount = 0;
    int syncCallCount = 0;
    
    void reset() {
        allocations.clear();
        shouldFailAlloc = false;
        mallocCallCount = 0;
        freeCallCount = 0;
        memcpyCallCount = 0;
        syncCallCount = 0;
    }
};

// Mock CUDA runtime functions
template<typename T>
inline cudaError_t cudaMalloc(T** devPtr, size_t size) {
    auto& state = CudaMockState::getInstance();
    state.mallocCallCount++;
    
    if (state.shouldFailAlloc) {
        *devPtr = nullptr;
        return cudaErrorMemoryAllocation;
    }
    
    *devPtr = (T*)malloc(size);
    state.allocations[(void*)*devPtr] = size;
    return cudaSuccess;
}

inline cudaError_t cudaFree(void* devPtr) {
    auto& state = CudaMockState::getInstance();
    state.freeCallCount++;
    
    if (devPtr && state.allocations.count(devPtr)) {
        free(devPtr);
        state.allocations.erase(devPtr);
    }
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int kind) {
    (void)kind; // Unused
    auto& state = CudaMockState::getInstance();
    state.memcpyCallCount++;
    
    if (dst && src && count > 0) {
        memcpy(dst, src, count);
    }
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize() {
    CudaMockState::getInstance().syncCallCount++;
    return cudaSuccess;
}

inline const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "no error";
        case cudaErrorInvalidValue: return "invalid value";
        case cudaErrorMemoryAllocation: return "memory allocation failed";
        default: return "unknown error";
    }
}

inline cudaError_t cudaGraphicsGLRegisterBuffer(cudaGraphicsResource_t* resource, 
                                                 GLuint buffer, 
                                                 unsigned int flags) {
    (void)buffer;
    (void)flags;
    *resource = (cudaGraphicsResource_t)malloc(sizeof(cudaGraphicsResource));
    return cudaSuccess;
}

inline cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) {
    if (resource) {
        free(resource);
    }
    return cudaSuccess;
}

inline cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, int stream) {
    (void)count;
    (void)resources;
    (void)stream;
    return cudaSuccess;
}

inline cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, int stream) {
    (void)count;
    (void)resources;
    (void)stream;
    return cudaSuccess;
}

inline cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, 
                                                         cudaGraphicsResource_t resource) {
    (void)resource;
    // Allocate a dummy buffer
    *devPtr = malloc(1024 * 1024); // 1MB buffer
    *size = 1024 * 1024;
    return cudaSuccess;
}
