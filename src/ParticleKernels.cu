#include "Particle.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while (0)

// Initialize random states for each particle
__global__ void initRandomStates(curandState* states, unsigned long seed, int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles)
        curand_init(seed, idx, 0, &states[idx]);
}

// Initialize particles with random properties
__global__ void initParticles(Particle* particles, curandState* randStates, 
                               SimulationParams params, int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    curandState localState = randStates[idx];
    
    // Random position in spawn radius
    float angle = curand_uniform(&localState) * 2.0f * 3.14159265f;
    float radius = curand_uniform(&localState) * params.spawnRadius;
    
    particles[idx].position.x = params.spawnPosition.x + radius * cosf(angle);
    particles[idx].position.y = params.spawnPosition.y;
    particles[idx].position.z = params.spawnPosition.z + radius * sinf(angle);
    
    // Random upward velocity (fountain effect)
    float velocityMag = 3.0f + curand_uniform(&localState) * 2.0f;
    float velAngle = curand_uniform(&localState) * 2.0f * 3.14159265f;
    float spreadAngle = (curand_uniform(&localState) - 0.5f) * 0.5f; // Small horizontal spread
    
    particles[idx].velocity.x = spreadAngle * cosf(velAngle);
    particles[idx].velocity.y = velocityMag;
    particles[idx].velocity.z = spreadAngle * sinf(velAngle);
    
    // Random color (gradient from yellow to red)
    float colorVar = curand_uniform(&localState);
    particles[idx].color.x = 1.0f;
    particles[idx].color.y = 0.5f + colorVar * 0.5f;
    particles[idx].color.z = 0.1f;
    particles[idx].color.w = 1.0f;
    
    // Random lifetime
    particles[idx].lifetime = curand_uniform(&localState) * params.maxLifetimeRange;
    particles[idx].maxLifetime = params.minLifetime + curand_uniform(&localState) * params.maxLifetimeRange;
    particles[idx].size = params.particleSize;
    
    randStates[idx] = localState;
}

// Update particle physics
__global__ void updateParticles(Particle* particles, SimulationParams params, 
                                 curandState* randStates, int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    Particle& p = particles[idx];
    
    // Update lifetime
    p.lifetime += params.deltaTime;
    
    // Respawn if lifetime exceeded
    if (p.lifetime >= p.maxLifetime)
    {
        curandState localState = randStates[idx];
        
        // Reset to spawn position
        float angle = curand_uniform(&localState) * 2.0f * 3.14159265f;
        float radius = curand_uniform(&localState) * params.spawnRadius;
        
        p.position.x = params.spawnPosition.x + radius * cosf(angle);
        p.position.y = params.spawnPosition.y;
        p.position.z = params.spawnPosition.z + radius * sinf(angle);
        
        // Random upward velocity
        float velocityMag = 3.0f + curand_uniform(&localState) * 2.0f;
        float velAngle = curand_uniform(&localState) * 2.0f * 3.14159265f;
        float spreadAngle = (curand_uniform(&localState) - 0.5f) * 0.5f;
        
        p.velocity.x = spreadAngle * cosf(velAngle);
        p.velocity.y = velocityMag;
        p.velocity.z = spreadAngle * sinf(velAngle);
        
        p.lifetime = 0.0f;
        p.maxLifetime = params.minLifetime + curand_uniform(&localState) * params.maxLifetimeRange;
        
        randStates[idx] = localState;
    }
    
    // Apply gravity
    p.velocity.x += params.gravity.x * params.deltaTime;
    p.velocity.y += params.gravity.y * params.deltaTime;
    p.velocity.z += params.gravity.z * params.deltaTime;
    
    // Apply drag
    float dragFactor = 1.0f - params.drag * params.deltaTime;
    p.velocity.x *= dragFactor;
    p.velocity.y *= dragFactor;
    p.velocity.z *= dragFactor;
    
    // Update position
    p.position.x += p.velocity.x * params.deltaTime;
    p.position.y += p.velocity.y * params.deltaTime;
    p.position.z += p.velocity.z * params.deltaTime;
    
    // Boundary collision detection
    // Ground collision
    if (p.position.y < params.boundaryMin.y)
    {
        p.position.y = params.boundaryMin.y;
        p.velocity.y = -p.velocity.y * params.groundRestitution;
    }
    
    // Side boundaries (wrap around or bounce)
    if (p.position.x < params.boundaryMin.x)
    {
        p.position.x = params.boundaryMin.x;
        p.velocity.x = -p.velocity.x * 0.5f;
    }
    if (p.position.x > params.boundaryMax.x)
    {
        p.position.x = params.boundaryMax.x;
        p.velocity.x = -p.velocity.x * 0.5f;
    }
    if (p.position.z < params.boundaryMin.z)
    {
        p.position.z = params.boundaryMin.z;
        p.velocity.z = -p.velocity.z * 0.5f;
    }
    if (p.position.z > params.boundaryMax.z)
    {
        p.position.z = params.boundaryMax.z;
        p.velocity.z = -p.velocity.z * 0.5f;
    }
    
    // Fade color based on lifetime
    float lifeFraction = p.lifetime / p.maxLifetime;
    p.color.w = 1.0f - lifeFraction; // Fade alpha
}

// Copy particle positions to VBO for rendering
__global__ void copyPositionsToVBO(Particle* particles, float* positions, 
                                     int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    // Copy position (x, y, z) and size to VBO
    positions[idx * 4 + 0] = particles[idx].position.x;
    positions[idx * 4 + 1] = particles[idx].position.y;
    positions[idx * 4 + 2] = particles[idx].position.z;
    positions[idx * 4 + 3] = particles[idx].size;
}

// Host-callable wrapper functions
extern "C"
{
    curandState* d_randStates = nullptr;
    
    void launchInitRandomStates(unsigned long seed, int numParticles)
    {
        if (d_randStates == nullptr)
            CUDA_CHECK(cudaMalloc(&d_randStates, numParticles * sizeof(curandState)));
        
        int threadsPerBlock = 256;
        int blocks = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
        initRandomStates<<<blocks, threadsPerBlock>>>(d_randStates, seed, numParticles);
        CUDA_CHECK(cudaGetLastError());
    }
    
    void launchInitParticles(Particle* d_particles, SimulationParams params, int numParticles)
    {
        int threadsPerBlock = 256;
        int blocks = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
        initParticles<<<blocks, threadsPerBlock>>>(d_particles, d_randStates, params, numParticles);
        CUDA_CHECK(cudaGetLastError());
    }
    
    void launchUpdateParticles(Particle* d_particles, SimulationParams params, int numParticles)
    {
        int threadsPerBlock = 256;
        int blocks = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
        updateParticles<<<blocks, threadsPerBlock>>>(d_particles, params, d_randStates, numParticles);
        CUDA_CHECK(cudaGetLastError());
    }
    
    void launchCopyPositionsToVBO(Particle* d_particles, float* d_positions, int numParticles)
    {
        int threadsPerBlock = 256;
        int blocks = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
        copyPositionsToVBO<<<blocks, threadsPerBlock>>>(d_particles, d_positions, numParticles);
        CUDA_CHECK(cudaGetLastError());
    }
    
    void cleanupRandomStates()
    {
        if (d_randStates != nullptr)
        {
            cudaFree(d_randStates);
            d_randStates = nullptr;
        }
    }
}
