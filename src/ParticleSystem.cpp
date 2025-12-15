#include "ParticleSystem.h"
#include <iostream>
#include <ctime>

// External CUDA kernel launcher functions
extern "C" {
    void launchInitRandomStates(unsigned long seed, int numParticles);
    void launchInitParticles(Particle* d_particles, SimulationParams params, int numParticles);
    void launchUpdateParticles(Particle* d_particles, SimulationParams params, int numParticles);
    void launchCopyPositionsToVBO(Particle* d_particles, float* d_positions, int numParticles);
    void cleanupRandomStates();
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while (0)

ParticleSystem::ParticleSystem(int maxParticles)
    : m_maxParticles(maxParticles)
    , m_numParticles(maxParticles)
    , d_particles(nullptr)
    , d_positions(nullptr)
    , d_params(nullptr)
    , m_cudaVBOResource(nullptr)
    , m_vboRegistered(false)
{
    // Initialize default parameters
    m_params.gravity = glm::vec3(0.0f, -9.81f, 0.0f);
    m_params.deltaTime = 0.016f; // 60 FPS
    m_params.spawnPosition = glm::vec3(0.0f, 0.5f, 0.0f);
    m_params.spawnRadius = 0.2f;
    m_params.boundaryMin = glm::vec3(-5.0f, 0.0f, -5.0f);
    m_params.boundaryMax = glm::vec3(5.0f, 10.0f, 5.0f);
    m_params.groundRestitution = 0.6f;
    m_params.drag = 0.1f;
    m_params.numParticles = m_numParticles;
    m_params.minLifetime = 2.0f;
    m_params.maxLifetimeRange = 3.0f;
    m_params.particleSize = 0.05f;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_particles, m_maxParticles * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(SimulationParams)));
    
    std::cout << "ParticleSystem created with " << m_maxParticles << " max particles" << std::endl;
}

ParticleSystem::~ParticleSystem()
{
    if (m_vboRegistered)
        unregisterVBO();
    
    if (d_particles)
        cudaFree(d_particles);
    
    if (d_params)
        cudaFree(d_params);
    
    cleanupRandomStates();
}

void ParticleSystem::initialize()
{
    // Initialize random states
    launchInitRandomStates(static_cast<unsigned long>(time(nullptr)), m_maxParticles);
    
    // Copy parameters to device
    CUDA_CHECK(cudaMemcpy(d_params, &m_params, sizeof(SimulationParams), cudaMemcpyHostToDevice));
    
    // Initialize particles
    launchInitParticles(d_particles, m_params, m_numParticles);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "ParticleSystem initialized" << std::endl;
}

void ParticleSystem::update(float deltaTime)
{
    m_params.deltaTime = deltaTime;
    
    // Launch update kernel
    launchUpdateParticles(d_particles, m_params, m_numParticles);
}

void ParticleSystem::reset()
{
    initialize();
}

void ParticleSystem::setGravity(const glm::vec3& gravity)
{
    m_params.gravity = gravity;
}

void ParticleSystem::setSpawnPosition(const glm::vec3& pos)
{
    m_params.spawnPosition = pos;
}

void ParticleSystem::setNumParticles(int count)
{
    if (count > 0 && count <= m_maxParticles)
    {
        m_numParticles = count;
        m_params.numParticles = count;
    }
}

void ParticleSystem::setParticleSize(float size)
{
    m_params.particleSize = size;
}

void ParticleSystem::setGroundRestitution(float restitution)
{
    m_params.groundRestitution = glm::clamp(restitution, 0.0f, 1.0f);
}

void ParticleSystem::setDrag(float drag)
{
    m_params.drag = glm::clamp(drag, 0.0f, 1.0f);
}

void ParticleSystem::registerVBO(unsigned int vbo)
{
    if (m_vboRegistered)
        unregisterVBO();
    
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cudaVBOResource, vbo, 
                                             cudaGraphicsMapFlagsWriteDiscard));
    m_vboRegistered = true;
    
    std::cout << "VBO registered with CUDA" << std::endl;
}

void ParticleSystem::unregisterVBO()
{
    if (m_vboRegistered && m_cudaVBOResource)
    {
        cudaGraphicsUnregisterResource(m_cudaVBOResource);
        m_cudaVBOResource = nullptr;
        m_vboRegistered = false;
    }
}

void ParticleSystem::mapGraphicsResource(unsigned int vbo)
{
    if (!m_vboRegistered)
        registerVBO(vbo);
    
    CUDA_CHECK(cudaGraphicsMapResources(1, &m_cudaVBOResource, 0));
    
    size_t size;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &size, m_cudaVBOResource));
    
    // Copy particle positions to VBO
    launchCopyPositionsToVBO(d_particles, d_positions, m_numParticles);
}

void ParticleSystem::unmapGraphicsResource()
{
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cudaVBOResource, 0));
    d_positions = nullptr;
}
