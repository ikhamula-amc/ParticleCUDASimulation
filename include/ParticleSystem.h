#pragma once

#include <glad/glad.h>  // OpenGL loader - must be before cuda_gl_interop
#include "Particle.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class ParticleSystem {
public:
    ParticleSystem(int maxParticles);
    ~ParticleSystem();

    // Initialize particles
    void initialize();
    
    // Update simulation
    void update(float deltaTime);
    
    // Reset simulation
    void reset();
    
    // Getters
    int getNumParticles() const { return m_numParticles; }
    const SimulationParams& getParams() const { return m_params; }
    
    // Setters for parameters
    void setGravity(const glm::vec3& gravity);
    void setSpawnPosition(const glm::vec3& pos);
    void setNumParticles(int count);
    void setParticleSize(float size);
    void setGroundRestitution(float restitution);
    void setDrag(float drag);
    
    // Map/unmap for rendering
    void mapGraphicsResource(unsigned int vbo);
    void unmapGraphicsResource();
    float* getMappedPositions() { return d_positions; }
    
    // Register OpenGL VBO with CUDA
    void registerVBO(unsigned int vbo);
    void unregisterVBO();

private:
    // Host parameters
    SimulationParams m_params;
    int m_numParticles;
    int m_maxParticles;
    
    // Device memory
    Particle* d_particles;
    float* d_positions;  // For rendering (mapped from VBO)
    SimulationParams* d_params;
    
    // CUDA-OpenGL interop
    cudaGraphicsResource* m_cudaVBOResource;
    bool m_vboRegistered;
    
    // Random number generation
    void initializeParticlesOnGPU();
};
