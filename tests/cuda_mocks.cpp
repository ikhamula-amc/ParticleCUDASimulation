/**
 * Mock CUDA kernel launcher functions for unit testing.
 * 
 * These stub implementations allow ParticleSystem to be tested
 * without requiring actual CUDA kernels or GPU hardware.
 */

#include "../include/Particle.h"

// Mock kernel launcher implementations
extern "C" {
    void launchInitRandomStates(unsigned long seed, int numParticles) {
        // Mock: Do nothing - random state initialization is GPU-only
        (void)seed;
        (void)numParticles;
    }
    
    void launchInitParticles(Particle* d_particles, SimulationParams params, int numParticles) {
        // Mock: Could initialize particles with deterministic values for testing
        (void)d_particles;
        (void)params;
        (void)numParticles;
    }
    
    void launchUpdateParticles(Particle* d_particles, SimulationParams params, int numParticles) {
        // Mock: Do nothing - particle update is GPU-only
        (void)d_particles;
        (void)params;
        (void)numParticles;
    }
    
    void launchCopyPositionsToVBO(Particle* d_particles, float* d_positions, int numParticles) {
        // Mock: Do nothing - VBO copy is GPU-only
        (void)d_particles;
        (void)d_positions;
        (void)numParticles;
    }
    
    void cleanupRandomStates() {
        // Mock: Do nothing - cleanup is GPU-only
    }
}
