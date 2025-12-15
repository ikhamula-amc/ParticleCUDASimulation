#pragma once

#include <glm/glm.hpp>

// Particle structure (aligned for GPU)
struct Particle {
    glm::vec3 position;
    float _pad1;  // Padding for alignment
    glm::vec3 velocity;
    float _pad2;
    glm::vec4 color;
    float lifetime;   // Current lifetime
    float maxLifetime; // Maximum lifetime before respawn
    float size;
    float _pad3;
};

// Simulation parameters
struct SimulationParams {
    glm::vec3 gravity;
    float deltaTime;
    glm::vec3 spawnPosition;
    float spawnRadius;
    glm::vec3 boundaryMin;
    float groundRestitution; // Bounce coefficient
    glm::vec3 boundaryMax;
    float drag;              // Air resistance
    int numParticles;
    float minLifetime;
    float maxLifetimeRange;
    float particleSize;
};
