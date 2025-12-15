#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

class ParticleRenderer {
public:
    ParticleRenderer();
    ~ParticleRenderer();
    
    void initialize(int maxParticles);
    void render(int numParticles, const glm::mat4& viewProj);
    void clearScreen(float r, float g, float b);
    
    unsigned int getVBO() const { return m_vbo; }
    
private:
    void createShaders();
    void createBuffers(int maxParticles);
    
    unsigned int m_vao;
    unsigned int m_vbo;
    unsigned int m_shaderProgram;
    
    int m_maxParticles;
};
