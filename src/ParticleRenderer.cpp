#include "ParticleRenderer.h"
#include <iostream>
#include <glm/gtc/type_ptr.hpp>

// Vertex shader for point sprites
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec4 aPositionSize;

uniform mat4 uViewProj;
uniform float uPointSize;

void main() {
    gl_Position = uViewProj * vec4(aPositionSize.xyz, 1.0);
    gl_PointSize = aPositionSize.w * uPointSize * 100.0; // Scale point size
}
)";

// Fragment shader for particles
const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

void main() {
    // Create circular particle
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    if (dist > 0.5) {
        discard;
    }
    
    // Smooth edges
    float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
    
    // Gradient color from yellow to orange
    vec3 color = mix(vec3(1.0, 1.0, 0.3), vec3(1.0, 0.5, 0.1), dist * 2.0);
    
    FragColor = vec4(color, alpha * 0.8);
}
)";

ParticleRenderer::ParticleRenderer()
    : m_vao(0)
    , m_vbo(0)
    , m_shaderProgram(0)
    , m_maxParticles(0)
{
}

ParticleRenderer::~ParticleRenderer()
{
    if (m_vao)
        glDeleteVertexArrays(1, &m_vao);
    if (m_vbo)
        glDeleteBuffers(1, &m_vbo);
    if (m_shaderProgram)
        glDeleteProgram(m_shaderProgram);
}

void ParticleRenderer::initialize(int maxParticles)
{
    m_maxParticles = maxParticles;
    
    createShaders();
    createBuffers(maxParticles);
    
    std::cout << "ParticleRenderer initialized for " << maxParticles << " particles" << std::endl;
}

void ParticleRenderer::createShaders()
{
    // Compile vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);
    
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cerr << "Vertex shader compilation failed:\n" << infoLog << std::endl;
    }
    
    // Compile fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cerr << "Fragment shader compilation failed:\n" << infoLog << std::endl;
    }
    
    // Link shader program
    m_shaderProgram = glCreateProgram();
    glAttachShader(m_shaderProgram, vertexShader);
    glAttachShader(m_shaderProgram, fragmentShader);
    glLinkProgram(m_shaderProgram);
    
    glGetProgramiv(m_shaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(m_shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Shader program linking failed:\n" << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void ParticleRenderer::createBuffers(int maxParticles)
{
    // Create VAO
    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);
    
    // Create VBO (position + size per particle)
    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, maxParticles * 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    
    // Position + size attribute
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    
    glBindVertexArray(0);
}

void ParticleRenderer::render(int numParticles, const glm::mat4& viewProj)
{
    glUseProgram(m_shaderProgram);
    
    // Set uniforms
    int viewProjLoc = glGetUniformLocation(m_shaderProgram, "uViewProj");
    glUniformMatrix4fv(viewProjLoc, 1, GL_FALSE, glm::value_ptr(viewProj));
    
    int pointSizeLoc = glGetUniformLocation(m_shaderProgram, "uPointSize");
    glUniform1f(pointSizeLoc, 1.0f);
    
    // Enable point sprites and blending
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE); // Additive blending for glow effect
    
    // Render particles
    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, numParticles);
    glBindVertexArray(0);
    
    glDisable(GL_BLEND);
    glDisable(GL_PROGRAM_POINT_SIZE);
}

void ParticleRenderer::clearScreen(float r, float g, float b)
{
    glClearColor(r, g, b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
}
