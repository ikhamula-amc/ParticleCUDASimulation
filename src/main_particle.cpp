#include <glad/glad.h>  // Must be first - OpenGL loader
#define GLFW_INCLUDE_NONE
#include "OpenglWindow.h"
#include "ParticleSystem.h"
#include "ParticleRenderer.h"
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <iostream>
#include <chrono>

int main()
{
    constexpr int width = 1920;
    constexpr int height = 1080;
    
    // Create window
    OpenglWindow window("Particle CUDA Simulation", width, height);
    
    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window.getGLFWHandle(), true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    // Create particle system and renderer
    constexpr int maxParticles = 100000;
    ParticleSystem particleSystem(maxParticles);
    ParticleRenderer particleRenderer;
    
    particleRenderer.initialize(maxParticles);
    particleSystem.registerVBO(particleRenderer.getVBO());
    particleSystem.initialize();
    
    // Camera setup
    glm::vec3 cameraPos(0.0f, 3.0f, 8.0f);
    glm::vec3 cameraTarget(0.0f, 2.0f, 0.0f);
    glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, glm::vec3(0, 1, 0));
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 
                                             (float)width / (float)height, 
                                             0.1f, 100.0f);
    glm::mat4 viewProj = projection * view;
    
    // Simulation state
    bool isRunning = true;
    int numParticles = 50000;
    float gravity = -9.81f;
    float particleSize = 0.05f;
    float restitution = 0.6f;
    float drag = 0.1f;
    glm::vec3 spawnPos(0.0f, 0.5f, 0.0f);
    
    particleSystem.setNumParticles(numParticles);

    auto lastTime = std::chrono::high_resolution_clock::now();
    float fps = 0.0f;

    ImVec4 clearColor = ImVec4(0.25f, 0.45f, 0.50f, 1.00f);
    
    // Main loop
    while (!window.shouldClose())
    {
        // Calculate delta time
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;
        fps = 1.0f / deltaTime;
        
        // Poll events
        window.pollEvents();
        
        // Update particle system
        if (isRunning)
            particleSystem.update(deltaTime);
        
        // Clear screen
        particleRenderer.clearScreen(clearColor.x, clearColor.y, clearColor.z);
        
        // Map VBO and copy particle data
        particleSystem.mapGraphicsResource(particleRenderer.getVBO());
        particleSystem.unmapGraphicsResource();
        
        // Render particles
        particleRenderer.render(numParticles, viewProj);
        
        // ImGui UI
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        ImGui::Begin("Particle Fountain Control", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("FPS: %.1f (%.3f ms/frame)", fps, 1000.0f / fps);
        ImGui::Separator();
        
        // Simulation controls
        if (ImGui::Button(isRunning ? "Pause" : "Play"))
            isRunning = !isRunning;
        
        ImGui::SameLine();
        if (ImGui::Button("Reset"))
            particleSystem.reset();
        
        ImGui::Separator();
        ImGui::Text("Particle Settings");
        
        if (ImGui::SliderInt("Particle Count", &numParticles, 1000, maxParticles))
            particleSystem.setNumParticles(numParticles);
        
        if (ImGui::SliderFloat("Particle Size", &particleSize, 0.01f, 0.2f))
            particleSystem.setParticleSize(particleSize);
        
        ImGui::Separator();
        ImGui::Text("Physics Settings");
        
        if (ImGui::SliderFloat("Gravity", &gravity, -20.0f, 0.0f))
            particleSystem.setGravity(glm::vec3(0.0f, gravity, 0.0f));
        
        if (ImGui::SliderFloat("Bounce", &restitution, 0.0f, 1.0f))
            particleSystem.setGroundRestitution(restitution);
        
        if (ImGui::SliderFloat("Drag", &drag, 0.0f, 1.0f))
            particleSystem.setDrag(drag);

        ImGui::Separator();
        ImGui::Text("Background Color: ");
        ImGui::SameLine();
        ImGui::ColorEdit3("", (float*)&clearColor);

        ImGui::Separator();
        ImGui::Text("Spawn Position");
        if (ImGui::SliderFloat3("Position", &spawnPos.x, -3.0f, 3.0f))
            particleSystem.setSpawnPosition(spawnPos);
        
        ImGui::End();
        
        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        // Swap buffers
        window.swapBuffers();
    }
    
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    return 0;
}
