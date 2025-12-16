/**
 * Unit tests for ParticleSystem class
 * 
 * These tests focus on CPU-side logic including:
 * - Parameter validation and clamping
 * - Setter/getter functionality
 * - State management
 * 
 * CUDA runtime and kernel calls are mocked to avoid GPU dependency.
 * 
 * Test Organization:
 * - ParticleSystemLifecycleTest - Constructor/Destructor tests
 * - ParticleSystemGettersTest - Getter method tests
 * - ParticleSystemBasicSettersTest - Setters without validation
 * - ParticleSystemClampingTest - Setters with clamping validation
 * - ParticleSystemStateTest - State management tests
 * - ParticleSystemIntegrationTest - Multi-operation tests
 */

#include <gtest/gtest.h>
#include <glm/glm.hpp>
#include <memory>

// Include mocks BEFORE ParticleSystem to override CUDA functions
#include "cuda_mocks.h"
#include "../include/ParticleSystem.h"

/**
 * Base Test Fixture - Shared utilities for all ParticleSystem tests
 */
class ParticleSystemTestBase : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Reset mock state before each test
        CudaMockState::getInstance().reset();
    }
    
    void TearDown() override
    {
        // Cleanup after each test
        CudaMockState::getInstance().reset();
    }
    
    // Helper: Create a ParticleSystem with default settings
    std::unique_ptr<ParticleSystem> createDefaultSystem(int maxParticles = 1000)
    {
        return std::make_unique<ParticleSystem>(maxParticles);
    }
    
    // Helper: Verify two vec3s are approximately equal
    void expectVec3Near(const glm::vec3& expected, const glm::vec3& actual, float epsilon = 0.001f)
    {
        EXPECT_NEAR(expected.x, actual.x, epsilon);
        EXPECT_NEAR(expected.y, actual.y, epsilon);
        EXPECT_NEAR(expected.z, actual.z, epsilon);
    }
};

// ============================================================
// Test Group 1: Lifecycle Tests (Constructor/Destructor)
// ============================================================

class ParticleSystemLifecycleTest : public ParticleSystemTestBase
{ };

TEST_F(ParticleSystemLifecycleTest, Constructor_SetsDefaultParameters)
{
    // Arrange & Act
    auto system = createDefaultSystem(5000);
    
    // Assert - Verify default parameters
    const auto& params = system->getParams();
    
    expectVec3Near(glm::vec3(0.0f, -9.81f, 0.0f), params.gravity);
    expectVec3Near(glm::vec3(0.0f, 0.5f, 0.0f), params.spawnPosition);
    expectVec3Near(glm::vec3(-5.0f, 0.0f, -5.0f), params.boundaryMin);
    expectVec3Near(glm::vec3(5.0f, 10.0f, 5.0f), params.boundaryMax);
    
    EXPECT_FLOAT_EQ(0.2f, params.spawnRadius);
    EXPECT_FLOAT_EQ(0.6f, params.groundRestitution);
    EXPECT_FLOAT_EQ(0.1f, params.drag);
    EXPECT_FLOAT_EQ(0.05f, params.particleSize);
    EXPECT_EQ(5000, params.numParticles);
    EXPECT_EQ(5000, system->getNumParticles());
}

TEST_F(ParticleSystemLifecycleTest, Constructor_AllocatesDeviceMemory)
{
    // Arrange & Act
    auto system = createDefaultSystem(1000);
    auto& mockState = CudaMockState::getInstance();
    
    // Assert - Verify CUDA allocations were called
    EXPECT_GE(mockState.mallocCallCount, 2); // At least d_particles and d_params
    EXPECT_EQ(2, mockState.allocations.size());
}

TEST_F(ParticleSystemLifecycleTest, Destructor_FreesDeviceMemory)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    auto& mockState = CudaMockState::getInstance();
    int allocCountBefore = mockState.mallocCallCount;
    
    // Act
    system.reset();

    // Assert - Verify cleanup
    EXPECT_GE(mockState.freeCallCount, 2); // d_particles and d_params
}

// ============================================================
// Test Group 2: Getter Tests
// ============================================================

class ParticleSystemGettersTest : public ParticleSystemTestBase
{ };

TEST_F(ParticleSystemGettersTest, GetNumParticles_ReturnsCorrectValue)
{
    // Arrange
    auto system = createDefaultSystem(3000);
    
    // Act
    int numParticles = system->getNumParticles();
    
    // Assert
    EXPECT_EQ(3000, numParticles);
}

TEST_F(ParticleSystemGettersTest, GetParams_ReturnsCorrectReference)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    
    // Act
    const SimulationParams& params = system->getParams();
    
    // Assert - Verify we got the actual params structure
    EXPECT_EQ(1000, params.numParticles);
    EXPECT_FLOAT_EQ(0.6f, params.groundRestitution);
}

// ============================================================
// Test Group 3: Basic Setters (No Validation/Clamping)
// ============================================================

class ParticleSystemBasicSettersTest : public ParticleSystemTestBase
{ };

// --- Gravity Setter Tests ---

TEST_F(ParticleSystemBasicSettersTest, SetGravity_UpdatesParameter)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    glm::vec3 newGravity(0.0f, -5.0f, 0.0f);
    
    // Act
    system->setGravity(newGravity);
    
    // Assert
    const auto& params = system->getParams();
    expectVec3Near(newGravity, params.gravity);
}

TEST_F(ParticleSystemBasicSettersTest, SetGravity_AcceptsZeroGravity)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    glm::vec3 zeroGravity(0.0f, 0.0f, 0.0f);
    
    // Act
    system->setGravity(zeroGravity);
    
    // Assert
    expectVec3Near(zeroGravity, system->getParams().gravity);
}

TEST_F(ParticleSystemBasicSettersTest, SetGravity_AcceptsNegativeValues)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    glm::vec3 customGravity(-1.0f, -2.0f, -3.0f);
    
    // Act
    system->setGravity(customGravity);
    
    // Assert
    expectVec3Near(customGravity, system->getParams().gravity);
}

// --- Spawn Position Setter Tests ---

TEST_F(ParticleSystemBasicSettersTest, SetSpawnPosition_UpdatesParameter)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    glm::vec3 newPos(1.5f, 2.5f, 3.5f);
    
    // Act
    system->setSpawnPosition(newPos);
    
    // Assert
    expectVec3Near(newPos, system->getParams().spawnPosition);
}

TEST_F(ParticleSystemBasicSettersTest, SetSpawnPosition_AcceptsNegativeCoordinates)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    glm::vec3 negativePos(-2.0f, -1.0f, -3.0f);
    
    // Act
    system->setSpawnPosition(negativePos);
    
    // Assert
    expectVec3Near(negativePos, system->getParams().spawnPosition);
}

// --- Particle Count Setter Tests (With Validation) ---

TEST_F(ParticleSystemBasicSettersTest, SetNumParticles_ValidCount_UpdatesCount)
{
    // Arrange
    auto system = createDefaultSystem(10000);
    
    // Act
    system->setNumParticles(5000);
    
    // Assert
    EXPECT_EQ(5000, system->getNumParticles());
    EXPECT_EQ(5000, system->getParams().numParticles);
}

TEST_F(ParticleSystemBasicSettersTest, SetNumParticles_MaximumValue_Accepted)
{
    // Arrange
    auto system = createDefaultSystem(8000);
    
    // Act
    system->setNumParticles(8000);
    
    // Assert
    EXPECT_EQ(8000, system->getNumParticles());
}

TEST_F(ParticleSystemBasicSettersTest, SetNumParticles_ExceedsMax_Ignored)
{
    // Arrange
    auto system = createDefaultSystem(5000);
    int originalCount = system->getNumParticles();
    
    // Act
    system->setNumParticles(10000); // Exceeds max
    
    // Assert - Should remain unchanged
    EXPECT_EQ(originalCount, system->getNumParticles());
}

TEST_F(ParticleSystemBasicSettersTest, SetNumParticles_ZeroCount_Ignored)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    int originalCount = system->getNumParticles();
    
    // Act
    system->setNumParticles(0);
    
    // Assert - Should remain unchanged
    EXPECT_EQ(originalCount, system->getNumParticles());
}

TEST_F(ParticleSystemBasicSettersTest, SetNumParticles_NegativeCount_Ignored)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    int originalCount = system->getNumParticles();
    
    // Act
    system->setNumParticles(-500);
    
    // Assert - Should remain unchanged
    EXPECT_EQ(originalCount, system->getNumParticles());
}

// --- Particle Size Setter Tests ---

TEST_F(ParticleSystemBasicSettersTest, SetParticleSize_UpdatesParameter)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    
    // Act
    system->setParticleSize(0.1f);
    
    // Assert
    EXPECT_FLOAT_EQ(0.1f, system->getParams().particleSize);
}

TEST_F(ParticleSystemBasicSettersTest, SetParticleSize_AcceptsSmallValues)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    
    // Act
    system->setParticleSize(0.001f);
    
    // Assert
    EXPECT_FLOAT_EQ(0.001f, system->getParams().particleSize);
}

TEST_F(ParticleSystemBasicSettersTest, SetParticleSize_AcceptsLargeValues)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    
    // Act
    system->setParticleSize(10.0f);
    
    // Assert
    EXPECT_FLOAT_EQ(10.0f, system->getParams().particleSize);
}

// ============================================================
// Test Group 4: Clamping Validation (Setters with Range Limits)
// ============================================================

class ParticleSystemClampingTest : public ParticleSystemTestBase
{ };

// --- Ground Restitution Clamping Tests ---

TEST_F(ParticleSystemClampingTest, SetGroundRestitution_ValidValue_UpdatesParameter)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    
    // Act
    system->setGroundRestitution(0.8f);
    
    // Assert
    EXPECT_FLOAT_EQ(0.8f, system->getParams().groundRestitution);
}

TEST_F(ParticleSystemClampingTest, SetGroundRestitution_Zero_Accepted)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    
    // Act
    system->setGroundRestitution(0.0f);
    
    // Assert
    EXPECT_FLOAT_EQ(0.0f, system->getParams().groundRestitution);
}

TEST_F(ParticleSystemClampingTest, SetGroundRestitution_One_Accepted)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    
    // Act
    system->setGroundRestitution(1.0f);
    
    // Assert
    EXPECT_FLOAT_EQ(1.0f, system->getParams().groundRestitution);
}

TEST_F(ParticleSystemClampingTest, SetGroundRestitution_ClampsBelowZero)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    
    // Act
    system->setGroundRestitution(-0.5f);
    
    // Assert - Should be clamped to 0
    EXPECT_FLOAT_EQ(0.0f, system->getParams().groundRestitution);
}

TEST_F(ParticleSystemClampingTest, SetGroundRestitution_ClampsAboveOne)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    
    // Act
    system->setGroundRestitution(1.5f);
    
    // Assert - Should be clamped to 1
    EXPECT_FLOAT_EQ(1.0f, system->getParams().groundRestitution);
}

TEST_F(ParticleSystemClampingTest, SetGroundRestitution_ClampsVeryLargeValue)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    
    // Act
    system->setGroundRestitution(100.0f);
    
    // Assert - Should be clamped to 1
    EXPECT_FLOAT_EQ(1.0f, system->getParams().groundRestitution);
}

// --- Drag Clamping Tests ---

TEST_F(ParticleSystemClampingTest, SetDrag_ValidValue_UpdatesParameter)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    
    // Act
    system->setDrag(0.5f);
    
    // Assert
    EXPECT_FLOAT_EQ(0.5f, system->getParams().drag);
}

TEST_F(ParticleSystemClampingTest, SetDrag_Zero_Accepted)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    
    // Act
    system->setDrag(0.0f);
    
    // Assert
    EXPECT_FLOAT_EQ(0.0f, system->getParams().drag);
}

TEST_F(ParticleSystemClampingTest, SetDrag_One_Accepted)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    
    // Act
    system->setDrag(1.0f);
    
    // Assert
    EXPECT_FLOAT_EQ(1.0f, system->getParams().drag);
}

TEST_F(ParticleSystemClampingTest, SetDrag_ClampsBelowZero)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    
    // Act
    system->setDrag(-0.3f);
    
    // Assert - Should be clamped to 0
    EXPECT_FLOAT_EQ(0.0f, system->getParams().drag);
}

TEST_F(ParticleSystemClampingTest, SetDrag_ClampsAboveOne)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    
    // Act
    system->setDrag(2.0f);
    
    // Assert - Should be clamped to 1
    EXPECT_FLOAT_EQ(1.0f, system->getParams().drag);
}

TEST_F(ParticleSystemClampingTest, SetDrag_ClampsVeryNegativeValue)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    
    // Act
    system->setDrag(-50.0f);
    
    // Assert - Should be clamped to 0
    EXPECT_FLOAT_EQ(0.0f, system->getParams().drag);
}

// ============================================================
// Test Group 5: State Management Tests
// ============================================================

class ParticleSystemStateTest : public ParticleSystemTestBase
{ };

TEST_F(ParticleSystemStateTest, Initialize_CallsCudaMemcpy)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    auto& mockState = CudaMockState::getInstance();
    mockState.memcpyCallCount = 0; // Reset after constructor
    
    // Act
    system->initialize();
    
    // Assert - Should copy params to device
    EXPECT_GE(mockState.memcpyCallCount, 1);
}

TEST_F(ParticleSystemStateTest, Update_CallsKernelLauncher)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    system->initialize();
    
    // Act - Just verify it doesn't crash
    system->update(0.016f);
    
    // Assert - Update should execute without error
    // (Actual kernel execution is mocked)
    SUCCEED();
}

TEST_F(ParticleSystemStateTest, Reset_ReinitializesSystem)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    system->initialize();
    auto& mockState = CudaMockState::getInstance();
    int memcpyBefore = mockState.memcpyCallCount;
    
    // Act
    system->reset();
    
    // Assert - Reset should trigger reinitialization
    EXPECT_GT(mockState.memcpyCallCount, memcpyBefore);
}

// ============================================================
// Test Group 6: Integration Tests (Multi-Operation Scenarios)
// ============================================================

class ParticleSystemIntegrationTest : public ParticleSystemTestBase
{ };

TEST_F(ParticleSystemIntegrationTest, MultipleParameterChanges_AllApplied)
{
    // Arrange
    auto system = createDefaultSystem(10000);
    
    // Act - Change multiple parameters
    system->setGravity(glm::vec3(0.0f, -15.0f, 0.0f));
    system->setSpawnPosition(glm::vec3(2.0f, 3.0f, 4.0f));
    system->setNumParticles(7500);
    system->setParticleSize(0.08f);
    system->setGroundRestitution(0.9f);
    system->setDrag(0.3f);
    
    // Assert - All changes should be reflected
    const auto& params = system->getParams();
    expectVec3Near(glm::vec3(0.0f, -15.0f, 0.0f), params.gravity);
    expectVec3Near(glm::vec3(2.0f, 3.0f, 4.0f), params.spawnPosition);
    EXPECT_EQ(7500, system->getNumParticles());
    EXPECT_FLOAT_EQ(0.08f, params.particleSize);
    EXPECT_FLOAT_EQ(0.9f, params.groundRestitution);
    EXPECT_FLOAT_EQ(0.3f, params.drag);
}

TEST_F(ParticleSystemIntegrationTest, ClampingBehavior_PreservesValidRange)
{
    // Arrange
    auto system = createDefaultSystem(1000);
    
    // Act - Set to boundary values
    system->setGroundRestitution(0.0f);
    system->setDrag(1.0f);
    
    // Assert - Boundary values should be preserved
    EXPECT_FLOAT_EQ(0.0f, system->getParams().groundRestitution);
    EXPECT_FLOAT_EQ(1.0f, system->getParams().drag);

    // Act - Set beyond boundaries
    system->setGroundRestitution(-1.0f);
    system->setDrag(5.0f);
    
    // Assert - Should clamp to valid range
    EXPECT_FLOAT_EQ(0.0f, system->getParams().groundRestitution);
    EXPECT_FLOAT_EQ(1.0f, system->getParams().drag);
}

// ============================================================
// End of Tests
// ============================================================
