#pragma once
// Limits number of simulation steps (used for profiling and other tests)
// Set to 0 for unlimited simulation
constexpr int SIMULATION_STEPS = 0;

// Size of actors in GL coordinates
const float TRIANGLE_SIZE = 0.0075f;

// Change to use either CUDA or sequential implementation
constexpr bool USE_CUDA = true;

// Defines number of spawned actors
constexpr int SPAWNED_ACTORS = 256;

// Number of cells per axis, this value squared = total cells
constexpr int CELLS_PER_AXIS = 9;

// Max number of actors per cell
constexpr int MAX_OCCUPATION = 32;

// Length of cell in meters, this value squared = cell size in m^2
constexpr int CELL_SIZE = 4;

// Minimum distance to goal, before it's marked as reached
constexpr float MIN_DIST = 1.f;

// Default speed of actors
constexpr float SPEED = 1.34f;

// Max FPS of simulation,
// only relevant if simulation takes less than 1/MAX_FPS seconds
constexpr float MAX_FPS = 30.;

// Parameters for social force calculations

// Weight of social force
constexpr float S = 1.f;
// Weight of own velocity
constexpr float EPSILON = 1.f;
// Max distance of social force
constexpr float R = 4.f;
// Time delta simulated per step
constexpr float DELTA = (1.f / MAX_FPS);
// Weight of sf for actors behind others
constexpr float THETA = 0.2f;
// Rounded version of Euler's number
constexpr float EULER = 2.7182818284f;

// Defines weight of congestion avoidance and dispersion force
constexpr float AVOIDANCE_FORCE = 1.f;