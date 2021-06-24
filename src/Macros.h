#pragma once
// Change to use either CUDA or sequential implementation
constexpr auto USE_CUDA = true;

// Should be a squared value for optimal spacing
constexpr auto SPAWNED_ACTORS = 2048;

// Number of actors drawn on screen (others will still be simulated)
constexpr auto DRAWN_ACTORS = 0;

// Number of cells per axis, this value squared = total cells
constexpr auto CELLS_PER_AXIS = 12;

// Max number of actors per cell
constexpr auto MAX_OCCUPATION = 32;

constexpr auto TOTAL_CELLS = (CELLS_PER_AXIS * CELLS_PER_AXIS);
constexpr auto TOTAL_SPACES = (TOTAL_CELLS * MAX_OCCUPATION);

// Size of cell in meters
constexpr auto CELL_SIZE = 4;

// Safezone to screen edges
constexpr auto SAFEZONE = 2;

// Minimum distance to goal, before it's marked as reached
constexpr auto MIN_DIST = 1.f;

// Default speed of actors
constexpr auto SPEED = 3.f;

// Max FPS of simulation,
// only relevant if simulation takes less than 1/MAX_FPS seconds
constexpr auto MAX_FPS = 60.;

// Parameters for social force calculations

// Weight of social force
constexpr auto S = 0.f;
// Weight of own velocity
constexpr auto EPSILON = 1.f;
// Max distance of social force
constexpr auto R = 4.f;
// Time delta simulated per step
constexpr auto DELTA = (1.f / MAX_FPS);
// Weight of sf for actors behind others
constexpr auto THETA = 0.2f;
// Rounded version of Euler's number
constexpr auto EULER = 2.7182818284f;

// Defines weight of congestion avoidance and dispersion force
constexpr auto AVOIDANCE_FORCE = 1.f;