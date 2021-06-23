#pragma once
// Change to use either CUDA or sequential implementation
#define	 USE_CUDA			true

// Should be a squared value for optimal spacing
#define  SPAWNED_ACTORS		3

// Number of actors drawn on screen (others will still be simulated)
#define  DRAWN_ACTORS		1

// Number of cells per axis, this value squared = total cells
#define  CELLS_PER_AXIS		3

// Max number of actors per cell
#define  MAX_OCCUPATION		32

#define  TOTAL_CELLS		(CELLS_PER_AXIS * CELLS_PER_AXIS)
#define  TOTAL_SPACES		(TOTAL_CELLS * MAX_OCCUPATION)

// Size of cell in meters
#define  CELL_SIZE			4

// Safezone to screen edges
#define	 SAFEZONE			2 

// Minimum distance to goal, before it's marked as reached
#define  MIN_DIST			1.f

// Default speed of actors
#define  SPEED				3.f

// Max FPS of simulation,
// only relevant if simulation takes less than 1/MAX_FPS seconds
#define	 MAX_FPS			60.

// Parameters for social force calculations

// Weight of social force
#define  S					1.f
// Weight of own velocity
#define  EPSILON			1.f
// Max distance of social force
#define  R					4.f
// Time delta simulated per step
#define  DELTA				(1.f / MAX_FPS)
// Weight of sf for actors behind others
#define  THETA				0.2f
// Rounded version of Euler's number
#define  EULER				2.7182818284f 

// Defines weight of congestion avoidance and dispersion force
#define AVOIDANCE_FORCE		1.f