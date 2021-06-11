#pragma once

// Change to use either CUDA or sequential implementation
#define	 USE_CUDA			false

// Should be a squared value for optimal spacing
#define  SPAWNED_ACTORS		3

// Number of cells per axis, this value squared = total cells
#define  CELLS_PER_AXIS		3

#define  CELL_SIZE			4 // Size of cell in meters
#define  MAX_OCCUPATION		32 // Max number of actors per cell
#define	 SAFEZONE			2 // Safezone to screen edges
#define  TOTAL_CELLS		(CELLS_PER_AXIS * CELLS_PER_AXIS)
#define  TOTAL_SPACES		(TOTAL_CELLS * MAX_OCCUPATION)
#define  MIN_DIST			1.f // Minimum distance to goal, before it's reached
#define  SPEED				3.f // Default speed of actors
#define  DRAWN_ACTORS		16384 // Number of actors drawn on screen (others will still be simulated)
#define	 MAX_FPS			60.f

// Parameters for social force calculations
#define  S					1.f // Weight of social force
#define  EPSILON			1.f // Weight of own velocity
#define  R					4.f // Max distance of social force
#define  DELTA				(1.f / MAX_FPS) // Time delta simulated per step
#define  THETA				0.2f // Weight of sf for actors behind others
#define  EULER				2.7182818284f // Rounded version of Euler's number

// Defines weight of congestion avoidance and dispersion force
#define AVOIDANCE_FORCE		1.f