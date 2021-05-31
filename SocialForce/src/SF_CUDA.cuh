#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include "SocialForce.h"

// Define scope of simulation
#define  SPAWNED_ACTORS		81 // should be a squared value for optimal spacing
#define  CELLS_PER_AXIS		12 // Number of cells per axis, this value squared = total cells
#define  CELL_SIZE			4 // Size of cell in meters
#define  MAX_OCCUPATION		32 // Max number of actors per cell
#define	 SAFEZONE			2 // Safezone to screen edges
#define  TOTAL_CELLS		CELLS_PER_AXIS * CELLS_PER_AXIS
#define  TOTAL_SPACES		TOTAL_CELLS * MAX_OCCUPATION
#define  MIN_DIST			.05f // Minimum distance to goal, before it's reached
#define  SPEED				1.f // Default speed of actors
#define  DRAWN_ACTORS		100 // Number of actors drawn on screen (others will still be simulated)

// Parameters for social force calculations
#define  S					1.f // Weight of social force
#define  EPSILON			1.f // Weight of own velocity
#define  R					2.f // Max distance of social force
#define  DELTA				.1f // Time delta simulated per step
#define  THETA				0.2f // Weight of sf for actors behind others
#define  EULER				2.7182818284f // Rounded version of Euler's number

// Defines weight of congestion avoidance and dispersion force
#define AVOIDANCE_FORCE		0.f

__device__ inline int cellPosToIndex(short x, short y)
{
	return x + y * CELLS_PER_AXIS;
}

__device__ inline int personToCell(int index)
{
	return index / MAX_OCCUPATION;
}

__device__ inline short2 cellToCellPos(int index)
{
	return make_short2(index % CELLS_PER_AXIS, index / CELLS_PER_AXIS);
}

__device__ inline int cellPosToCell(short2 pos)
{
	return pos.x + pos.y * CELLS_PER_AXIS;
}

__device__ inline void debugCUDA(Person* grid)
{
	grid[0].velocity = make_float2(13.f, 37.f);
	if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
	{
		grid[0].velocity = make_float2(13.f, 37.f);
	}
}

__device__ inline int maskToInt(int mask)
{
	int result = 0;
	while (mask != 0)
	{
		mask = mask & (mask - 1);
		result++;
	}
	return result;
}

__device__ inline bool isnan(float2 vec)
{
	return isnan(vec.x) || isnan(vec.y);
}

inline float2 normalizeH(const float2 val)
{
	const float mag = sqrtf(val.x * val.x + val.y * val.y);
	return make_float2(val.x / mag, val.y / mag);
}

inline float2 getRandomPos()
{
	short x = rand() % (CELL_SIZE * CELLS_PER_AXIS - SAFEZONE) + SAFEZONE;
	short y = rand() % (CELL_SIZE * CELLS_PER_AXIS - SAFEZONE) + SAFEZONE;
	return make_float2(x,y);
}

__device__ int posToCell(int x, int y);
int toIndexH(int x, int y);
bool addToGrid(Person& p);
__device__ float2 calculateSF(Person* personA, Person* personB);
__device__ bool reserveSpace(Person* grid, int newCell, int oldIndex);

inline float magnitudeH(const float2& val)
{
	return sqrtf(val.x * val.x + val.y * val.y);
}

int simulate();
std::vector<PersonVisuals> convertToVisual(bool debugPrint);
float2 simToGL(float2 pos);

void init();
void initTest();
void close();