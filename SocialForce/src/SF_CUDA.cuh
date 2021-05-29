#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include "SocialForce.h"

#define  SPAWNED_ACTORS		65536 // has to be squared value
#define  CELLS_PER_AXIS		120
#define  CELL_SIZE			4
#define	 SAFEZONE			2

#define  EPSILON			10.f
#define  S					.02f
#define  R					2.f
#define  DELTA				1.f
#define  THETA				0.2f
#define  EULER				2.7182818284f
#define  STATE_FREE			1

#define  minDist			2.f
#define  speed				1.f
#define  MAX_OCCUPATION		32

#define TOTAL_CELLS			CELLS_PER_AXIS * CELLS_PER_AXIS
#define TOTAL_SPACES		TOTAL_CELLS * MAX_OCCUPATION

#define DRAWN_ACTORS		100

__device__ inline float2 operator+(const float2& lhs, const float2& rhs)
{
	return make_float2(lhs.x + rhs.x, lhs.y + rhs.y);
}

__device__ inline float2 operator-(const float2& lhs, const float2& rhs)
{
	return make_float2(lhs.x - rhs.x, lhs.y - rhs.y);
}

__device__ inline float dot(const float2& lhs, const float2& rhs)
{
	return lhs.x * rhs.x + lhs.y * rhs.y;
}

__device__ inline float2 operator*(const float lhs, const float2& rhs)
{
	return make_float2(lhs * rhs.x, lhs * rhs.y);
}

__device__ inline float2 operator*(const float2& lhs, const float rhs)
{
	return rhs * lhs;
}

__device__ inline float2 operator/(const float2& lhs, const float rhs)
{
	return make_float2(lhs.x / rhs, lhs.y / rhs);
}

__device__ inline float magnitude(const float2& val)
{
	return sqrtf(val.x * val.x + val.y * val.y);
}

__device__ inline float2 normalize(const float2 val)
{
	const float mag = sqrtf(val.x * val.x + val.y * val.y);
	return make_float2(val.x / mag, val.y / mag);
}

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
void close();