#pragma once
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

// Collection of math functions used for calculations

// float2 operators
__host__ __device__ inline float2 operator+(const float2& lhs, const float2& rhs)
{
	return make_float2(lhs.x + rhs.x, lhs.y + rhs.y);
}

__host__ __device__ inline float2 operator*(const float lhs, const float2& rhs)
{
	return make_float2(lhs * rhs.x, lhs * rhs.y);
}

__host__ __device__ inline float2 operator*(const float2& lhs, const float rhs)
{
	return rhs * lhs;
}

__host__ __device__ inline float2 operator/(const float2& lhs, const float rhs)
{
	return make_float2(lhs.x / rhs, lhs.y / rhs);
}

__host__ __device__ inline float2 operator-(const float2& lhs, const float2& rhs)
{
	return make_float2(lhs.x - rhs.x, lhs.y - rhs.y);
}

__host__ __device__ inline float dot(const float2& lhs, const float2& rhs)
{
	return lhs.x * rhs.x + lhs.y * rhs.y;
}
__host__ __device__ inline float magnitude(const float2& val)
{
	return sqrtf(val.x * val.x + val.y * val.y);
}

__host__ __device__ inline float2 normalize(const float2 val)
{
	const float mag = sqrtf(val.x * val.x + val.y * val.y);
	return make_float2(val.x / mag, val.y / mag);
}

__host__ __device__ inline bool float2_isnan(float2 vec)
{
	return isnan(vec.x) || isnan(vec.y);
}

// Conversion for positions & indices

// Converts cell coordinates to cell index
__host__ __device__ inline int cellPosToIndex(int x, int y)
{
	return x + y * CELLS_PER_AXIS;
}

// Converts cell coordinates to cell index
__host__ __device__ inline int cellPosToIndex(uint2 pos)
{
	return cellPosToIndex(pos.x, pos.y);
}

// Converts cell coordinates to cell index
__host__ __device__ inline int cellPosToIndex(short2 pos)
{
	return cellPosToIndex(pos.x, pos.y);
}

// Converts cell coordinates to cell index
__host__ __device__ inline int cellPosToIndex(float2 pos)
{
	return cellPosToIndex((int)pos.x, (int)pos.y);
}

// Converts cell index to cell coordinates
__host__ __device__ inline short2 cellIndexToPos(int index)
{
	return make_short2(index % CELLS_PER_AXIS, index / CELLS_PER_AXIS);
}

// Returns cell person is located in based on index
__host__ __device__ inline int personToCell(int index)
{
	return index / MAX_OCCUPATION;
}

// Converts persons position to cell index
__host__ __device__ inline int personPosToCellIndex(float x, float y)
{
	int cellX = x / (float) CELL_SIZE;
	int cellY = y / (float)CELL_SIZE;

	return cellX + cellY * CELLS_PER_AXIS;
}

// Returns random position in grid
// Only works on host due to rand() usage
__host__ inline float2 getRandomPos()
{
	float x = rand() % (CELL_SIZE * CELLS_PER_AXIS);
	float y = rand() % (CELL_SIZE * CELLS_PER_AXIS);

	return make_float2(x, y);
}

// Convert simulation to OpenGL coordinates 
inline float2 simCoordToGL(float2 pos)
{
	float maxVal = CELLS_PER_AXIS * CELL_SIZE;
	float xPos = pos.x / maxVal * 2.f - 1.f;
	float yPos = (pos.y / maxVal * 2.f - 1.f) * -1.f;

	return make_float2(xPos, yPos);
}

inline float dist(float2 posA, float2 posB)
{
	return abs(posA.x - posB.x) + abs(posA.y - posB.y);
}