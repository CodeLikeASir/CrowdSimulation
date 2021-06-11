#pragma once
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include "Person.h"
#include "Macros.h"

namespace SF_CUDA
{
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

	__device__ inline int cellPosToCell(int x, int y)
	{
		return x + y * CELLS_PER_AXIS;
	}

	__device__ inline void debugCUDA(Person* grid)
	{
		grid[0].velocity = make_float2(13.f, 37.f);
		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
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

	__device__ inline bool float2_isnan(float2 vec)
	{
		return isnan(vec.x) || isnan(vec.y);
	}

	inline float2 getRandomPos()
	{
		float x = rand() % (CELL_SIZE * CELLS_PER_AXIS - SAFEZONE) + SAFEZONE;
		float y = rand() % (CELL_SIZE * CELLS_PER_AXIS - SAFEZONE) + SAFEZONE;
		return make_float2(x, y);
	}

	__device__ int posToCell(int x, int y);
	void add_to_grid(const Person& p);
	__device__ float2 calculateSF(Person* personA, Person* personB);

	void simulate();
	std::vector<PersonVisuals> convertToVisual(bool debugPrint);
	float2 simToGL(float2 pos);

	void init();
	void initTest();
	void close();
}