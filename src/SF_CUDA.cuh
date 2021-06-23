#pragma once
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include "PersonStructs.h"

namespace SF_CUDA
{
	// Converts ballot mask to number of positive results
	__device__ inline int mask_to_int(int mask)
	{
		int result = 0;
		while (mask != 0)
		{
			mask = mask & (mask - 1);
			result++;
		}
		return result;
	}

	// Initializes cells, spawns actors and allocates memory on main & device
	void init();

	// Adds new person
	void add_to_grid(const Person& p);

	// Calculates one simulation step
	void simulate();

	// Calculates all social forces on one person
	__global__ void calculateCellForce(Person* device_grid, int* debugVal);

	// Calculate social force of a given person pair
	__device__ float2 calculateSF(Person* personA, Person* personB);

	// Completes move of persons,
	// moves to other cell if needed
	__global__ void completeMove(Person* device_grid);

	// Extracts all simulated persons from cells and returns them in OpenGL coordinates
	std::vector<PersonVisuals> convertToVisual();
}