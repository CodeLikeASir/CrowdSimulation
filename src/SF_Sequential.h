#pragma once
#include "Macros.h"
#include "device_launch_parameters.h"
#include <string>
#include <vector>

struct PersonVisuals;
#include "PersonStructs.h"

// Sequential CPU-based version of SF_CUDA
namespace SF_Sequential
{
	static int getFirstCellIndex(dim3 cell)
	{
		return (cell.x + cell.y * CELLS_PER_AXIS) * MAX_OCCUPATION;
	}
	
	static std::string floatToString(const float2 vec)
	{
		return "(" + std::to_string(vec.x) + " | " + std::to_string(vec.y) + ")";
	}

	void init();
	
	int getCellPopulation(uint3 cell);
	float2 calculateSF(struct Person* personA, Person* personB);
	void add_to_grid(const Person& p);
	void simulate();
	float2 calculateCellForce(int cellAPop, uint3 pBlockIdx, uint3 pThreadIdx);
	void update_position(Person personA, float2 total_force, int cellIndex, int threadX);
	void completeMove(uint3 pBlockIdx);
	std::vector<PersonVisuals> convertToVisual();
}
