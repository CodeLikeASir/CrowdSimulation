#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include <cstdlib>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include "Math_Helper.cuh"
#include "SocialForce.h"
#include "Macros.h"

class SF_Sequential
{
public:
	//float2 forceVectors[TOTAL_CELLS * MAX_OCCUPATION * 9];
	int debug_level = 0;

	static int cellPosToIndex(short x, short y)
	{
		return x + y * CELLS_PER_AXIS;
	}

	static int personToCell(int index)
	{
		return index / MAX_OCCUPATION;
	}

	static short2 cellToCellPos(int index)
	{
		return make_short2(index % CELLS_PER_AXIS, index / CELLS_PER_AXIS);
	}

	static int cellPosToCell(short2 pos)
	{
		return pos.x + pos.y * CELLS_PER_AXIS;
	}

	static int cellPosToCell(int x, int y)
	{
		return x + y * CELLS_PER_AXIS;
	}

	static int posToCell(int x, int y)
	{
		int cellX = x / CELL_SIZE;
		int cellY = y / CELL_SIZE;
		return cellX + cellY * CELLS_PER_AXIS;
	}

	static int getFirstCellIndex(dim3 cell)
	{
		return (cell.x + cell.y * CELLS_PER_AXIS) * MAX_OCCUPATION;
	}

	float2 getRandomPos()
	{
		short x = rand() % (CELL_SIZE * CELLS_PER_AXIS - SAFEZONE) + SAFEZONE;
		short y = rand() % (CELL_SIZE * CELLS_PER_AXIS - SAFEZONE) + SAFEZONE;
		return make_float2(x, y);
	}


public:
	static std::string floatToString(const float2 vec)
	{
		return "(" + std::to_string(vec.x) + " | " + std::to_string(vec.y) + ")";
	}

	int getCellPopulation(uint3 cell);
	void init();
	float2 calculateSF(struct Person* personA, Person* personB);
	int toIndex(int x, int y);
	bool addToGrid(Person p);
	void host_function();
	float2 calculateCellForce(int cellAPop, uint3 pseudeBlockIdx, uint3 pseudoThreadIdx);
	void update_position(Person personA, float2 total_force, int cellIndex, int threadX);
	void completeMove(uint3 pseudoBlockIdx);
	std::vector<struct PersonVisuals> convertToVisual(bool debugPrint);
	float2 simToGL(float2 pos);
};