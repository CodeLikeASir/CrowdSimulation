#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include <cstdlib>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include "SocialForce.h"
/*
inline std::string float2ToString(float2 vec)
{
	return "(" + std::to_string(vec.x) + "|" + std::to_string(vec.y) + ")";
}

struct PersonVisuals
{
	float2 position;
	float2 direction;
};

class SF_Sequential
{
public:
	Person cells[totalCells * maxOccupation];
	float2 forceVectors[totalCells * maxOccupation * 9];
	int debug_level = 0;
	
	const float S = .2f;
	const float R = 4.f;
	float delta = 0.1f;
	const float epsilon = .75f;
	const float theta = 0.3f;

	const float euler = 2.7182818284f;

	float maxDiff = 0.f;

public:
	// Helper functions
	static float2 getRandomfloat2(short min, short max);
	static float2 normalize(float2 vector);

	float2 sum(const float2& lhs, const float2& rhs)
	{
		return make_float2(lhs.x + rhs.x, lhs.y + rhs.y);
	}

	float2 multiply(float lhs, const float2& rhs) const
	{
		return make_float2(lhs * rhs.x, lhs * rhs.y);
	}

	float2 mutiply(float lhs, const float2& rhs) const
	{
		return make_float2(lhs * rhs.x, lhs * rhs.y);
	}

	float2 diff(const float2& lhs, const float2& rhs)
	{
		return make_float2(lhs.x - rhs.x, lhs.y - rhs.y);
	}

	float2 divide(const float2& lhs, const float rhs)
	{
		return make_float2(lhs.x / rhs, lhs.y / rhs);
	}

	float2 tofloat2(const float2& in)
	{
		return make_float2(in.x, in.y);
	}

	float2 udiff(const float2& lhs, const float2& rhs)
	{
		return make_float2(lhs.x - rhs.x, lhs.y - rhs.y);
	}

	float dot(const float2& lhs, const float2& rhs)
	{
		return lhs.x * rhs.x + lhs.y * rhs.y;
	}

	float2 abs(float2 vec)
	{
		return make_float2(std::abs(vec.x), std::abs(vec.y));
	}

	float magnitude(const float2& vec)
	{
		return std::sqrtf(vec.x * vec.x + vec.y * vec.y);
	}

	float2 multiply(float2 vec, float scalar);

	float2 calculate_ve(float v, float2 e);

	// Initializes cells
	void init();

	float2 calculateSF(Person* personA, Person* personB);

	static std::string floatToString(const float2 vec)
	{
		return "(" + std::to_string(vec.x) + " | " + std::to_string(vec.y) + ")";
	}

	// Prints states of cells
	void printGrid();
	bool addToGrid(Person* p);

	// Calculates population of cell
	int getCellPopulation(int cell)
	{
		int population = 0;
		for (int i = cell * maxOccupation; i < (cell + 1) * maxOccupation; i++)
		{
			if (cells[i].state != FREE)
				population++;
		}

		return population;
	}

	int posToCell(float2 pos)
	{
		int x = pos.x / cellSize;
		int y = pos.y / cellSize;
		return x + y * cellsPerAxis;
	}

	int cellPosToIndex(float2 cellPos)
	{
		return cellPosToIndex(cellPos.x, cellPos.y);
	}

	int cellPosToIndex(short x, short y)
	{
		return x + y * cellsPerAxis;
	}

	bool addToGrid(Person p);

	float2 add_force(Person* p, float2 shortForce);
	void update_positions();
	void updatePosition(Person* p, float2 newPos);
	float2 getRandomPos();
	void debugerino();
	void completeTraversal();
	bool reserveSpace(Person* p, int cell, float2 newPos);
	bool reserveSpace(Person* p, int cell);
	
	int getCellIndex(int x, int y);
	//void calculateCellForce(int cellIndex, int posX, int posY);
	int posToCell(int x, int y);
	Person* init_test1(float newDelta);
	Person* init_test2(float newDelta);
	Person* init_test3(float newDelta);

	void simulate(int steps);

	void hard_reset();

	std::vector<PersonVisuals> convertToVisual(bool debugPrint);
	float2 simToGL(float2 pos);
	float2 dirToGL(float2 dir);
};
*/