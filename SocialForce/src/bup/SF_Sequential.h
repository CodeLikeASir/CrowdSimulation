#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include <cstdlib>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>

const static int cellsPerAxis = 6;
const static int totalCells = cellsPerAxis * cellsPerAxis;
const static int cellSize = 4;
const static int maxOccupation = 32;
const float speed = 2.0f;

const static short FREE = 1;
const static short OCCUPIED = 2;
const static short RESERVED = 3;
const static  short TRAVERSING = 4;

inline std::string short2ToString(short2 vec)
{
	return "(" + std::to_string(vec.x) + "|" + std::to_string(vec.y) + ")";
}

inline std::string ushort2ToString(ushort2 vec)
{
	return "(" + std::to_string(vec.x) + "|" + std::to_string(vec.y) + ")";
}

struct PersonVisuals
{
	float2 position;
	float2 direction;
};

struct Person
{
	short state;
	ushort2 position;
	ushort2 goal;
	short2 velocity;
	short2 direction;

	short velocityMag()
	{
		return std::abs(velocity.x) + std::abs(velocity.y);
	}

	Person(ushort2 pos, ushort2 g)
	{
		state = OCCUPIED;
		position = pos;
		goal = g;

		direction = make_short2(0, 0);
		short2 goalDir = make_short2(goal.x - position.x, goal.y - position.y);

		if (goalDir.x > 0) direction.x = 1;
		else if (goalDir.x < 0) direction.x = -1;

		if (goalDir.y > 0) direction.y = 1;
		else if (goalDir.y < 0) direction.y = -1;

		velocity = make_short2(direction.x * speed, direction.y * speed);
	}

	Person()
	{
		state = FREE;
		position = make_ushort2(1337, 1337);
		goal = make_ushort2(0, 0);
		direction = make_short2(0, 0);
		velocity = make_short2(0, 0);
	}

	std::string toString()
	{
		if (state == FREE)
			return "FREE";

		return "Hi. I'm currently at " + ushort2ToString(position) + " and moving to " + ushort2ToString(goal) + ". " +
			"My velocity is " + short2ToString(velocity) + " and my direction is " + short2ToString(direction) + ".";
	}
};

const static float S = 1.f;
const static float R = 4.f;
const static float delta = .1f;
const static float epsilon = 1.f;
const static float theta = 0.9f;

const static float euler = 2.7182818284f;

class SF_Sequential
{
public:
	Person grid[totalCells * maxOccupation];
	float2 forceVectors[totalCells * maxOccupation * 9];
	int debug_level = 0;

public:
	// Helper functions
	static ushort2 getRandomUShort2(short min, short max);
	static short2 getRandomShort2(short min, short max);
	static short2 normalize(short2 vector);

	float2 sum(const float2& lhs, const float2& rhs)
	{
		return make_float2(lhs.x + rhs.x, lhs.y + rhs.y);
	}

	short2 sum(const short2& lhs, const short2& rhs)
	{
		return make_short2(lhs.x + rhs.x, lhs.y + rhs.y);
	}

	ushort2 sum(const ushort2& lhs, const short2& rhs)
	{
		return make_ushort2(lhs.x + rhs.x, lhs.y + rhs.y);
	}

	short2 multiply(float lhs, const short2& rhs) const
	{
		return make_short2(lhs * rhs.x, lhs * rhs.y);
	}

	ushort2 mutiply(float lhs, const ushort2& rhs) const
	{
		return make_ushort2(lhs * rhs.x, lhs * rhs.y);
	}

	short2 diff(const short2& lhs, const short2& rhs)
	{
		return make_short2(lhs.x - rhs.x, lhs.y - rhs.y);
	}

	short2 divide(const short2& lhs, const float rhs)
	{
		return make_short2(lhs.x / rhs, lhs.y / rhs);
	}

	short2 toShort2(const ushort2& in)
	{
		return make_short2(in.x, in.y);
	}

	ushort2 udiff(const ushort2& lhs, const ushort2& rhs)
	{
		return make_ushort2(lhs.x - rhs.x, lhs.y - rhs.y);
	}

	float dot(const short2& lhs, const short2& rhs)
	{
		return lhs.x * rhs.x + lhs.y * rhs.y;
	}

	short2 abs(short2 vec)
	{
		return make_short2(std::abs(vec.x), std::abs(vec.y));
	}

	float magnitude(const ushort2& vec)
	{
		return std::sqrtf(vec.x * vec.x + vec.y * vec.y);
	}

	float magnitude(const short2& vec)
	{
		return std::sqrtf(vec.x * vec.x + vec.y * vec.y);
	}

	short2 multiply(short2 vec, float scalar);

	short2 calculate_ve(float v, short2 e);

	// Initializes grid
	void init();

	float2 calculateSF(Person personA, Person personB);

	static std::string floatToString(const float2 vec)
	{
		return "(" + std::to_string(vec.x) + " | " + std::to_string(vec.y) + ")";
	}

	// Prints states of grid
	void printGrid();

	// Calculates population of cell
	int getCellPopulation(int cell)
	{
		int population = 0;
		for (int i = cell * maxOccupation; i < (cell + 1) * maxOccupation; i++)
		{
			if (grid[i].state != FREE)
				population++;
		}

		return population;
	}

	int posToCell(short2 pos)
	{
		int x = pos.x / cellSize;
		int y = pos.y / cellSize;
		return x + y * cellsPerAxis;
	}

	int cellPosToIndex(short2 cellPos)
	{
		return cellPosToIndex(cellPos.x, cellPos.y);
	}

	int cellPosToIndex(short x, short y)
	{
		return x + y * cellsPerAxis;
	}

	bool addToGrid(Person p);

	ushort2 add_force(ushort2 position, short2 shortForce);
	void update_positions();
	int getCellIndex(int x, int y);
	ushort2 getFirstCellPos(int cellIndex);
	void calculateCellForce(int cellIndex, int posX, int posY);
	int toIndex(int x, int y);
	int getCellIndex();
	Person* init_test1();

	void simulate(int steps);

	std::vector<PersonVisuals> convertToVisual(bool debugPrint);
	float2 simToGL(ushort2 pos);
	float2 simToGL(short2 pos);
};