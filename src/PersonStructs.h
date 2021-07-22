#pragma once
#include "Config.h"
#include "Math_Helper.cuh"

// Person states
constexpr short FREE = 1;
constexpr short OCCUPIED = 2;
constexpr short RESERVED = 3;
constexpr short LEAVING = 4;

// Simulated person
struct Person
{
	int state = FREE;
	float2 position = make_float2(0, 0);
	float2 goal = make_float2(0, 0);
	float2 velocity = make_float2(0, 0);
	float2 direction = make_float2(0, 0);

	Person() = default;

	Person(float2 pos, float2 g)
	{
		state = OCCUPIED;
		position = pos;
		goal = g;

		direction = normalize(make_float2(goal.x - position.x, goal.y - position.y));
		velocity = make_float2(direction.x * SPEED, direction.y * SPEED);
	}

	Person(Person* p)
	{
		state = RESERVED;
		position = p->position;
		goal = p->goal;
		direction = p->direction;
		velocity = p->velocity;
	}

	__host__ __device__ void updateVelocity(float2 newVelocity)
	{
		velocity = newVelocity;
	}

	__host__ __device__ void updateVelocity(float newX, float newY)
	{
		velocity = make_float2(newX, newY);
	}
};

// Simplified version of Person struct used for visualization
struct PersonVisuals
{
	float2 position = make_float2(0, 0);
	float2 direction = make_float2(0, 0);

	PersonVisuals(float2 pos, float2 dir)
	{
		position = pos;
		direction = dir;
	}
};