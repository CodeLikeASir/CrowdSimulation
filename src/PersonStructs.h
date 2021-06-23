#pragma once
#include "SF_Sequential.h"
#include "Macros.h"
#include "Math_Helper.cuh"

// Person states
const static short FREE = 1;
const static short OCCUPIED = 2;
const static short RESERVED = 3;
const static short LEAVING = 4;

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
		velocity = make_float2(p->velocity.x, p->velocity.y);
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