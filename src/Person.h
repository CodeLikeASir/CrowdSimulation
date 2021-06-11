#pragma once
#include "SF_Sequential.h"
#include "Macros.h"
#include "Math_Helper.cuh"

const static short FREE = 1;
const static short OCCUPIED = 2;
const static short RESERVED = 3;
const static  short TRAVERSING = 4;
const static short LEAVING = 5;
/*
inline float2 normalizeSF(const float2 val)
{
	const float mag = sqrtf(val.x * val.x + val.y * val.y);
	return make_float2(val.x / mag, val.y / mag);
}
*/

struct Person
{
	int state;
	float2 position;
	float2 goal;
	float2 velocity;
	float2 direction;

	Person(float2 pos, float2 g)
	{
		state = OCCUPIED;
		position = pos;
		goal = g;
		
		direction = normalize(make_float2(goal.x - position.x, goal.y - position.y));
		velocity = make_float2(direction.x * SPEED, direction.y * SPEED);
	}

	Person()
	{
		state = FREE;
		position = make_float2(0, 0);
		goal = make_float2(0, 0);
		direction = make_float2(0, 0);
		velocity = make_float2(0, 0);
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

struct PersonVisuals
{
	float2 position;
	float2 direction;

	PersonVisuals(float2 pos, float2 dir)
	{
		position = pos;
		direction = dir;
	}
};