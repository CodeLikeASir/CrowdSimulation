#pragma once
#include "SF_Sequential.h"
/*
const static int cellsPerAxis = 12;
const static int totalCells = cellsPerAxis * cellsPerAxis;
const static int cellSize = 4;
const static int maxOccupation = 32;
const float speed = 1.f;
const float minDist = 0.05f;
const short safezone = 5;
*/
const float speed = 1.f;

const static short FREE = 1;
const static short OCCUPIED = 2;
const static short RESERVED = 3;
const static  short TRAVERSING = 4;
const static short LEAVING = 5;

inline float2 normalizeSF(const float2 val)
{
	const float mag = sqrtf(val.x * val.x + val.y * val.y);
	return make_float2(val.x / mag, val.y / mag);
}

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
		
		direction = normalizeSF(make_float2(goal.x - position.x, goal.y - position.y));
		velocity = make_float2(direction.x * speed, direction.y * speed);
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

	/*
	std::string float2_to_string()
	{
		if (state == FREE)
			return "FREE";
			
		return "Hi. I'm currently at " + float2ToString(position) + " and moving to " + float2ToString(goal) + ". " +
			"My velocity is " + float2ToString(velocity) + " and my direction is " + float2ToString(direction) + ".";
	}
	*/
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

class SocialForce
{
public:
	
};
