// ReSharper disable CppLocalVariableMayBeConst
#include "SF_Sequential.h"

#include <iostream>

#include "PersonStructs.h"

namespace SF_Sequential
{
	static Person* cells;

	int getCellPopulation(uint3 cell)
	{
		int count = 0;
		int firstIndex = getFirstCellIndex(cell);
		for (int i = firstIndex; i < firstIndex + MAX_OCCUPATION; i++)
		{
			if (cells[i].state == OCCUPIED)
			{
				count++;
			}
		}

		return count;
	}

	void init()
	{
		cells = static_cast<Person*>(malloc(sizeof(Person) * TOTAL_SPACES));
		for (int i = 0; i < TOTAL_SPACES; i++)
		{
			cells[i] = Person();
		}

		int totallySpawned = 0;
		int remainingSpawns = SPAWNED_ACTORS;

		int spawnsPerRow = ceil(sqrtf(SPAWNED_ACTORS));
		float spacing = CELLS_PER_AXIS * CELL_SIZE / spawnsPerRow;
		for (int x = 0; x < spawnsPerRow; x++)
		{
			for (int y = 0; y < spawnsPerRow; y++)
			{
				float2 spawnPos = make_float2(x * spacing, y * spacing);
				add_to_grid(Person(spawnPos, getRandomPos()));

				totallySpawned++;
				if (--remainingSpawns <= 0)
					goto endspawn;
			}
		}

	endspawn:
		std::cout << "Spawned " << totallySpawned << " people.\n";
	}

	float2 calculateSF(Person* personA, Person* personB)
	{
		float v_a0 = magnitude(personA->velocity);
		float v_b0 = magnitude(personB->velocity);

		if (v_a0 * v_a0 < 0.001f || v_b0 * v_b0 < 0.001f)
		{
			return make_float2(0.f, 0.f);
		}

		float2 dir_a = personA->goal - personA->position;
		float2 e_a = dir_a / magnitude(dir_a);

		float2 dir_b = personB->goal - personB->position;
		float2 e_b = dir_b / magnitude(dir_b);

		float2 e2 = EPSILON * v_a0 * e_a - v_b0 * e_b;
		e2 = normalize(e2);
		float2 e1 = make_float2(e2.y, -e2.x);

		const float2 r_ab = personA->position - personB->position;
		float e1_result = dot(r_ab, e1);
		e1_result *= e1_result;
		float e2_result = dot(r_ab, e2);
		e2_result *= e2_result;

		float gamma_a = dot(r_ab, e2) >= 0.f ? THETA : 1 + DELTA * v_a0;

		float V_ab = S * std::powf(EULER, -std::sqrtf(e1_result + e2_result / (gamma_a * gamma_a)) / R);

		float2 f_ab = make_float2(-r_ab.x * V_ab, -r_ab.y * V_ab);

		return f_ab;
	}

	void add_to_grid(const Person& p)
	{
		float2 cell_coords = p.position / CELL_SIZE;
		int cell = cell_coords.x + cell_coords.y * CELLS_PER_AXIS;

		for (int i = 0; i < MAX_OCCUPATION; i++)
		{
			int index = cell * MAX_OCCUPATION + i;

			if (cells[index].state != FREE)
				continue;

			cells[index] = Person(p);
			break;
		}
	}

	void simulate()
	{
		for (int cellX = 0; cellX < CELLS_PER_AXIS; cellX++)
		{
			for (int cellY = 0; cellY < CELLS_PER_AXIS; cellY++)
			{
				// pBlockIdx simulates behavior of CUDA blockIdx
				uint3 pBlockIdx = make_uint3(cellX, cellY, 0);
				
				short cellA = cellPosToIndex(cellX, cellY);
				int cellPop = getCellPopulation(pBlockIdx);

				// threadX/Y/Z and pThreadIdx simulate behavior of CUDA threadIdx
				for (int threadX = 0; threadX <= 32; threadX++)
				{
					Person* personA = &cells[cellA * MAX_OCCUPATION + threadX];
					if (personA->state == FREE)
						continue;

					float2 total_force = make_float2(0.f, 0.f);
					for (int threadY = 0; threadY <= 2; threadY++)
					{
						for (int threadZ = 0; threadZ <= 2; threadZ++)
						{
							uint3 pThreadIdx = make_uint3(threadX, threadY, threadZ);
							total_force = total_force + calculateCellForce(cellPop, pBlockIdx, pThreadIdx);
						}
					}

					update_position(personA, total_force, cellA, threadX);
				}
			}
		}

		// 2nd CUDA kernel starts here
		for (int cellX = 0; cellX < CELLS_PER_AXIS; cellX++)
		{
			for (int cellY = 0; cellY < CELLS_PER_AXIS; cellY++)
			{
				uint3 pBlockIdx = make_uint3(cellX, cellY, 0);
				completeMove(pBlockIdx);
			}
		}
	}

	float2 calculateCellForce(int cellAPop, uint3 pBlockIdx, uint3 pThreadIdx)
	{
		short2 cellAPos = make_short2(pBlockIdx.x, pBlockIdx.y);
		short cellA = cellPosToIndex(cellAPos);

		Person* personA = &cells[cellA * MAX_OCCUPATION + pThreadIdx.x];

		short2 cellBPos = make_short2(cellAPos.x - 1 + pThreadIdx.y, cellAPos.y - 1 + pThreadIdx.z);

		short cellB = cellPosToIndex(cellBPos);
		float2 forceVector = make_float2(0.f, 0.f);

		if (!(cellB < 0 || cellB >= CELLS_PER_AXIS * CELLS_PER_AXIS))
		{
			// People in analyzed cell
			int blockppl = 0;
			
			// Iterate over spaces in neighbor cell
			for (int i = 0; i < MAX_OCCUPATION; i++)
			{
				// Ignore yourself
				if (pThreadIdx.y == 1 && pThreadIdx.z == 1 && pThreadIdx.x % MAX_OCCUPATION == i)
					continue;

				Person* other = &cells[cellB * MAX_OCCUPATION + i];

				if (other->state == FREE)
					continue;

				forceVector = forceVector + calculateSF(personA, other);
				blockppl++;
			}

			// People in main/influenced cell
			int ppl = cellAPop;

			if ((pThreadIdx.y != 1 || pThreadIdx.z != 1) && (blockppl > 20 || ppl > 26))
			{
				forceVector.x -= (pThreadIdx.y - 1) * (blockppl - 20) * AVOIDANCE_FORCE;
				forceVector.y -= (pThreadIdx.z - 1) * (blockppl - 20) * AVOIDANCE_FORCE;
			}
		}
		
		return forceVector;
	}

	void update_position(Person personA, float2 total_force, int cellIndex, int threadX)
	{
		personA.velocity = make_float2(personA.velocity.x - total_force.x, personA.velocity.y - total_force.y);

		float2 newPos = personA.position + personA.velocity * DELTA;

		// Check if person moves to other cell
		int oldCell = personPosToCellIndex(personA.position.x, personA.position.y);
		int newCell = personPosToCellIndex(newPos.x, newPos.y);

		if (oldCell != newCell)
		{
			bool cellChanged = false;

			if (newCell >= 0 && newCell < TOTAL_CELLS)
			{
				// Look for space in new cell
				for (int i = newCell * MAX_OCCUPATION; i < (newCell + 1) * MAX_OCCUPATION; i++)
				{
					if (cells[i].state == FREE)
					{
						cells[i].state = RESERVED;

						cells[cellIndex * MAX_OCCUPATION + threadX].state = LEAVING;

						cells[i] = Person(cells[cellIndex * MAX_OCCUPATION + threadX]);
						cells[i].state = RESERVED;

						cellChanged = true;
						break;
					}
				}
			}

			if (!cellChanged)
			{
				personA.velocity = make_float2(0.f, 0.f);
			}
		}
	}

	void completeMove(uint3 pBlockIdx)
	{
		short2 cellAPos = make_short2(pBlockIdx.x, pBlockIdx.y);
		short cellA = cellPosToIndex(cellAPos);

		for (int i = 0; i < MAX_OCCUPATION; i++)
		{
			Person* personA = &cells[cellA * MAX_OCCUPATION + i];

			if (personA->state == FREE)
				continue;

			if (personA->state == LEAVING)
			{
				personA->state = FREE;
				personA->velocity = make_float2(0.f, 0.f);
				continue;
			}

			if (personA->state == RESERVED)
			{
				personA->state = OCCUPIED;
			}

			personA->position = personA->position + personA->velocity * DELTA;

			float2 goalDir = make_float2(
				personA->goal.x - personA->position.x,
				personA->goal.y - personA->position.y);

			if (magnitude(goalDir) < MIN_DIST)
			{
				personA->goal = personA->position; //make_float2(0.f, 0.f); //getRandomPos();
				personA->velocity = make_float2(0.f, 0.f);
				personA->direction = make_float2(0.f, 0.f);
			}
			else
			{
				goalDir = normalize(goalDir);
				personA->direction = goalDir;
				personA->velocity = goalDir * SPEED;
			}
		}
	}

	std::vector<PersonVisuals> convertToVisual()
	{
		std::vector<PersonVisuals> persons;
		int remainingDraws = DRAWN_ACTORS > 0 ? DRAWN_ACTORS : SPAWNED_ACTORS;

		for (int i = 0; i < TOTAL_SPACES; i++)
		{
			Person& p = cells[i];
			if (p.state != FREE)
			{
				float2 dir = p.direction;
				dir.y = -dir.y;
				persons.emplace_back(simCoordToGL(p.position), dir);

				if (--remainingDraws <= 0) break;
			}
		}

		return persons;
	}
}