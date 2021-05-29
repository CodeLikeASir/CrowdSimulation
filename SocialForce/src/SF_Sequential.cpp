// ReSharper disable CppLocalVariableMayBeConst
#include "SF_Sequential.h"
#include <vector>
/*
float2 SF_Sequential::getRandomfloat2(short min, short max)
{
	short x = rand() % (max - min + 1) + min;
	short y = rand() % (max - min + 1) + min;
	return make_float2(x, y);
}

float2 SF_Sequential::normalize(float2 vector)
{
	float magnitude = sqrtf(vector.x * vector.x + vector.y * vector.y);
	vector.x /= magnitude;
	vector.y /= magnitude;
	return vector;
}

float2 SF_Sequential::multiply(float2 vec, float scalar)
{
	vec.x *= scalar;
	vec.y *= scalar;
	return vec;
}

float2 operator*(float lhs, const float2& rhs)
{
	return make_float2(lhs * rhs.x, lhs * rhs.y);
}

float2 SF_Sequential::calculate_ve(float v, float2 e)
{
	return v * e;
}

float2 operator-(const float2& lhs, const float2& rhs)
{
	return make_float2(lhs.x - rhs.x, lhs.x - rhs.x);
}

float2 SF_Sequential::calculateSF(Person* personA, Person* personB)
{
	if (debug_level >= 1)
		std::cout << "Calculating sf for " << float2ToString(personA->position) << " from " << float2ToString(personB->position) << "\n";
	float v_a0 = personA->velocityMag();
	float v_b0 = personB->velocityMag();

	if(v_a0 * v_a0 < 0.001f ||v_b0 * v_b0 < 0.001f)
	{
		return make_float2(0.f, 0.f);
	}

	float2 dir_a = tofloat2(udiff(personA->goal, personA->position));
	float2 e_a = divide(dir_a, magnitude(dir_a));

	float2 dir_b = tofloat2(udiff(personB->goal, personB->position));
	float2 e_b = divide(dir_b, magnitude(dir_b));

	float2 e2 = multiply(epsilon, calculate_ve(v_a0, e_a)) - calculate_ve(v_b0, e_b);
	e2 = normalize(e2);
	float2 e1 = make_float2(e2.y, -e2.x);

	const float2 r_ab = tofloat2(udiff(personA->position, personB->position));
	float e1_result = dot(r_ab, e1);
	e1_result *= e1_result;
	float e2_result = dot(r_ab, e2);
	e2_result *= e2_result;

	float gamma_a = dot(r_ab, e2) >= 0.f ? theta : 1 + delta * v_a0;

	float test = -std::sqrtf(e1_result + e2_result / (gamma_a * gamma_a)) / R;
	float V_ab = S * std::powf(euler, -std::sqrtf(e1_result + e2_result / (gamma_a * gamma_a)) / R);
	
	float2 f_ab = make_float2(-r_ab.x * V_ab, -r_ab.y * V_ab);

	if(isnan(f_ab.x) || isnan(f_ab.y))
	{
		std::cout << "blub";
	}

	if (debug_level >= 1)
		std::cout << "F_ab = " << floatToString(f_ab) << "\n";

	return f_ab;
}

void SF_Sequential::printGrid()
{
	std::cout << "\nCurrent cell populations:\n\n";
	std::cout << "    0   1   2   3   4   5\n__|_______________________\n";
	for (int x = 0; x < cellsPerAxis; x++)
	{
		std::cout << x << " | ";
		for (int y = 0; y < cellsPerAxis; y++)
		{
			std::cout << getCellPopulation(x * cellsPerAxis + y) << "   ";
		}
		std::cout << "\n  |\n";
	}
}

bool SF_Sequential::addToGrid(Person* p)
{
	int cell = toIndex(p->position.x, p->position.y);
	bool placed = false;

	for (int i = 0; i < maxOccupation; i++)
	{
		int index = cell * maxOccupation + i;
		if (grid[index].state != FREE)
			continue;

		grid[index] = *p;
		placed = true;
		break;
	}

	return placed;
}

bool SF_Sequential::addToGrid(Person p)
{
	int cell = toIndex(p.position.x, p.position.y);
	bool placed = false;

	for (int i = 0; i < maxOccupation; i++)
	{
		int index = cell * maxOccupation + i;
		if (grid[index].state != FREE)
			continue;

		grid[index] = p;
		placed = true;
		break;
	}

	return placed;
}

float2 SF_Sequential::add_force(Person* p, float2 shortForce)
{
	//std::cout << "Force = (" << p->velocity.x << "|" << p->velocity.y << ") - (" << shortForce.x << "|" << shortForce.y << ")\n";
	//std::cout << "Moving ( " << (p->velocity.x - shortForce.x) * delta << "|" << (p->velocity.y - shortForce.y) * delta << ")\n";
	float xPos = p->position.x + (p->velocity.x - shortForce.x) * delta;
	float yPos = p->position.y + (p->velocity.y - shortForce.y) * delta;
	return make_float2(xPos, yPos);
}

void SF_Sequential::update_positions()
{
	bool printed = false;
	
	// Iterate over all cells
	for (int cell = 0; cell < totalCells; cell++)
	{
		float2 cellPos = make_float2(cell % cellsPerAxis, cell / cellsPerAxis);

		// Iterate over all spaces in cell | Thread [x]
		for (int i = 0; i < maxOccupation; i++)
		{
			if (grid[cell * maxOccupation + i].state == FREE)
				continue;

			if(grid[cell * maxOccupation + i].state == RESERVED || grid[cell * maxOccupation + i].state == TRAVERSING)
			{
				continue;
			}

			Person* person = &grid[cell * maxOccupation + i];
			if (debug_level >= 1)
				std::cout << "\n\nFound one in cell " << float2ToString(person->position) << "\n";

			float2 forceTemp[9];
			for (int i = 0; i < 9; i++)
			{
				forceTemp[i] = make_float2(0.f, 0.f);
			}

			// Iterate over neighbor cells | Thread [x][y]
			for (int y = 0; y <= 2; y++)
			{
				// Thread [x][y][z]
				for (int x = 0; x <= 2; x++)
				{
					short currX = cellPos.x - 1 + x;
					short currY = cellPos.y - 1 + y;

					// Skip invalid cells
					if (currX < 0 || currY < 0 || currX >= cellsPerAxis || currY >= cellsPerAxis) continue;

					int otherCell = cellPosToIndex(currX, currY);

					float2 forceVector = make_float2(0.f, 0.f);

					// Iterate over space in neighbor cell
					for (int j = 0; j < maxOccupation; j++)
					{
						// Ignore yourself
						if (cell * maxOccupation + i == otherCell * maxOccupation + j)
							continue;

						Person* other = &grid[otherCell * maxOccupation + j];

						if (other->state == FREE)
							continue;

						if (debug_level >= 1)
							std::cout << "Found influencer! \n";
						forceVector = sum(forceVector, calculateSF(person, other));
					}
					
					int neighbor = x * 3 + y;
					//forceVectors[influenced + neighbor] = forceVector;
					//std::cout << "Force " << neighbor << ": " << forceVector.x << " | " << forceVector.y << "\n";
					forceTemp[neighbor] = forceVector;

					// When all force vectors for current individual are computed
					// !!! In CUDA this has to be synchronized first !!!
				}
			}

			float2 totalForce = make_float2(0.f, 0.f);
			for (int tf = 0; tf < 9; tf++)
			{
				float2 curr = forceTemp[tf]; //forceVectors[cell * maxOccupation + i + tf];
				if (std::abs(curr.x) > 1000 || std::abs(curr.y) > 1000)
					continue;

				totalForce = sum(totalForce, curr);
			}

			if(isnan(totalForce.x) || isnan(totalForce.y))
			{
				std::cout << "NaN for " << person->position.x << "|" << person->position.y << "\n";
				return;
			}
			
			if (debug_level >= 2)
				std::cout << "Total force = " << floatToString(totalForce) << "\n";

			if(debug_level >= 1 && !printed)
			
			if(debug_level >= 1 && !printed)
			{
				std::cout << "Moved from (" << person->position.x << "|" << person->position.y << ") " << 
				"to (" << grid[cell * maxOccupation + i].position.x << "|" << grid[cell * maxOccupation + i].position.y << ")\n";
			}
			
			float2 newPos = add_force(person, totalForce);
			
			// Check if person moves to other cell
			int oldCell = toIndex(person->position.x, person->position.y);
			int newCell = toIndex(newPos.x, newPos.y);
			
			bool moveSuccessful = true;
			
			if(oldCell != newCell)
			{
				moveSuccessful = reserveSpace(person, newCell, newPos);
				
				if(moveSuccessful)
				{
					person->state = TRAVERSING;
				}
			}
			else
			{
				updatePosition(person, newPos);
				//person->position = newPos;
			}
		}
	}
}

void SF_Sequential::updatePosition(Person* p, float2 newPos)
{
	//float2 dir = make_float2(newPos.x - p->position.x, newPos.y - p->position.y);
	p->position = newPos;

	float2 goalDir = make_float2(
						p->goal.x - p->position.x, 
						p->goal.y - p->position.y);

	if(magnitude(goalDir) < minDist)
	{
		p->goal = getRandomPos();
		goalDir = make_float2(
			p->goal.x - p->position.x,
			p->goal.y - p->position.y);
	}
	
	goalDir = normalize(goalDir);
	p->direction = goalDir;
	p->velocity = multiply(goalDir, speed);
}

float2 SF_Sequential::getRandomPos()
{
	short x = rand() % (cellSize * cellsPerAxis - safezone) + safezone;
	short y = rand() % (cellSize * cellsPerAxis - safezone) + safezone;
	return make_float2(x,y);
}

void SF_Sequential::debugerino()
{
	for(int i = 0; i < totalCells * maxOccupation; i++)
	{
		if(grid[i].state != FREE)
		{
			std::cout << i << " is " << grid[i].state << "\n";
		}
	}
}

void SF_Sequential::completeTraversal()
{
	// Iterate over all cells
	for (int cell = 0; cell < totalCells; cell++)
	{
		// Iterate over all spaces in cell | Thread [x]
		for (int i = 0; i < maxOccupation; i++)
		{
			short state = grid[cell * maxOccupation + i].state;

			//if(state != OCCUPIED && state != FREE) std::cout << "updating " << cell * maxOccupation + i << " from " << state << " to ";

			if(state == TRAVERSING)
			{
				grid[cell * maxOccupation + i].state = FREE;
			}
			else if(state == RESERVED)
			{
				grid[cell * maxOccupation + i].state = OCCUPIED;
			}
			
			//if (state != OCCUPIED && state != FREE) std::cout << grid[cell * maxOccupation + i].state << "\n";
		}
	}
}

bool SF_Sequential::reserveSpace(Person* p, int cell, float2 newPos)
{
	bool foundSpace = false;
	
	for(int i = 0; i < maxOccupation; i++)
	{
		int index = cell * maxOccupation + i;
		if (grid[index].state != FREE)
		{
			continue;
		}

		//p->position = newPos;
		updatePosition(p, newPos);
		grid[index] = Person(p);
		foundSpace = true;
		break;
	}
	
	return foundSpace;
}

int SF_Sequential::getCellIndex(int x, int y)
{
	return y / cellSize * cellsPerAxis + x / cellSize;
}

int SF_Sequential::toIndex(int x, int y)
{
	int cellX = x / cellSize;
	int cellY = y / cellSize;
	return cellX + cellY * cellsPerAxis;
}

Person* SF_Sequential::init_test1(float newDelta)
{
	for (Person& i : grid)
	{
		i = Person();
	}

	addToGrid(Person(make_float2(2, 2), make_float2(3, 13)));
	addToGrid(Person(make_float2(3, 1), make_float2(8, 1)));
	addToGrid(Person(make_float2(14, 6), make_float2(6, 9)));
	addToGrid(Person(make_float2(18, 11), make_float2(10, 6)));
	addToGrid(Person(make_float2(7, 12), make_float2(13, 12)));
	addToGrid(Person(make_float2(5, 14), make_float2(14, 11)));
	addToGrid(Person(make_float2(17, 21), make_float2(7, 7)));

	printGrid();

	delta = newDelta;

	return grid;
}

Person* SF_Sequential::init_test2(float newDelta)
{
	for (Person& i : grid)
	{
		i = Person();
	}

	addToGrid(Person(make_float2(1, 9), make_float2(10, 9)));
	addToGrid(Person(make_float2(10, 10), make_float2(1, 10)));

	printGrid();

	delta = newDelta;

	return grid;
}

Person* SF_Sequential::init_test3(float newDelta)
{
	for (Person& i : grid)
	{
		i = Person();
	}

	/*
	for(int cell = cellsPerAxis; cell < totalCells - cellsPerAxis; cell += 3)
	{
		int cellX = cell % cellsPerAxis;
		int cellY = cell / cellsPerAxis;
		
		float x = rand() % 4 + cellX * 4;
		float y = rand() % 4 + cellY * 4;

		addToGrid(Person(make_float2(x, y), make_float2(0.f, 0.f)));
	}

	for(int cell = 5; cell < 15; cell++)
	{
		int cellX = 15 % cellsPerAxis;
		int cellY = 15 / cellsPerAxis;

		float x = rand() % 4 + cellX * 4;
		float y = rand() % 4 + cellY * 4;

		addToGrid(Person(make_float2(x, y), getRandomPos()));
	}

	printGrid();

	delta = newDelta;

	return grid;
}

void SF_Sequential::simulate(int steps)
{
	for (int i = 0; i < steps; i++)
	{
		update_positions();
		completeTraversal();
	}
}

void SF_Sequential::hard_reset()
{
	for(int i = 0; i < totalCells * maxOccupation; i++)
	{
		grid[i] = Person();
	}
}

std::vector<PersonVisuals> SF_Sequential::convertToVisual(bool debugPrint)
{
	std::vector<PersonVisuals> persons;
	if(debugPrint)
		std::cout << " Starting conversion \n";

	for (auto p : grid)
	{
		if (p.state != FREE)
		{
			PersonVisuals pv;
			pv.position = simToGL(p.position);
			if(debugPrint)
				std::cout << "Position: " << pv.position.x << " | " << pv.position.y << "\n";
			pv.direction = p.direction;
			persons.push_back(pv);
		}
	}

	return persons;
}

float2 SF_Sequential::simToGL(float2 pos)
{
	float maxVal = cellsPerAxis * cellSize;
	float xPos = pos.x / maxVal * 2.f - 1.f;
	float yPos = (pos.y / maxVal * 2.f - 1.f) * -1.f;

	return make_float2(xPos, yPos);
}
*/