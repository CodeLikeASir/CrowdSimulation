// ReSharper disable CppLocalVariableMayBeConst
#include "SF_Sequential.h"
#include <vector>

ushort2 SF_Sequential::getRandomUShort2(short min, short max)
{
	short x = rand() % (max - min + 1) + min;
	short y = rand() % (max - min + 1) + min;
	return make_ushort2(x, y);
}

short2 SF_Sequential::getRandomShort2(short min, short max)
{
	short x = rand() % (max - min + 1) + min;
	short y = rand() % (max - min + 1) + min;
	return make_short2(x, y);
}

short2 SF_Sequential::normalize(short2 vector)
{
	float magnitude = vector.x + vector.y;
	vector.x /= magnitude;
	vector.y /= magnitude;
	return vector;
}

short2 SF_Sequential::multiply(short2 vec, float scalar)
{
	vec.x *= scalar;
	vec.y *= scalar;
	return vec;
}

short2 operator*(float lhs, const short2& rhs)
{
	return make_short2(lhs * rhs.x, lhs * rhs.y);
}

short2 SF_Sequential::calculate_ve(float v, short2 e)
{
	return v * e;
}

void SF_Sequential::init()
{
	return;
	/*
	Person grid[10 * 10];
	const int size = 10;

	for (int x = 0; x < size; x++)
	{
		for (int y = 0; y < size; y++)
		{
			short2 velocity = getRandomShort2(-5, 5);
			short2 direction = velocity;
			if ((x * size + y) % 3 == 0)
			{
				grid[x * size + y] = Person{ OCCUPIED,getRandomUShort2(0,500), make_ushort2(x,y), velocity, direction };
			}
			else
			{
				grid[x * size + y] = Person{ FREE, make_ushort2(0,0), make_ushort2(x,y), make_short2(0,0), make_short2(0,0) };
			}
		}
	}

	printGrid(grid, size);
	*/
}

short2 operator-(const short2& lhs, const short2& rhs)
{
	return make_short2(lhs.x - rhs.x, lhs.x - rhs.x);
}
/*
short2&& operator/(const short2& lhs, const short2& rhs)
{
	return make_short2(lhs.x / rhs.x, lhs.x / rhs.x);
}
*/
float2 SF_Sequential::calculateSF(Person personA, Person personB)
{
	if (debug_level >= 1)
		std::cout << "Calculating sf for " << ushort2ToString(personA.position) << " from " << ushort2ToString(personB.position) << "\n";
	short v_a0 = personA.velocityMag();
	short v_b0 = personB.velocityMag();

	short2 dir_a = toShort2(udiff(personA.goal, personA.position));
	short2 e_a = divide(dir_a, magnitude(dir_a));

	short2 dir_b = toShort2(udiff(personB.goal, personB.position));
	short2 e_b = divide(dir_b, magnitude(dir_b));

	short2 e2 = multiply(epsilon, calculate_ve(v_a0, e_a)) - calculate_ve(v_b0, e_b);
	e2 = multiply(e2, magnitude(e2));
	short2 e1 = make_short2(e2.y, -e2.x);

	const short2 r_ab = toShort2(udiff(personA.position, personB.position));
	float e1_result = dot(r_ab, e1);
	e1_result *= e1_result;
	float e2_result = dot(r_ab, e2);
	e2_result *= e2_result;

	float gamma_a = dot(r_ab, e2) >= 0.f ? theta : 1 + delta * v_a0;

	//std::cout << S << " * " << euler << " ^ -(sqrt( " << e1_result << " + " << e2_result << " / " << (gamma_a * gamma_a) << ") / " << R << " )\n";
	float V_ab = S * std::powf(euler, -std::sqrtf(e1_result + e2_result / (gamma_a * gamma_a)) / R);

	//std::cout << "V_ab = " << V_ab << "\n";
	//std::cout << "r_ab = (" << r_ab.x << "|" << r_ab.y << ")\n";
	float2 f_ab = make_float2(-r_ab.x * delta, -r_ab.y * delta);

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
			std::cout << getCellPopulation(y * cellsPerAxis + x) << "   ";
		}
		std::cout << "\n  |\n";
	}
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

ushort2 SF_Sequential::add_force(ushort2 position, short2 shortForce)
{
	return make_ushort2(position.x - delta * shortForce.x, position.y - delta* shortForce.y);
}

void SF_Sequential::update_positions()
{
	bool printed = false;
	std::cout << "Starting update\n";
	// Iterate over all cells
	for (int cell = 0; cell < totalCells; cell++)
	{
		short2 cellPos = make_short2(cell % cellsPerAxis, cell / cellsPerAxis);

		// Iterate over all spaces in cell | Thread [x]
		for (int i = 0; i < maxOccupation; i++)
		{
			if (grid[cell * maxOccupation + i].state == FREE)
				continue;

			Person person = grid[cell * maxOccupation + i];
			if (debug_level >= 1)
				std::cout << "\n\nFound one in cell " << ushort2ToString(person.position) << "\n";

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

						Person other = grid[otherCell * maxOccupation + j];

						if (other.state == FREE)
							continue;

						if (debug_level >= 1)
							std::cout << "Found influencer! \n";
						forceVector = sum(forceVector, calculateSF(person, other));
					}

					int influenced = cell * maxOccupation + i;
					int neighbor = x * 3 + y;
					//forceVectors[influenced + neighbor] = forceVector;
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

			short2 oldVel = grid[cell * maxOccupation + i].velocity;
			if (debug_level >= 1)
				std::cout << "Total force = " << floatToString(totalForce) << "\n";
			
			short2 shortForce = make_short2(totalForce.x, totalForce.y);

			if(!printed)
				std::cout << "Moved from (" << person.position.x << "|" << person.position.y << ") ";
			
			//person.position = make_ushort2(person.position.x - shortForce.x, person.position.y - shortForce.y); //sum(person.position, shortForce);
			person.position = add_force(person.position, shortForce);
			
			grid[cell * maxOccupation + i] = person;
			
			if(!printed)
			{
				std::cout << "to (" << grid[cell * maxOccupation + i].position.x << "|" << grid[cell * maxOccupation + i].position.y << ")\n";
				//printed = true;
			}
		}
	}
}

int SF_Sequential::getCellIndex(int x, int y)
{
	return y / cellSize * cellsPerAxis + x / cellSize;
}

void SF_Sequential::calculateCellForce(int cellIndex, int posX, int posY)
{
	float2 total_force = make_float2(0.f, 0.f);
	const Person current = grid[toIndex(posX, posY)];

	int firstX = cellIndex % cellsPerAxis * cellSize;
	int firstY = cellIndex / cellsPerAxis * cellSize;

	for (int x = firstX; x < firstX + cellSize - 1; x++)
	{
		for (int y = firstY; y < firstY + cellSize - 1; y++)
		{
			if (x == posX && y == posY)
			{
				if (debug_level >= 1)
					std::cout << "Skipping (" << x << "|" << y << ") because that's me!\n";
				continue;
			}

			Person person = grid[toIndex(x, y)];

			if (person.state == FREE)
				continue;

			if (debug_level >= 1)
				std::cout << "Checking (" << x << "|" << y << ")\n";
			float2 new_force = calculateSF(current, person);
			total_force = make_float2(total_force.x + new_force.x, total_force.y + new_force.y);
		}
	}

	if (debug_level >= 1)
		std::cout << "Total force for (" << posX << "|" << posY << ") = (" << total_force.x << "|" << total_force.y << ")\n";
}

int SF_Sequential::toIndex(int x, int y)
{
	int cellX = x / cellSize;
	int cellY = y / cellSize;
	return cellX + cellY * cellsPerAxis;
}

Person* SF_Sequential::init_test1()
{
	for (Person& i : grid)
	{
		i = Person();
	}

	addToGrid(Person(make_ushort2(2, 2), make_ushort2(3, 13)));
	addToGrid(Person(make_ushort2(3, 1), make_ushort2(8, 1)));
	addToGrid(Person(make_ushort2(14, 6), make_ushort2(6, 9)));
	addToGrid(Person(make_ushort2(18, 11), make_ushort2(10, 6)));
	addToGrid(Person(make_ushort2(7, 12), make_ushort2(13, 12)));
	addToGrid(Person(make_ushort2(5, 14), make_ushort2(14, 11)));
	addToGrid(Person(make_ushort2(17, 21), make_ushort2(7, 7)));

	printGrid();

	return grid;
}

void SF_Sequential::simulate(int steps)
{
	init_test1();

	for (int i = 0; i < steps; i++)
	{
		update_positions();
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
			pv.direction = simToGL(p.velocity);
			persons.push_back(pv);
		}
	}

	return persons;
}

float2 SF_Sequential::simToGL(ushort2 pos)
{
	float maxVal = cellsPerAxis * cellSize;
	float xPos = pos.x / maxVal * 2.f - 1.f;
	float yPos = pos.y / maxVal * 2.f - 1.f;

	return make_float2(xPos, yPos);
}

float2 SF_Sequential::simToGL(short2 pos)
{
	float maxVal = cellsPerAxis * cellSize;
	float xPos = pos.x / maxVal * 2.f - 1.f;
	float yPos = pos.y / maxVal * 2.f - 1.f;

	return make_float2(xPos, yPos);
}