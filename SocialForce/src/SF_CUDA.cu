#include "SF_CUDA.cuh"

Person* grid;

Person* deviceGrid;
int* debugDevice;
int* debugHost;

// 1 block = 1 cell
dim3 blocksPerGrid(CELLS_PER_AXIS, CELLS_PER_AXIS);

// 32 Threads per block/cell, 3x3 for main cell + neighbors
dim3 threadsPerBlock(MAX_OCCUPATION, 3, 3);

// function to add the elements of two arrays
__global__ void debug(Person* grid_ptr, int* debugVal)
{
	for (int i = 0; i < SPAWNED_ACTORS; i++)
	{
		grid_ptr[i].state = 2;
	}

	//atomicAdd(debugVal, 1);
	atomicMax(debugVal, blockIdx.x * 10 + blockIdx.y);

	//grid_ptr[0].velocity = calculateSF(&grid_ptr[0], &grid_ptr[1]);
}

__device__ float2 calculateSF(Person* personA, Person* personB)
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

	float V_ab = 1.0f * std::powf(EULER, -std::sqrtf(e1_result + e2_result / (gamma_a * gamma_a)) / R);

	float2 f_ab = make_float2(-r_ab.x * V_ab, -r_ab.y * V_ab);

	return f_ab;
}

__global__ void calculateCellForce(Person* device_grid, int* debugVal)
{
	__shared__ float2 totalForces[MAX_OCCUPATION][9];

	short2 cellAPos = make_short2(blockIdx.x, blockIdx.y);
	short cellA = cellPosToCell(cellAPos);

	Person* personA = &device_grid[cellA * MAX_OCCUPATION + threadIdx.x];

	short2 cellBPos = make_short2(cellAPos.x - 1 + threadIdx.y, cellAPos.y - 1 + threadIdx.z);
	short cellB = cellPosToCell(cellBPos);
	float2 forceVector = make_float2(0.f, 0.f);

	if (!(cellB < 0 || cellB >= CELLS_PER_AXIS * CELLS_PER_AXIS))
	{
		// Iterate over space in neighbor cell
		for (int i = 0; i < MAX_OCCUPATION; i++)
		{
			// Ignore yourself
			if (threadIdx.y == 1 && threadIdx.z == 1 && threadIdx.x % MAX_OCCUPATION == i)
				continue;

			Person* other = &device_grid[cellB * MAX_OCCUPATION + i];

			if (other->state == FREE)
				continue;

			forceVector = forceVector + calculateSF(personA, other);
		}
	}

	totalForces[threadIdx.x][threadIdx.y + threadIdx.z * 3] = forceVector;

	__syncthreads();

	if (threadIdx.y == 0 && threadIdx.z == 0)
	{
		float2 resultForce = make_float2(0.f, 0.f);
		for (int i = 0; i < 9; i++)
		{
			resultForce = resultForce + totalForces[threadIdx.x][i];
		}

		personA->velocity = make_float2(personA->velocity.x - resultForce.x, personA->velocity.y - resultForce.y);

		float2 newPos = personA->position + personA->velocity;

		// Check if person moves to other cell
		int oldCell = posToCell(personA->position.x, personA->position.y);
		int newCell = posToCell(newPos.x, newPos.y);

		if (oldCell != newCell)
		{
			bool cellChanged = false;
			//bool moveSuccessful = reserveSpace(device_grid, newCell, cellA * MAX_OCCUPATION + threadIdx.x);

			// Look for space in new cell
			for (int i = newCell * MAX_OCCUPATION; i < (newCell + 1) * MAX_OCCUPATION; i++)
			{
				if (atomicCAS((int*)&device_grid[i].state, FREE, RESERVED) == FREE)
				{
					device_grid[cellA * MAX_OCCUPATION + threadIdx.x].state = LEAVING;

					device_grid[i] = Person(device_grid[cellA * MAX_OCCUPATION + threadIdx.x]);
					device_grid[i].velocity = device_grid[cellA * MAX_OCCUPATION + threadIdx.x].velocity;
					device_grid[i].state = RESERVED;

					cellChanged = true;
					break;
				}
			}

			if (!cellChanged)
			{
				personA->velocity = make_float2(0.f, 0.f);
			}
		}
	}
}

__global__ void completeMove(Person* device_grid, int* debugVal)
{
	short2 cellAPos = make_short2(blockIdx.x, blockIdx.y);
	short cellA = cellPosToCell(cellAPos);

	for (int i = 0; i < MAX_OCCUPATION; i++)
	{
		Person* personA = &device_grid[cellA * MAX_OCCUPATION + i];

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

		if (magnitude(goalDir) < minDist)
		{
			personA->goal = personA->position; //make_float2(0.f, 0.f); //getRandomPos();
			personA->velocity = make_float2(0.f, 0.f);
			personA->direction = make_float2(0.f, 0.f);
		}
		else
		{
			goalDir = normalize(goalDir);
			personA->direction = goalDir;
			personA->velocity = goalDir * speed;
		}
	}

	atomicAdd(debugVal, 1);
}

__device__ bool reserveSpace(int newCell, int oldIndex)
{
	bool cellChanged = false;



	return cellChanged;
}

__device__ int posToCell(int x, int y)
{
	int cellX = x / CELL_SIZE;
	int cellY = y / CELL_SIZE;
	return cellX + cellY * CELLS_PER_AXIS;
}

int toIndexH(int x, int y)
{
	int cellX = x / CELL_SIZE;
	int cellY = y / CELL_SIZE;
	return cellX + cellY * CELLS_PER_AXIS;
}

bool addToGrid(Person p)
{
	int cell = toIndexH(p.position.x, p.position.y);
	bool placed = false;

	for (int i = 0; i < MAX_OCCUPATION; i++)
	{
		int index = cell * MAX_OCCUPATION + i;

		if (grid[index].state != STATE_FREE)
			continue;

		grid[index] = Person(p);
		placed = true;
		break;
	}

	return placed;
}

void init()
{
	grid = (Person*)malloc(sizeof(Person) * TOTAL_SPACES);
	for (int i = 0; i < TOTAL_SPACES; i++)
	{
		grid[i] = Person();
	}
	/*
	// Generate random persons here!
	int actorsPerAxis = sqrtf(SPAWNED_ACTORS) - 1;
	int spacing = CELLS_PER_AXIS * CELL_SIZE / actorsPerAxis;
	for(int x = 1; x < actorsPerAxis; x++)
	{
		for(int y = 1; y < actorsPerAxis; y++)
		{
			float2 spawnPos = make_float2(x * spacing, y * spacing);
			addToGrid(Person(spawnPos, getRandomPos()));
			std::cout << "Spawned at (" << spawnPos.x << "|" << spawnPos.y << ")\n";
		}
	}
	*/
	
	//grid[0] = ;
	//grid[1] = Person(make_float2(10, 10), make_float2(1, 10));
	//grid[2] = Person(make_float2(23, 23), make_float2(5, 5));
	
	addToGrid(Person(make_float2(1, 1), make_float2(10, 4)));
	addToGrid(Person(make_float2(10, 10), make_float2(1, 10)));
	addToGrid(Person(make_float2(23, 23), make_float2(5, 5)));

	cudaMalloc((void**)&deviceGrid, TOTAL_SPACES * sizeof(Person));
	cudaMemcpy(deviceGrid, grid, TOTAL_SPACES * sizeof(Person), cudaMemcpyHostToDevice);

	int temp = 0;
	debugHost = &temp;

	debugDevice = 0;
	cudaMalloc((void**)&debugDevice, sizeof(int));
	cudaMemcpy(debugDevice, debugHost, sizeof(int), cudaMemcpyHostToDevice);
}

void close()
{
	// Free memory
	delete[] grid;
	delete debugHost;

	cudaFree(deviceGrid);
	cudaFree(debugDevice);
}

int simulate()
{
	calculateCellForce << < blocksPerGrid, threadsPerBlock >> > (deviceGrid, debugDevice);
	cudaDeviceSynchronize();

	completeMove << < blocksPerGrid, 1 >> > (deviceGrid, debugDevice);
	cudaDeviceSynchronize();

	cudaMemcpy(grid, deviceGrid, TOTAL_SPACES * sizeof(Person), cudaMemcpyDeviceToHost);
	cudaMemcpy(debugHost, debugDevice, sizeof(int), cudaMemcpyDeviceToHost);

	//std::cout << "\n\n!!!Important: " << *debugHost << " !!!\n\n\n";

	for (int i = 0; i < TOTAL_SPACES; i++)
	{
		if (magnitudeH(grid[i].position) > 0.001f && false)
			std::cout << i << ": " << grid[i].position.x << "|" << grid[i].position.y << "\n";
	}

	//std::cout << "SF: " << grid[0].velocity.x << "|" << grid[0].velocity.y << "\n";

	return 0;
}

std::vector<PersonVisuals> convertToVisual(bool debugPrint)
{
	std::vector<PersonVisuals> persons;

	for (int i = 0; i < TOTAL_SPACES; i++)
	{
		Person& p = grid[i];
		if (p.state != FREE)
		{
			persons.push_back(PersonVisuals(simToGL(p.position), p.direction));
		}
	}

	return persons;
}

float2 simToGL(float2 pos)
{
	float maxVal = CELLS_PER_AXIS * CELL_SIZE;
	float xPos = pos.x / maxVal * 2.f - 1.f;
	float yPos = (pos.y / maxVal * 2.f - 1.f) * -1.f;

	return make_float2(xPos, yPos);
}