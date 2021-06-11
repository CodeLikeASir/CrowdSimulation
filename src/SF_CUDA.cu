#include "SF_CUDA.cuh"
#include "Math_Helper.cuh"

namespace SF_CUDA
{

	// Host variables
	Person* cells;

	// Device variables
	Person* deviceCells;
	int* debugDevice;
	int* debugHost;

	// 1 block = 1 cell
	dim3 blocksPerGrid(CELLS_PER_AXIS, CELLS_PER_AXIS, 1);

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

		float V_ab = S * std::powf(EULER, -std::sqrtf(e1_result + e2_result / (gamma_a * gamma_a)) / R);

		float2 f_ab = make_float2(-r_ab.x * V_ab, -r_ab.y * V_ab);

		return f_ab;
	}

	__global__ void calculateCellForce(Person* device_grid, int* debugVal)
	{
		__shared__ float2 totalForces[MAX_OCCUPATION][9];

		short2 cellAPos = make_short2(blockIdx.x, blockIdx.y);
		int cellA = cellPosToCell(cellAPos);

		Person* personA = &device_grid[cellA * MAX_OCCUPATION + threadIdx.x];
		if (personA->state == FREE)
			return;

		short2 cellBPos = make_short2(cellAPos.x - 1 + threadIdx.y, cellAPos.y - 1 + threadIdx.z);

		int cellB = cellPosToCell(cellBPos);
		float2 forceVector = make_float2(0.f, 0.f);

		if (!(cellB < 0 || cellB >= CELLS_PER_AXIS * CELLS_PER_AXIS))
		{
			// People in analyzed cell
			int blockppl = 0;
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
				blockppl++;
			}

			// People in main/influenced cell
			int ppl = maskToInt(__ballot_sync(0xFFFFFFFF, personA->state == OCCUPIED));

			if ((threadIdx.y != 1 || threadIdx.z != 1) && (blockppl > 20 || ppl > 26))
			{
				forceVector.x -= (threadIdx.y - 1) * (blockppl - 20) * AVOIDANCE_FORCE;
				forceVector.y -= (threadIdx.z - 1) * (blockppl - 20) * AVOIDANCE_FORCE;
			}
		}

		totalForces[threadIdx.x][threadIdx.y + threadIdx.z * 3] = forceVector;

		__syncthreads();

		if (threadIdx.y == 1 && threadIdx.z == 1)
		{
			float2 resultForce = make_float2(0.f, 0.f);
			for (int i = 0; i < 9; i++)
			{
				if (float2_isnan(totalForces[threadIdx.x][i]))
					continue;

				resultForce = resultForce + totalForces[threadIdx.x][i];
			}

			personA->velocity = make_float2(personA->velocity.x - resultForce.x, personA->velocity.y - resultForce.y);

			float2 newPos = personA->position + personA->velocity * DELTA;

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
					if (atomicCAS(&device_grid[i].state, FREE, RESERVED) == FREE)
					{
						device_grid[cellA * MAX_OCCUPATION + threadIdx.x].state = LEAVING;

						device_grid[i] = Person(device_grid[cellA * MAX_OCCUPATION + threadIdx.x]);
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
		int cell = cellPosToCell(blockIdx.x, blockIdx.y);
		Person* person = &device_grid[cell * MAX_OCCUPATION + threadIdx.x];

		if (person->state == FREE)
			return;

		if (person->state == LEAVING)
		{
			person->state = FREE;
			person->velocity = make_float2(0.f, 0.f);
			return;
		}

		if (person->state == RESERVED)
		{
			person->state = OCCUPIED;
		}

		atomicAdd(debugVal, 1);

		person->position = person->position + person->velocity * DELTA;

		float2 goalDir = make_float2(
			person->goal.x - person->position.x,
			person->goal.y - person->position.y);

		if (magnitude(goalDir) < MIN_DIST)
		{
			person->goal = person->position;
			person->velocity = make_float2(0.f, 0.f);
			person->direction = make_float2(0.f, 0.f);
		}
		else
		{
			goalDir = normalize(goalDir);
			person->direction = goalDir;
			person->velocity = goalDir * SPEED;
		}
	}

	__device__ int posToCell(int x, int y)
	{
		int cellX = x / CELL_SIZE;
		int cellY = y / CELL_SIZE;
		return cellX + cellY * CELLS_PER_AXIS;
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
		
		cudaMalloc((void**)&deviceCells, TOTAL_SPACES * sizeof(Person));
		cudaMemcpy(deviceCells, cells, TOTAL_SPACES * sizeof(Person), cudaMemcpyHostToDevice);

		int temp = 0;
		debugHost = &temp;

		debugDevice = nullptr;
		cudaMalloc((void**)&debugDevice, sizeof(int));
		cudaMemcpy(debugDevice, debugHost, sizeof(int), cudaMemcpyHostToDevice);
	}

	void close()
	{
		// Free memory
		//free(cells);
		//delete debugHost;

		cudaFree(deviceCells);
		cudaFree(debugDevice);
	}

	void simulate()
	{
		*debugHost = 0;
		cudaMemcpy(debugDevice, debugHost, sizeof(int), cudaMemcpyHostToDevice);
		calculateCellForce << < blocksPerGrid, threadsPerBlock >> > (deviceCells, debugDevice);
		cudaDeviceSynchronize();
		auto error = cudaDeviceSynchronize();
		if (error)
		{
			std::cout << cudaGetErrorName << ": " << cudaGetErrorString(error) << "\n";
		}

		completeMove << < blocksPerGrid, MAX_OCCUPATION >> > (deviceCells, debugDevice);
		cudaDeviceSynchronize();

		cudaMemcpy(cells, deviceCells, TOTAL_SPACES * sizeof(Person), cudaMemcpyDeviceToHost);
		cudaMemcpy(debugHost, debugDevice, sizeof(int), cudaMemcpyDeviceToHost);
	}

	std::vector<PersonVisuals> convertToVisual(bool debugPrint)
	{
		std::vector<PersonVisuals> persons;
		int addedActors = 0;

		for (int i = 0; i < TOTAL_SPACES; i++)
		{
			Person& p = cells[i];
			if (p.state != FREE)
			{
				float2 dir = p.direction;
				dir.y = -dir.y;
				persons.push_back(PersonVisuals(simToGL(p.position), dir));

				if (++addedActors >= DRAWN_ACTORS)
				{
					break;
				}
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
}