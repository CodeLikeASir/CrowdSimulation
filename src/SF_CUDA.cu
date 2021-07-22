#include "SF_CUDA.cuh"
#include <chrono>
#include <iostream>
#include "Math_Helper.cuh"

namespace SF_CUDA
{
	// Host variables
	Person* cells;

	// Device variables
	Person* deviceCells;

	// 1 (thread) block = 1 cell
	dim3 blocksPerGrid(CELLS_PER_AXIS, CELLS_PER_AXIS, 1);
	
	// Per space: 9 threads, 1 for each influencing cell
	dim3 threadsPerBlock(MAX_OCCUPATION, 3, 3);

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

	__global__ void calculateCellForce(Person* device_grid)
	{
		// Saves forces of each cell on influenced person
		__shared__ float2 totalForces[MAX_OCCUPATION][9];

		short2 cellAPos = make_short2(blockIdx.x, blockIdx.y);
		int cellA = cellPosToIndex(cellAPos);

		// Influenced person/space
		Person* personA = &device_grid[cellA * MAX_OCCUPATION + threadIdx.x];

		// If space is empty, thread can terminate early
		if (personA->state != OCCUPIED)
			return;

		short2 cellBPos = make_short2(cellAPos.x - 1 + threadIdx.y, cellAPos.y - 1 + threadIdx.z);

		int cellB = cellPosToIndex(cellBPos);
		float2 forceVector = make_float2(0.f, 0.f);

		if (cellB >= 0 && cellB < CELLS_PER_AXIS * CELLS_PER_AXIS)
		{
			// Number of people in influencing cell, important for congestion avoidance
			int blockppl = 0;

			// Iterate over space in neighbor cell
			for (int i = 0; i < MAX_OCCUPATION; i++)
			{
				// Ignore yourself
				if (threadIdx.y == 1 && threadIdx.z == 1 && threadIdx.x % MAX_OCCUPATION == i)
					continue;

				Person* other = &device_grid[cellB * MAX_OCCUPATION + i];

				if (other->state != OCCUPIED)
					continue;

				forceVector = forceVector + calculateSF(personA, other);
				blockppl++;
			}

			// Number of people in influenced cell
			int ppl = mask_to_int(__ballot_sync(0xFFFFFFFF, personA->state == OCCUPIED));

			// Only calculate avoidance force if influencing cell =/= influenced cell
			if ((threadIdx.y != 1 || threadIdx.z != 1) && (blockppl > 20 || ppl > 26))
			{
				forceVector.x -= (threadIdx.y - 1.f) * (blockppl - 20) * AVOIDANCE_FORCE;
				forceVector.y -= (threadIdx.z - 1.f) * (blockppl - 20) * AVOIDANCE_FORCE;
			}
		}

		// Save calculated force in shared memory
		totalForces[threadIdx.x][threadIdx.y + threadIdx.z * 3] = forceVector;

		// Wait for all threads to complete calculation
		__syncthreads();

		// Only center cell sums and applies all social forces
		if (threadIdx.y == 1 && threadIdx.z == 1)
		{
			float2 resultForce = make_float2(0.f, 0.f);
			for (int i = 0; i < 9; i++)
			{
				if (float2_isnan(totalForces[threadIdx.x][i]))
					continue;

				resultForce = resultForce + totalForces[threadIdx.x][i];
			}

			personA->updateVelocity(personA->velocity - resultForce * DELTA);

			float2 newPos = personA->position + personA->velocity * DELTA;

			// Check if person moves to other cell
			int oldCell = personPosToCellIndex(personA->position.x, personA->position.y);
			int newCell = personPosToCellIndex(newPos.x, newPos.y);

			if (oldCell != newCell)
			{
				bool reservedSpace = false;

				if (newCell >= 0 && newCell < CELLS_PER_AXIS * CELLS_PER_AXIS)
				{
					// Look for space in new cell
					for (int i = newCell * MAX_OCCUPATION; i < (newCell + 1) * MAX_OCCUPATION; i++)
					{
						if (atomicCAS(&device_grid[i].state, FREE, RESERVED) == FREE)
						{
							device_grid[cellA * MAX_OCCUPATION + threadIdx.x].state = LEAVING;

							device_grid[i] = Person(device_grid[cellA * MAX_OCCUPATION + threadIdx.x]);
							device_grid[i].state = RESERVED;

							reservedSpace = true;

							break;
						}
					}
				}

				// If entry to other cell was denied, block movement
				if (!reservedSpace)
				{
					personA->velocity = make_float2(0.f, 0.f);
				}
			}
		}
	}

	__global__ void completeMove(Person* device_grid)
	{
		int cell = cellPosToIndex(blockIdx.x, blockIdx.y);
		Person* person = &device_grid[cell * MAX_OCCUPATION + threadIdx.x];

		// Terminate early if space is empty
		if (person->state == FREE)
			return;

		// Mark space as FREE again and terminate
		if (person->state == LEAVING)
		{
			person->state = FREE;
			return;
		}

		// If person moved to other cell, mark space as OCCUPIED
		if (person->state == RESERVED)
		{
			person->state = OCCUPIED;
		}

		// Update position
		person->position = person->position + person->velocity * DELTA;

		// Update direction to goal
		float2 goalDir = make_float2(
			person->goal.x - person->position.x,
			person->goal.y - person->position.y);

		goalDir = normalize(goalDir);
		person->direction = goalDir;

		person->updateVelocity(goalDir * SPEED);
	}

	bool add_to_grid(const Person& p)
	{
		int cell = cellPosToIndex(p.position / CELL_SIZE);

		for (int i = 0; i < MAX_OCCUPATION; i++)
		{
			int index = cell * MAX_OCCUPATION + i;

			if (cells[index].state != FREE)
				continue;

			cells[index] = Person(p);
			return true;
		}

		return false;
	}

	void init()
	{
		cells = static_cast<Person*>(malloc(sizeof(Person) * CELLS_PER_AXIS * CELLS_PER_AXIS * MAX_OCCUPATION));
		for (int i = 0; i < CELLS_PER_AXIS * CELLS_PER_AXIS * MAX_OCCUPATION; i++)
		{
			cells[i] = Person();
		}

		int totallySpawned = 0;

		for (int i = 0; i < SPAWNED_ACTORS; i++)
		{
			bool spawned = false;
			while (!spawned)
			{
				spawned = add_to_grid(Person(getRandomPos(), getRandomPos()));
			}

			totallySpawned++;
		}

		std::cout << "Spawned " << totallySpawned << " people.\n";

		cudaError_t error = cudaMalloc((void**)&deviceCells, CELLS_PER_AXIS * CELLS_PER_AXIS * MAX_OCCUPATION * sizeof(Person));
		if (error)
			std::cout << "Error while allocating CUDA memory\n";

		std::cout << "Allocated " << CELLS_PER_AXIS * CELLS_PER_AXIS * MAX_OCCUPATION * sizeof(Person) / 1024 / 1024 << " MB on GPU\n";

		cudaMemcpy(deviceCells, cells, CELLS_PER_AXIS * CELLS_PER_AXIS * MAX_OCCUPATION * sizeof(Person), cudaMemcpyHostToDevice);
	}

	double timeH2D = 0.f;
	double timeD2H = 0.f;
	int transfersMeasured = 0;

	void simulate()
	{
		auto t1 = std::chrono::high_resolution_clock::now();
		cudaMemcpy(deviceCells, cells, CELLS_PER_AXIS * CELLS_PER_AXIS * MAX_OCCUPATION * sizeof(Person), cudaMemcpyHostToDevice);
		auto t2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> ms_double = t2 - t1;
		timeH2D += ms_double.count();

		calculateCellForce << < blocksPerGrid, threadsPerBlock >> > (deviceCells);

		cudaError_t error = cudaDeviceSynchronize();
		if (error)
		{
			std::cout << "CalculateForce: " << cudaGetErrorName << ": " << cudaGetErrorString(error) << "\n";
		}

		completeMove << < blocksPerGrid, MAX_OCCUPATION >> > (deviceCells);
		cudaDeviceSynchronize();

		error = cudaDeviceSynchronize();
		if (error)
		{
			std::cout << "CompleteMove: " << cudaGetErrorName << ": " << cudaGetErrorString(error) << "\n";
		}

		t1 = std::chrono::high_resolution_clock::now();
		cudaMemcpy(cells, deviceCells, CELLS_PER_AXIS * CELLS_PER_AXIS * MAX_OCCUPATION * sizeof(Person), cudaMemcpyDeviceToHost);
		t2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> ms_double2 = t2 - t1;
		timeD2H += ms_double2.count();

		transfersMeasured++;
	}

	void printTransferTime()
	{
		std::cout << "Avg. Host to Device = " << timeH2D / transfersMeasured << "\n";
		std::cout << "Avg. Device to Host = " << timeD2H / transfersMeasured << "\n";
	}

	std::vector<PersonVisuals> convertToVisual()
	{
		std::vector<PersonVisuals> persons;

		for (int i = 0; i < CELLS_PER_AXIS * CELLS_PER_AXIS * MAX_OCCUPATION; i++)
		{
			Person& p = cells[i];
			if (p.state != FREE)
			{
				if (dist(p.position, p.goal) < MIN_DIST)
				{
					p.goal = getRandomPos();
				}

				float2 dir = p.direction;
				dir.y = -dir.y;

				persons.emplace_back(simCoordToGL(p.position), dir);
			}
		}
		
		return persons;
	}
}