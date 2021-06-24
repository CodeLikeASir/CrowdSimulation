#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector_functions.h>
#include <vector_types.h>
#include <Windows.h>
#include <chrono>

#include "SF_CUDA.cuh"
#include "SF_Sequential.h"
#include "Math_Helper.cuh"

float vertices[SPAWNED_ACTORS * 9];

// window size in px
const unsigned int SCR_WIDTH = 1440;
const unsigned int SCR_HEIGHT = 1080;

// Size of actors in GL coordinates
const float size = 0.005f;

// Basic shader code
const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"}\0";

const GLchar* fragmentShaderSource = "#version 330 core\n"
"out vec4 color;\n"
"void main()\n"
"{\n"
"color = vec4(0.5f, 0.3f, 0.2f, 1.0f);\n"
"}\n\0";

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

// Creates triangles based on position and facing direction
bool construct_triangle(GLfloat* triangle, float2 pos, float2 dir)
{
	dir = normalize(dir);
	const float2 orth_dir = make_float2(dir.y, -dir.x);
	
	if(magnitude(dir) > 0.1f)
	{
		float2 dirSize = make_float2(1.5f * size * dir.x, 1.5f * size * dir.y);
        triangle[0] = pos.x + 1.2f * size * dir.x;
        triangle[1] = pos.y + 1.2f * size * dir.y;
        triangle[2] = 0.f;

        triangle[3] = pos.x - dirSize.x + size * orth_dir.x;
        triangle[4] = pos.y - dirSize.y + size * orth_dir.y;
        triangle[5] = 0.f;

        triangle[6] = pos.x - dirSize.x - size * orth_dir.x;
        triangle[7] = pos.y - dirSize.y - size * orth_dir.y;
        triangle[8] = 0.f;

		return true;
	}
	// If person reached their goal, they are drawn as up facing
	else
	{
        triangle[0] = pos.x;
        triangle[1] = pos.y + 2.f * size;
        triangle[2] = 0.f;

        triangle[3] = pos.x + size;
        triangle[4] = pos.y - size;
        triangle[5] = 0.f;

        triangle[6] = pos.x - size;
        triangle[7] = pos.y - size;
        triangle[8] = 0.f;

        return false;
	}
}

// Updates all triangle positions and directions based on current simulation
bool updateVisuals(std::vector<PersonVisuals> pv)
{
    bool didMove = false;
    for (int i = 0; i < pv.size(); i++)
    {
        PersonVisuals person = pv[i];

        GLfloat triangle[9];
        if (construct_triangle(triangle, person.position, person.direction))
            didMove = true;

        for (int j = 0; j < 9; j++)
        {
            vertices[i * 9 + j] = triangle[j];
        }
    }

    return didMove;
}

int main()
{
	double minTime = 1000.;
	double maxTime = 0.;
	double totalTime = 0.;
	int sampleCount = 0;
	
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "CrowdSim", nullptr, nullptr);
    if (window == nullptr)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

	unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glUseProgram(shaderProgram);
	
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    unsigned int VBO;
    glGenBuffers(1, &VBO);

	// Initializes crowd simulation
	if(USE_CUDA)
	{
		SF_CUDA::init();
        updateVisuals(SF_CUDA::convertToVisual());
	}
	else
	{
        SF_Sequential::init();
        updateVisuals(SF_Sequential::convertToVisual());
	}

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    
    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
	int verticesCount = sizeof(vertices) / sizeof(GLfloat);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)nullptr);
    glEnableVertexAttribArray(0);

    int remainingUpdates = 10000;
    // render + simulation loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    	
        // Render
        glClearColor(.8f, .8f, .8f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw our first triangle
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);

        glDrawArrays(GL_TRIANGLES, 0, verticesCount);
        
        glfwSwapBuffers(window);
        glfwPollEvents();

    	// Start next simulation step
        auto t1 = std::chrono::high_resolution_clock::now();

    	// Calculates crowd simulation
        if(USE_CUDA)
            SF_CUDA::simulate();
    	else
            SF_Sequential::simulate();
    	
        auto t2 = std::chrono::high_resolution_clock::now();

        // Getting number of milliseconds as a double.
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;

    	minTime = ms_double.count() < minTime ? ms_double.count() : minTime;
        maxTime = ms_double.count() > maxTime ? ms_double.count() : maxTime;
    	totalTime += ms_double.count();
    	sampleCount++;

    	// Updates visual representation
    	if(USE_CUDA)
			updateVisuals(SF_CUDA::convertToVisual());
    	else
			updateVisuals(SF_Sequential::convertToVisual());
        
        std::chrono::duration<double, std::milli> update_time = t2 - t1;
    	double remaining_time = (1000. / MAX_FPS) - update_time.count();

    	if(remaining_time > 0.)
			Sleep(remaining_time);
        
    	if(--remainingUpdates <= 0)
			break;
    }

	std::cout << "Min: " << minTime << " | Max: " << maxTime << "| Avg: " << (totalTime / sampleCount) << "\n";

    // Terminate, clearing all previously allocated GLFW resources.
    glfwTerminate();
    return 0;
}