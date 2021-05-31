#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector_functions.h>
#include <vector_types.h>
#include <windows.h>

#include "SF_CUDA.cuh"
#include "SF_Sequential.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

//SF_Sequential sf;

float vertices[SPAWNED_ACTORS * 9]; //SPAWNED_ACTORS * 9
const float FPS = 25.f;
int maxIterations = 10000;

const float size = 0.008f;

// settings
const unsigned int SCR_WIDTH = 1440;
const unsigned int SCR_HEIGHT = 1080;

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
"color = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
"}\n\0";

bool construct_triangle(GLfloat* triangle, float2 pos, float2 dir)
{
	//std::cout << "(" << pos.x << "|" << pos.y << ") moving (" << dir.x << "|" << dir.y << ")\n";
	dir = normalizeH(dir);
	const float2 orth_dir = make_float2(dir.y, -dir.x);
	
	if(magnitudeH(dir) > 0.1f)
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

void printPositions(GLfloat* vertices, int size)
{
    std::cout << "\nPRINTING VERTICES\n";

    for (int person = 0; person < size / 3 / 3; person++)
    {
        for (int corner = 0; corner < 3; corner++)
        {
            for (int axis = 0; axis < 3; axis++)
            {
                std::cout << vertices[person * 9 + corner * 3 + axis] << " ";
            }
            std::cout << " | ";
        }

        std::cout << "\n";
    }
}

bool updateVisuals(bool debugPrint)
{
    std::vector<PersonVisuals> pv = convertToVisual(debugPrint);
	bool drewTri = false;
	
    for (int i = 0; i < pv.size(); i++)
    {
        PersonVisuals person = pv[i];

        GLfloat triangle[9];

    	/*drewTri =*/ construct_triangle(triangle, person.position, person.direction);

        for (int j = 0; j < 9; j++)
        {
            vertices[i * 9 + j] = triangle[j];
        }
    }

	//return drewTri;
	return true;
}

bool updateSimulation()
{
    simulate();
    if(!updateVisuals(false))
		return false;

	return true;
}

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "CrowdSim", NULL, NULL);
    if (window == NULL)
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

	// Shader
    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    unsigned int shaderProgram;
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glUseProgram(shaderProgram);
	
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

	// Bacck to vertices
    unsigned int VBO;
    glGenBuffers(1, &VBO);

	//TODO: Change init here!
	//init();
	initTest();
	updateVisuals(false);

    glBindBuffer(GL_ARRAY_BUFFER, 0); // Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound vertex buffer object so afterwards we can safely unbind
    glBindVertexArray(0); // Unbind VAO (it's always a good thing to unbind any buffer/array to prevent strange bugs)
    
    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
	int verticesCount = sizeof(vertices) / sizeof(GLfloat);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
	
    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);
        
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    	
        // render
        // ------
        glClearColor(.8f, .8f, .8f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw our first triangle
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);

        glDrawArrays(GL_TRIANGLES, 0, verticesCount);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();

    	Sleep(1000.f / FPS);
        
        if(!updateSimulation())
			break;

    	if(--maxIterations <= 0)
    	{
    		break;
    	}
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

bool simProcessed;

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
    	close();
        glfwSetWindowShouldClose(window, true);
    }

	if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS && !simProcessed)
	{
        updateSimulation();
		simProcessed = true;
		//printGrid();
	}

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_RELEASE)
    {
	    simProcessed = false;
    }

	if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
	{
		//hard_reset();
	}
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}