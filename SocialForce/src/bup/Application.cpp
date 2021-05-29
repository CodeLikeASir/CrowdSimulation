#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector_functions.h>
#include <vector_types.h>

#include "SF_Sequential.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

SF_Sequential sf;

float vertices[7 * 9];

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

const float TRIANGLE_SIZE = 0.1f;

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

float2 normalize(float2 vec)
{
	if(std::abs(vec.x) < 0.001f && std::abs(vec.y) < 0.001f)
		return vec;
	
    float mag = sqrtf(vec.x * vec.x + vec.y * vec.y);
    vec.x /= mag;
    vec.y /= mag;

    return vec;
}

void construct_triangle(GLfloat* triangle, float2 pos, float2 dir)
{
	dir = normalize(dir);
	const float2 orth_dir = make_float2(dir.y, -dir.x);
    /*
    triangle[0] = pos.x + TRIANGLE_SIZE * dir.x;
    triangle[1] = pos.y + TRIANGLE_SIZE * dir.y;
    triangle[2] = 0.f;
    
    triangle[3] = pos.x - TRIANGLE_SIZE * dir.x + TRIANGLE_SIZE * orth_dir.x;
    triangle[4] = pos.y - TRIANGLE_SIZE * dir.y + TRIANGLE_SIZE * orth_dir.y;
    triangle[5] = 0.f;

    triangle[6] = pos.x - TRIANGLE_SIZE * dir.x - TRIANGLE_SIZE * orth_dir.x;
    triangle[7] = pos.y - TRIANGLE_SIZE * dir.y - TRIANGLE_SIZE * orth_dir.y;
    triangle[8] = 0.f;
    */

	const float size = 0.02f;

    triangle[0] = pos.x;
    triangle[1] = pos.y + size;
    triangle[2] = 0.f;

    triangle[3] = pos.x - size;
    triangle[4] = pos.y - size;
    triangle[5] = 0.f;

    triangle[6] = pos.x + size;
    triangle[7] = pos.y - size;
    triangle[8] = 0.f;
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

void updateVisuals(bool debugPrint)
{
    std::vector<PersonVisuals> pv = sf.convertToVisual(debugPrint);
	
    for (int i = 0; i < pv.size(); i++)
    {
        PersonVisuals person = pv[i];

        GLfloat triangle[9];
        construct_triangle(triangle, person.position, person.direction);
        //std::cout << "Constructing tri for " << person.position.x << "|" << person.position.y << "\n";

        for (int j = 0; j < 9; j++)
        {
            vertices[i * 9 + j] = triangle[j];
        }
    }
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
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
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

    sf.init_test1();
	updateVisuals(false);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

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
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

void updateSimulation(int steps)
{
    sf.simulate(steps);
	updateVisuals(true);
}

bool simProcessed;

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

	if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS && !simProcessed)
	{
        updateSimulation(1);
		simProcessed = true;
	}

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_RELEASE)
    {
	    simProcessed = false;
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

