#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <cmath>

int getMaxIter(const double scale) {
    constexpr int baseIter = 500;

    return static_cast<int>(baseIter + 50 * std::log10(3.0 / scale));
}


constexpr int WIDTH = 1600;
constexpr int HEIGHT = 1200;

double centerX = -0.5;
double centerY = 0.0;
double scale = 3.0;

int MAX_ITER = getMaxIter(scale);


// the cuda kernel
void mandelbrot_cuda(uint8_t *pixels_d, int width, int height,
                     double centerX, double centerY, double scale, int maxIter);

// OpenGL stuff
auto vertexShaderSrc = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    TexCoord = aTexCoord;
    gl_Position = vec4(aPos,0,1);
})";

auto fragmentShaderSrc = R"(
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D screenTexture;
void main() {
    FragColor = texture(screenTexture, TexCoord);
})";

GLuint compileShader(const GLenum type, const char *src) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info[512];
        glGetShaderInfoLog(shader, 512, nullptr, info);
        std::cerr << "Shader compile error: " << info << "\n";
    }
    return shader;
}

GLuint createProgram(const char *vsSrc, const char *fsSrc) {
    const GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
    const GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
    const GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// zoom handler
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
    double mx, my;
    glfwGetCursorPos(window, &mx, &my);
    const double relX = (mx - WIDTH / 2.0) * scale / WIDTH;
    const double relY = (-my + HEIGHT / 2.0) * scale / WIDTH;
    if (yoffset > 0) scale *= 0.8;
    else scale *= 1.25;
    const double factor = (yoffset > 0) ? 0.8 : 1.25;
    centerX += relX * (1.0 - factor);
    centerY += relY * (1.0 - factor);
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "Mandelbrot CUDA", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::cerr << "Failed to initialize GLAD\n";
        return -1;
    }

    glfwSetScrollCallback(window, scroll_callback);

    constexpr float vertices[] = {
        -1, -1, 0, 0,
        1, -1, 1, 0,
        -1, 1, 0, 1,
        1, 1, 1, 1
    };

    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2,GL_FLOAT,GL_FALSE, 4 * sizeof(float), static_cast<void *>(nullptr));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2,GL_FLOAT,GL_FALSE, 4 * sizeof(float), reinterpret_cast<void *>(2 * sizeof(float)));

    const GLuint shaderProgram = createProgram(vertexShaderSrc, fragmentShaderSrc);

    std::vector<uint8_t> pixels(WIDTH * HEIGHT * 4);
    uint8_t *d_pixels;
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA8, WIDTH, HEIGHT, 0,GL_RGBA,GL_UNSIGNED_BYTE, pixels.data());
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);

    while (!glfwWindowShouldClose(window)) {
        const double panSpeed = 0.02 * scale;
        MAX_ITER = getMaxIter(scale);

        if (glfwGetKey(window,GLFW_KEY_LEFT) == GLFW_PRESS) centerX -= panSpeed;
        if (glfwGetKey(window,GLFW_KEY_RIGHT) == GLFW_PRESS) centerX += panSpeed;
        if (glfwGetKey(window,GLFW_KEY_UP) == GLFW_PRESS) centerY += panSpeed;
        if (glfwGetKey(window,GLFW_KEY_DOWN) == GLFW_PRESS) centerY -= panSpeed;


        mandelbrot_cuda(d_pixels, WIDTH, HEIGHT, centerX, centerY, scale, MAX_ITER);
        cudaMemcpy(pixels.data(), d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);

        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT,GL_RGBA,GL_UNSIGNED_BYTE, pixels.data());

        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glBindTexture(GL_TEXTURE_2D, tex);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaFree(d_pixels);
    glfwTerminate();
}
