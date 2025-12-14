#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include "shader_utils.h"
#include "args_parser.h"

#include <vector>
#include <iostream>
#include <cmath>
#include <cstring>

double juliaRe = -0.715; // wtf
double juliaIm = 0.257; // TODO: add flag to customize julia constants

int getMaxIter(const double scale, const int baseIter) {
    return static_cast<int>(baseIter + 50 * std::log10(3.0 / scale));
}

constexpr int WIDTH = 1600;
constexpr int HEIGHT = 1200;

double centerX = -0.5;
double centerY = 0.0;
double scale = 3.0;

// the cuda kernels
void mandelbrot_cuda(uint8_t *pixels_d, int width, int height,
                     double centerX, double centerY, double scale, int maxIter);

void julia_cuda(uint8_t *pixels_d, int width, int height,
                double centerX, double centerY, double scale, int maxIter,
                double juliaRe, double juliaIm);

void burningship_cuda(uint8_t *pixels_d, int width, int height,
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

// zoom handler
void scroll_callback(GLFWwindow *window, const double x_offset, const double y_offset) {
    double mx, my;
    glfwGetCursorPos(window, &mx, &my);
    const double relX = (mx - WIDTH / 2.0) * scale / WIDTH;
    const double relY = (-my + HEIGHT / 2.0) * scale / WIDTH;
    if (y_offset > 0) scale *= 0.8;
    else scale *= 1.25;
    const double factor = (y_offset > 0) ? 0.8 : 1.25;
    centerX += relX * (1.0 - factor);
    centerY += relY * (1.0 - factor);
}

int main(const int argc, char *argv[]) {
    Config config = parseArgs(argc, argv);

    std::cout << "Base iterations: " << config.baseIter << "\n";
    std::cout << "Fractal type: ";
    switch (config.fractalType) {
        case FractalType::JULIA: std::cout << "Julia";
            break;
        case FractalType::MANDELBROT: std::cout << "Mandelbrot";
            break;
        case FractalType::BURNING_SHIP: std::cout << "Burning Ship";
            break;
    }
    std::cout << "\n";

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    const char *windowTitle = "Fractal CUDA";
    if (config.fractalType == FractalType::JULIA) windowTitle = "Julia CUDA";
    else if (config.fractalType == FractalType::MANDELBROT) windowTitle = "Mandelbrot CUDA";
    else if (config.fractalType == FractalType::BURNING_SHIP) windowTitle = "Burning Ship CUDA";

    GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, windowTitle, nullptr, nullptr);
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
        int MAX_ITER = getMaxIter(scale, config.baseIter);

        if (glfwGetKey(window,GLFW_KEY_LEFT) == GLFW_PRESS) centerX -= panSpeed;
        if (glfwGetKey(window,GLFW_KEY_RIGHT) == GLFW_PRESS) centerX += panSpeed;
        if (glfwGetKey(window,GLFW_KEY_UP) == GLFW_PRESS) centerY += panSpeed;
        if (glfwGetKey(window,GLFW_KEY_DOWN) == GLFW_PRESS) centerY -= panSpeed;


        if (config.fractalType == FractalType::JULIA) {
            julia_cuda(d_pixels, WIDTH, HEIGHT, centerX, centerY, scale, MAX_ITER, juliaRe, juliaIm);
        } else if (config.fractalType == FractalType::BURNING_SHIP) {
            burningship_cuda(d_pixels, WIDTH, HEIGHT, centerX, centerY, scale, MAX_ITER);
        } else {
            mandelbrot_cuda(d_pixels, WIDTH, HEIGHT, centerX, centerY, scale, MAX_ITER);
        }
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
