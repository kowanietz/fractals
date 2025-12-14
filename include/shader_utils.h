#ifndef SHADER_UTILS_H
#define SHADER_UTILS_H

#include <glad/glad.h>

GLuint compileShader(GLenum type, const char *src);
GLuint createProgram(const char *vsSrc, const char *fsSrc);

#endif // SHADER_UTILS_H

