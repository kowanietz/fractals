#include "args_parser.h"
#include <iostream>
#include <cstring>

Config parseArgs(const int argc, char *argv[]) {
    Config config;
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--iterations") == 0 || strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            config.baseIter = std::atoi(argv[++i]);
            if (config.baseIter <= 0) {
                std::cerr << "Invalid iteration count. Using default (500).\n";
                config.baseIter = 500;
            }
        } else if ((strcmp(argv[i], "--fractal") == 0 || strcmp(argv[i], "-f") == 0) && i + 1 < argc) {
            if (const char *type = argv[++i]; strcmp(type, "julia") == 0) {
                config.fractalType = FractalType::JULIA;
            } else if (strcmp(type, "mandelbrot") == 0) {
                config.fractalType = FractalType::MANDELBROT;
            } else if (strcmp(type, "burningship") == 0) {
                config.fractalType = FractalType::BURNING_SHIP;
            } else {
                std::cerr << "Unknown fractal type. Using mandelbrot.\n";
            }
        }
    }
    return config;
}
